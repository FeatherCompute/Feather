using System.Reflection;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Feather.Math;
using Feather.RenderGraph;
using Feather.Resources;

namespace Feather.Blender.RenderHost;

internal sealed class ProjectPassAssemblyManager : IDisposable
{
    private PassAssemblyGeneration? current;
    private bool disposed;

    internal WeakReference? LastUnloadedContextForTesting { get; private set; }

    public ProjectPassExecutionResult Execute(
        string manifestPath,
        RenderGraphExecution graph,
        RenderSceneResources scene,
        int width,
        int height,
        float4x4 viewProjection,
        RenderViewState viewState)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        var manifest = ProjectPassManifest.Load(manifestPath);
        manifest.ValidateGraph(graph);
        var reloaded = current is null || !current.Matches(manifest);
        if (reloaded)
        {
            var replacement = PassAssemblyGeneration.Load(manifest);
            var previous = current;
            current = replacement;
            if (previous is not null)
            {
                LastUnloadedContextForTesting = previous.UnloadReference;
                previous.Dispose();
            }
        }

        // Each View remembers the assembly build it last executed. A reload may happen while a
        // different View is active, so checking every execution prevents stale per-View history.
        viewState.PreparePassBuild(manifest.BuildId, graph);

        return current!.Execute(graph, scene, width, height, viewProjection, reloaded, viewState);
    }

    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        if (current is not null)
        {
            LastUnloadedContextForTesting = current.UnloadReference;
            current.Dispose();
            current = null;
        }
        disposed = true;
    }
}

internal sealed record ProjectPassExecutionResult(
    RenderedFrame Frame,
    string BuildId,
    string PassType,
    int PassCount,
    bool Reloaded,
    double GpuReadbackMilliseconds);

internal sealed class PassAssemblyGeneration : IDisposable
{
    private readonly ProjectPassLoadContext loadContext;
    private readonly Assembly assembly;
    private readonly ProjectPassManifest manifest;
    private readonly Dictionary<string, Type> passTypes;

    // Owned here rather than by the per-execution backend so GPU-resident history survives across
    // frames instead of being reallocated and cleared every render.
    private readonly GraphTexturePool texturePool = new();
    private bool disposed;

    private PassAssemblyGeneration(
        ProjectPassLoadContext loadContext,
        Assembly assembly,
        ProjectPassManifest manifest,
        Dictionary<string, Type> passTypes)
    {
        this.loadContext = loadContext;
        this.assembly = assembly;
        this.manifest = manifest;
        this.passTypes = passTypes;
        UnloadReference = new WeakReference(loadContext, trackResurrection: false);
    }

    public WeakReference UnloadReference { get; }

    public static PassAssemblyGeneration Load(ProjectPassManifest manifest)
    {
        var assemblyBytes = ReadFileShared(manifest.AssemblyPath);
        manifest.ValidateBuildId(assemblyBytes);
        var loadContext = new ProjectPassLoadContext(manifest.AssemblyPath);
        try
        {
            using var assemblyStream = new MemoryStream(assemblyBytes, writable: false);
            var pdbPath = Path.ChangeExtension(manifest.AssemblyPath, ".pdb");
            Assembly assembly;
            if (File.Exists(pdbPath))
            {
                using var pdbStream = new MemoryStream(ReadFileShared(pdbPath), writable: false);
                assembly = loadContext.LoadFromStream(assemblyStream, pdbStream);
            }
            else
            {
                assembly = loadContext.LoadFromStream(assemblyStream);
            }

            var passTypes = new Dictionary<string, Type>(StringComparer.OrdinalIgnoreCase);
            foreach (var definition in manifest.Passes)
            {
                var type = assembly.GetType(definition.TypeName, throwOnError: false, ignoreCase: false)
                    ?? throw new InvalidDataException(
                        $"Pass type '{definition.TypeName}' is not present in '{manifest.AssemblyPath}'.");
                if (!type.IsClass || type.IsAbstract || type.ContainsGenericParameters ||
                    !typeof(IRenderPass).IsAssignableFrom(type))
                {
                    throw new InvalidDataException(
                        $"Pass type '{definition.TypeName}' must be a concrete IRenderPass class.");
                }
                if (type.GetConstructor(Type.EmptyTypes) is null)
                {
                    throw new InvalidDataException(
                        $"Pass type '{definition.TypeName}' must have a public parameterless constructor.");
                }

                var attribute = type.GetCustomAttribute<FeatherPassAttribute>(inherit: false)
                    ?? throw new InvalidDataException(
                        $"Pass type '{definition.TypeName}' has no FeatherPass attribute.");
                if (!Guid.TryParseExact(attribute.Guid, "D", out var attributeGuid) ||
                    !string.Equals(
                        attributeGuid.ToString("D"),
                        definition.PassGuid,
                        StringComparison.OrdinalIgnoreCase))
                {
                    throw new InvalidDataException(
                        $"Pass type '{definition.TypeName}' does not declare manifest GUID {definition.PassGuid}.");
                }
                passTypes.Add(definition.PassGuid, type);
            }

            return new PassAssemblyGeneration(loadContext, assembly, manifest, passTypes);
        }
        catch
        {
            loadContext.Unload();
            throw;
        }
    }

    public bool Matches(ProjectPassManifest candidate)
        => string.Equals(manifest.Path, candidate.Path, PathComparison) &&
           string.Equals(manifest.BuildId, candidate.BuildId, StringComparison.Ordinal) &&
           string.Equals(manifest.ContentIdentity, candidate.ContentIdentity, StringComparison.Ordinal);

    public ProjectPassExecutionResult Execute(
        RenderGraphExecution graph,
        RenderSceneResources scene,
        int width,
        int height,
        float4x4 viewProjection,
        bool reloaded,
        RenderViewState viewState)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        var resources = new GraphResourceResolver(graph, manifest);
        // Socket identities are only comparable within one topology, so a changed graph has to
        // start from an empty pool.
        texturePool.PrepareForGraph(graph.GraphFingerprint);
        var backend = new ProjectRenderContextBackend(
            scene,
            width,
            height,
            graph.SampleCount,
            viewProjection,
            resources,
            viewState.History,
            texturePool);
        var executedTypes = new List<string>(graph.Passes.Length);
        foreach (var passNode in graph.Passes)
        {
            if (passNode.Muted)
            {
                continue;
            }

            var definition = manifest.DefinitionFor(passNode);
            var type = passTypes[definition.PassGuid];
            var instance = (IRenderPass)(Activator.CreateInstance(type)
                ?? throw new InvalidDataException($"Unable to create pass type '{definition.TypeName}'."));
            try
            {
                PassMemberBinder.BindResources(instance, passNode, definition, resources);
                PassMemberBinder.BindParameters(instance, passNode.Parameters);
                instance.Execute(new RenderContext(backend));
                executedTypes.Add(definition.TypeName);
            }
            finally
            {
                (instance as IDisposable)?.Dispose();
            }
        }

        var finalHandle = resources.ResolveTextureSource(
            graph.OutputLink.FromNode,
            graph.OutputLink.FromSocket);
        var frame = backend.TakeFrame(finalHandle);
        var historyUpdates = backend.CaptureHistory();
        viewState.CommitHistory(historyUpdates);
        return new ProjectPassExecutionResult(
            frame,
            manifest.BuildId,
            executedTypes.LastOrDefault() ?? "bypass",
            executedTypes.Count,
            reloaded,
            backend.GpuReadbackMilliseconds);
    }

    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        texturePool.Dispose();
        passTypes.Clear();
        loadContext.Unload();
        disposed = true;
    }

    private static byte[] ReadFileShared(string path)
    {
        using var stream = new FileStream(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.ReadWrite | FileShare.Delete,
            81920,
            FileOptions.SequentialScan);
        if (stream.Length > int.MaxValue)
        {
            throw new InvalidDataException($"Project artifact is too large to load: {path}");
        }
        var bytes = new byte[checked((int)stream.Length)];
        stream.ReadExactly(bytes);
        return bytes;
    }

    private static StringComparison PathComparison
        => OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
}

internal sealed class ProjectPassLoadContext : AssemblyLoadContext
{
    private static readonly Assembly FeatherAssembly = typeof(IRenderPass).Assembly;
    private static readonly Assembly FeatherNativeAssembly = typeof(Feather.Native.FeBufferHandle).Assembly;
    private readonly AssemblyDependencyResolver resolver;

    public ProjectPassLoadContext(string mainAssemblyPath)
        : base($"FeatherPass:{Path.GetFileNameWithoutExtension(mainAssemblyPath)}:{Guid.NewGuid():N}", isCollectible: true)
    {
        resolver = new AssemblyDependencyResolver(mainAssemblyPath);
    }

    protected override Assembly? Load(AssemblyName assemblyName)
    {
        if (AssemblyName.ReferenceMatchesDefinition(assemblyName, FeatherAssembly.GetName()))
        {
            return FeatherAssembly;
        }
        if (AssemblyName.ReferenceMatchesDefinition(assemblyName, FeatherNativeAssembly.GetName()))
        {
            return FeatherNativeAssembly;
        }

        var path = resolver.ResolveAssemblyToPath(assemblyName);
        return path is null ? null : LoadFromAssemblyPath(path);
    }

    protected override nint LoadUnmanagedDll(string unmanagedDllName)
    {
        var path = resolver.ResolveUnmanagedDllToPath(unmanagedDllName);
        return path is null ? nint.Zero : LoadUnmanagedDllFromPath(path);
    }
}

internal sealed class ProjectPassManifest
{
    private static readonly JsonSerializerOptions IndentedJson = new() { WriteIndented = true };

    private ProjectPassManifest(
        string path,
        string buildId,
        string assemblyPath,
        ProjectPassDefinition[] passes,
        JsonObject hashDocument)
    {
        Path = path;
        BuildId = buildId;
        AssemblyPath = assemblyPath;
        Passes = passes;
        HashDocument = hashDocument;
        ContentIdentity = Convert.ToHexString(SHA256.HashData(
            Encoding.UTF8.GetBytes(hashDocument.ToJsonString(IndentedJson))));
    }

    public string Path { get; }
    public string BuildId { get; }
    public string ContentIdentity { get; }
    public string AssemblyPath { get; }
    public ProjectPassDefinition[] Passes { get; }
    private JsonObject HashDocument { get; }

    public static ProjectPassManifest Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        path = System.IO.Path.GetFullPath(path);
        JsonObject root;
        try
        {
            using var stream = new FileStream(
                path,
                FileMode.Open,
                FileAccess.Read,
                FileShare.ReadWrite | FileShare.Delete);
            root = JsonNode.Parse(stream)?.AsObject()
                ?? throw new InvalidDataException("Pass manifest contains null.");
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException($"Pass manifest JSON is invalid: {exception.Message}", exception);
        }

        var schemaVersion = RequiredInt32(root, "schemaVersion");
        if (schemaVersion != 1)
        {
            throw new InvalidDataException($"Unsupported pass manifest schema version: {schemaVersion}.");
        }
        var buildId = RequiredString(root, "buildId");
        if (buildId.Length != 71 || !buildId.StartsWith("sha256:", StringComparison.Ordinal) ||
            buildId.Skip(7).Any(static character => character is not (
                >= '0' and <= '9' or >= 'a' and <= 'f')))
        {
            throw new InvalidDataException("Pass manifest buildId must be a lowercase SHA-256 identifier.");
        }

        var manifestDirectory = System.IO.Path.GetDirectoryName(path)
            ?? throw new InvalidDataException("Pass manifest has no parent directory.");
        var projectRootValue = OptionalString(root, "projectRoot");
        var projectRoot = projectRootValue is null
            ? FindLegacyProjectRoot(manifestDirectory)
            : ResolvePath(manifestDirectory, projectRootValue);
        var rootAssemblyPath = RequiredString(root, "assemblyPath");
        var assemblyPath = ResolvePath(projectRoot, rootAssemblyPath);

        if (root["passes"] is not JsonArray passArray || passArray.Count == 0)
        {
            throw new InvalidDataException("Pass manifest must contain at least one pass.");
        }
        var passes = new List<ProjectPassDefinition>(passArray.Count);
        var passGuids = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var node in passArray)
        {
            var pass = node?.AsObject()
                ?? throw new InvalidDataException("Pass manifest entries must be objects.");
            var guidValue = RequiredString(pass, "passGuid");
            if (!Guid.TryParseExact(guidValue, "D", out var guid))
            {
                throw new InvalidDataException($"Pass manifest GUID '{guidValue}' is invalid.");
            }
            var normalizedGuid = guid.ToString("D");
            if (!passGuids.Add(normalizedGuid))
            {
                throw new InvalidDataException($"Pass manifest contains duplicate GUID {normalizedGuid}.");
            }
            var passAssembly = OptionalString(pass, "assemblyPath") ?? rootAssemblyPath;
            if (!string.Equals(
                    ResolvePath(projectRoot, passAssembly),
                    assemblyPath,
                    OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal))
            {
                throw new InvalidDataException("A single manifest build must use one project pass assembly.");
            }
            passes.Add(new ProjectPassDefinition(
                normalizedGuid,
                RequiredString(pass, "typeName"),
                ReadSocketDefinitions(pass, "inputs"),
                ReadSocketDefinitions(pass, "outputs")));
        }

        return new ProjectPassManifest(path, buildId, assemblyPath, passes.ToArray(), root);
    }

    public void ValidateBuildId(ReadOnlySpan<byte> assemblyBytes)
    {
        var hashDocument = (JsonObject)HashDocument.DeepClone();
        hashDocument["buildId"] = string.Empty;
        var hashInput = hashDocument.ToJsonString(IndentedJson);
        using var hasher = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        hasher.AppendData(Encoding.UTF8.GetBytes(hashInput));
        hasher.AppendData([0]);
        hasher.AppendData(assemblyBytes);
        var actual = "sha256:" + Convert.ToHexString(hasher.GetHashAndReset()).ToLowerInvariant();
        if (!string.Equals(actual, BuildId, StringComparison.Ordinal))
        {
            throw new IOException(
                $"Pass manifest buildId does not match assembly '{AssemblyPath}'. The build may still be publishing.");
        }
    }

    public void ValidateGraph(RenderGraphExecution graph)
    {
        foreach (var passNode in graph.Passes)
        {
            _ = DefinitionFor(passNode);
        }
        _ = new GraphResourceResolver(graph, this);
    }

    public ProjectPassDefinition DefinitionFor(GraphNode passNode)
    {
        var definition = Passes.SingleOrDefault(pass =>
            string.Equals(pass.PassGuid, passNode.PassGuid, StringComparison.OrdinalIgnoreCase))
            ?? throw new InvalidDataException(
                $"Manifest build '{BuildId}' does not define pass GUID {passNode.PassGuid}.");
        if (!string.Equals(definition.TypeName, passNode.TypeName, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Graph pass type '{passNode.TypeName}' does not match manifest type '{definition.TypeName}'. " +
                "Refresh the Feather pass nodes after building.");
        }
        return definition;
    }

    private static string FindLegacyProjectRoot(string manifestDirectory)
    {
        for (var directory = new DirectoryInfo(manifestDirectory);
             directory is not null;
             directory = directory.Parent)
        {
            if (File.Exists(System.IO.Path.Combine(directory.FullName, ".feather", "project.json")))
            {
                return directory.FullName;
            }
        }

        var info = new DirectoryInfo(manifestDirectory);
        return string.Equals(info.Name, "Generated", StringComparison.OrdinalIgnoreCase) && info.Parent is not null
            ? info.Parent.FullName
            : manifestDirectory;
    }

    private static string ResolvePath(string baseDirectory, string value)
        => System.IO.Path.GetFullPath(
            System.IO.Path.IsPathRooted(value) ? value : System.IO.Path.Combine(baseDirectory, value));

    private static string RequiredString(JsonObject value, string name)
        => OptionalString(value, name) is { } result
            ? result
            : throw new InvalidDataException($"Pass manifest {name} is required.");

    private static string? OptionalString(JsonObject value, string name)
        => value[name] is JsonValue item && item.TryGetValue<string>(out var result) && !string.IsNullOrWhiteSpace(result)
            ? result
            : null;

    private static int RequiredInt32(JsonObject value, string name)
        => value[name] is JsonValue item && item.TryGetValue<int>(out var result)
            ? result
            : throw new InvalidDataException($"Pass manifest {name} must be an integer.");

    private static ProjectPassSocketDefinition[] ReadSocketDefinitions(JsonObject pass, string name)
    {
        if (pass[name] is not JsonArray sockets)
        {
            throw new InvalidDataException($"Pass manifest {name} must be an array.");
        }

        var result = new List<ProjectPassSocketDefinition>(sockets.Count);
        var unique = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var node in sockets)
        {
            var socket = node?.AsObject()
                ?? throw new InvalidDataException($"Pass manifest {name} entries must be objects.");
            var value = RequiredString(socket, "socketGuid");
            if (!Guid.TryParseExact(value, "D", out var guid))
            {
                throw new InvalidDataException($"Pass manifest socket GUID '{value}' is invalid.");
            }
            var normalized = guid.ToString("D");
            if (!unique.Add(normalized))
            {
                throw new InvalidDataException($"Pass manifest contains duplicate socket GUID {normalized}.");
            }
            var resourceKind = OptionalString(socket, "resourceKind") ?? InferResourceKind(normalized);
            var format = OptionalString(socket, "format") ?? "Unknown";
            var elementType = BufferElementTypeNames.NormalizeManifest(
                OptionalString(socket, "elementType") ?? "Unknown");
            if (string.Equals(resourceKind, "Buffer", StringComparison.OrdinalIgnoreCase))
            {
                if (!string.Equals(format, "Unknown", StringComparison.OrdinalIgnoreCase))
                {
                    throw new InvalidDataException(
                        $"Pass manifest Buffer socket {normalized} cannot declare texture format {format}.");
                }
            }
            else if (!string.Equals(elementType, "Unknown", StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidDataException(
                    $"Pass manifest {resourceKind} socket {normalized} cannot declare Buffer element type {elementType}.");
            }
            result.Add(new ProjectPassSocketDefinition(
                normalized, resourceKind, format, elementType));
        }
        return result.ToArray();
    }

    private static string InferResourceKind(string socketGuid)
        => socketGuid switch
        {
            RenderGraphDocument.GeometryInputSocketGuid => "SceneGeometry",
            RenderGraphDocument.MaterialsInputSocketGuid => "MaterialTable",
            RenderGraphDocument.CameraInputSocketGuid => "Camera",
            RenderGraphDocument.ColorOutputSocketGuid => "Texture2D",
            _ => throw new InvalidDataException(
                $"Pass manifest socket {socketGuid} has no resourceKind.")
        };
}

internal sealed record ProjectPassDefinition(
    string PassGuid,
    string TypeName,
    ProjectPassSocketDefinition[] Inputs,
    ProjectPassSocketDefinition[] Outputs)
{
    public ProjectPassSocketDefinition Input(string socketGuid)
        => Inputs.SingleOrDefault(socket =>
            string.Equals(socket.SocketGuid, socketGuid, StringComparison.OrdinalIgnoreCase))
            ?? throw new InvalidDataException(
                $"Pass {TypeName} does not define input socket {socketGuid}.");

    public ProjectPassSocketDefinition Output(string socketGuid)
        => Outputs.SingleOrDefault(socket =>
            string.Equals(socket.SocketGuid, socketGuid, StringComparison.OrdinalIgnoreCase))
            ?? throw new InvalidDataException(
                $"Pass {TypeName} does not define output socket {socketGuid}.");
}

internal sealed record ProjectPassSocketDefinition(
    string SocketGuid,
    string ResourceKind,
    string Format,
    string ElementType);

internal sealed class GraphResourceResolver
{
    private const ulong FirstTextureHandle = 1024;
    private const ulong FirstBufferHandle = 1UL << 32;

    private readonly RenderGraphExecution graph;
    private readonly ProjectPassManifest manifest;
    private readonly Dictionary<(string NodeId, string SocketGuid), TextureHandle> textureHandles = new();
    private readonly Dictionary<ulong, string> textureIdentities = [];
    private readonly HashSet<ulong> writableTextureHandles = [];
    private readonly Dictionary<(string NodeId, string SocketGuid), BufferHandle> bufferHandles = new();
    private readonly Dictionary<ulong, string> writableBufferElementTypes = [];
    private readonly Dictionary<string, TextureHandle> historyReadHandles = new(StringComparer.Ordinal);
    private readonly Dictionary<string, TextureHandle> historyWriteSources = new(StringComparer.Ordinal);
    private ulong nextTextureHandle = FirstTextureHandle;
    private ulong nextBufferHandle = FirstBufferHandle;

    public GraphResourceResolver(RenderGraphExecution graph, ProjectPassManifest manifest)
    {
        this.graph = graph;
        this.manifest = manifest;

        foreach (var historyRead in graph.HistoryReads)
        {
            var handle = new TextureHandle(nextTextureHandle++);
            historyReadHandles.Add(historyRead.HistoryKey, handle);
            textureIdentities.Add(handle.Value, $"history|{historyRead.HistoryKey}");
        }

        foreach (var passNode in graph.Passes)
        {
            var definition = manifest.DefinitionFor(passNode);
            foreach (var input in definition.Inputs)
            {
                _ = ResolveInput(passNode, input);
            }
            foreach (var link in graph.Links.Where(link =>
                         string.Equals(link.FromNode, passNode.NodeId, StringComparison.Ordinal)))
            {
                _ = definition.Output(link.FromSocket);
            }
        }

        foreach (var link in graph.Links)
        {
            var target = graph.Nodes.Single(node =>
                string.Equals(node.NodeId, link.ToNode, StringComparison.Ordinal));
            var source = ResolveSource(link.FromNode, link.FromSocket);
            if (target.Kind == "pass")
            {
                var targetSocket = manifest.DefinitionFor(target).Input(link.ToSocket);
                RequireCompatible(source, targetSocket, link);
            }
            else if (target.Kind == "output" && !IsTexture(source.ResourceKind))
            {
                throw new InvalidDataException(
                    $"Graph output '{target.NodeId}.{link.ToSocket}' requires a Texture2D resource.");
            }
            else if (target.Kind == "history-write")
            {
                if (!IsTexture(source.ResourceKind) || source.Handle is not TextureHandle texture)
                {
                    throw new InvalidDataException(
                        $"History Write '{target.HistoryKey}' requires a Texture2D resource.");
                }
                historyWriteSources.Add(target.HistoryKey, texture);
            }
        }

        _ = ResolveTextureSource(graph.OutputLink.FromNode, graph.OutputLink.FromSocket);
    }

    public object ResolveInputHandle(GraphNode passNode, ProjectPassSocketDefinition input)
        => ResolveInput(passNode, input).Handle;

    public object ResolveOutputHandle(GraphNode passNode, ProjectPassSocketDefinition output)
    {
        if (passNode.Muted)
        {
            return ResolveSource(passNode.NodeId, output.SocketGuid).Handle;
        }
        if (IsTexture(output.ResourceKind))
        {
            return TextureHandleFor(passNode.NodeId, output.SocketGuid);
        }
        if (IsBuffer(output.ResourceKind))
        {
            return BufferHandleFor(passNode.NodeId, output.SocketGuid, output.ElementType);
        }
        throw new InvalidDataException(
            $"Pass output resource kind '{output.ResourceKind}' is not supported yet.");
    }

    public TextureHandle ResolveTextureSource(string nodeId, string socketGuid)
    {
        var resource = ResolveSource(nodeId, socketGuid);
        if (!IsTexture(resource.ResourceKind) || resource.Handle is not TextureHandle texture)
        {
            throw new InvalidDataException(
                $"Graph resource '{nodeId}.{socketGuid}' is not a Texture2D.");
        }
        return texture;
    }

    public bool IsWritable(TextureHandle handle)
        => writableTextureHandles.Contains(handle.Value);

    /// <summary>
    /// Returns the graph-stable identity of a texture handle. Handle values are assigned in
    /// resolution order and so are only meaningful within one execution; the node and socket pair
    /// is stable across executions and is what a cross-frame texture pool must key on.
    /// </summary>
    public string TextureIdentity(TextureHandle handle)
        => textureIdentities.TryGetValue(handle.Value, out var identity)
            ? identity
            : throw new KeyNotFoundException($"Unknown texture handle {handle.Value}.");

    public bool IsWritable(BufferHandle handle)
        => writableBufferElementTypes.ContainsKey(handle.Value);

    public string BufferElementType(BufferHandle handle)
        => writableBufferElementTypes.TryGetValue(handle.Value, out var elementType)
            ? elementType
            : throw new KeyNotFoundException($"Unknown buffer output handle {handle.Value}.");

    public IReadOnlyDictionary<string, TextureHandle> HistoryReadHandles => historyReadHandles;

    public IReadOnlyDictionary<string, TextureHandle> HistoryWriteSources => historyWriteSources;

    private ResolvedGraphResource ResolveInput(
        GraphNode passNode,
        ProjectPassSocketDefinition input)
    {
        var link = graph.IncomingLink(passNode.NodeId, input.SocketGuid)
            ?? throw new InvalidDataException(
                $"Pass '{passNode.TypeName}' input {input.SocketGuid} is not connected.");
        var source = ResolveSource(link.FromNode, link.FromSocket);
        RequireCompatible(source, input, link);
        return source;
    }

    private ResolvedGraphResource ResolveSource(string nodeId, string socketGuid)
    {
        var node = graph.Nodes.Single(item =>
            string.Equals(item.NodeId, nodeId, StringComparison.Ordinal));
        if (node.Kind == "scene")
        {
            return SceneResource(socketGuid);
        }
        if (node.Kind == "history-read")
        {
            if (!string.Equals(
                    socketGuid,
                    RenderGraphDocument.HistoryReadSocketGuid,
                    StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidDataException(
                    $"History Read '{node.HistoryKey}' has no output socket {socketGuid}.");
            }
            return new ResolvedGraphResource(
                "Texture2D",
                "Unknown",
                "Unknown",
                historyReadHandles[node.HistoryKey]);
        }
        if (node.Kind != "pass")
        {
            throw new InvalidDataException(
                $"Graph node '{nodeId}' cannot produce resource socket {socketGuid}.");
        }

        var definition = manifest.DefinitionFor(node);
        var output = definition.Output(socketGuid);
        if (!node.Muted)
        {
            object handle;
            if (IsTexture(output.ResourceKind))
            {
                handle = TextureHandleFor(node.NodeId, output.SocketGuid);
            }
            else if (IsBuffer(output.ResourceKind))
            {
                handle = BufferHandleFor(node.NodeId, output.SocketGuid, output.ElementType);
            }
            else
            {
                throw new InvalidDataException(
                    $"Pass output resource kind '{output.ResourceKind}' is not supported yet.");
            }
            return new ResolvedGraphResource(
                output.ResourceKind,
                output.Format,
                output.ElementType,
                handle);
        }

        var bypassInputs = definition.Inputs
            .Where(input => SocketMetadataCompatible(input, output) &&
                            graph.IncomingLink(node.NodeId, input.SocketGuid) is not null)
            .ToArray();
        if (bypassInputs.Length != 1)
        {
            throw new InvalidDataException(
                $"Muted pass '{node.TypeName}' can be bypassed only with one connected " +
                $"{output.ResourceKind} input.");
        }
        var inputLink = graph.IncomingLink(node.NodeId, bypassInputs[0].SocketGuid)!;
        var source = ResolveSource(inputLink.FromNode, inputLink.FromSocket);
        RequireCompatible(source, output, inputLink);
        return source;
    }

    private TextureHandle TextureHandleFor(string nodeId, string socketGuid)
    {
        var key = (nodeId, socketGuid.ToLowerInvariant());
        if (!textureHandles.TryGetValue(key, out var handle))
        {
            handle = new TextureHandle(nextTextureHandle++);
            textureHandles.Add(key, handle);
            // GUIDs cannot contain '|', so this identity is unambiguous.
            textureIdentities.Add(handle.Value, $"socket|{key.Item1}|{key.Item2}");
            writableTextureHandles.Add(handle.Value);
        }
        return handle;
    }

    private BufferHandle BufferHandleFor(
        string nodeId,
        string socketGuid,
        string elementType)
    {
        var key = (nodeId, socketGuid.ToLowerInvariant());
        if (!bufferHandles.TryGetValue(key, out var handle))
        {
            handle = new BufferHandle(nextBufferHandle++);
            bufferHandles.Add(key, handle);
            writableBufferElementTypes.Add(handle.Value, elementType);
        }
        return handle;
    }

    private static ResolvedGraphResource SceneResource(string socketGuid)
        => socketGuid.ToLowerInvariant() switch
        {
            RenderGraphDocument.SceneGeometrySocketGuid => new(
                "SceneGeometry", "Unknown", "Unknown", new SceneGeometryHandle(PassMemberBinder.GeometryHandleValue)),
            RenderGraphDocument.SceneMaterialsSocketGuid => new(
                "MaterialTable", "Unknown", "Unknown", new MaterialTableHandle(PassMemberBinder.MaterialsHandleValue)),
            RenderGraphDocument.SceneTexturesSocketGuid => new(
                "TextureTable", "Unknown", "Unknown", new TextureTableHandle(PassMemberBinder.TexturesHandleValue)),
            RenderGraphDocument.SceneCameraSocketGuid => new(
                "Camera", "Unknown", "Unknown", new CameraHandle(PassMemberBinder.CameraHandleValue)),
            RenderGraphDocument.SceneLightsSocketGuid => new(
                "LightTable", "Unknown", "Unknown", new LightTableHandle(PassMemberBinder.LightsHandleValue)),
            RenderGraphDocument.SceneTimeSocketGuid => new(
                "Time", "Unknown", "Unknown", new TimeHandle(PassMemberBinder.TimeHandleValue)),
            _ => throw new InvalidDataException($"Unknown scene resource socket {socketGuid}.")
        };

    private static void RequireCompatible(
        ResolvedGraphResource source,
        ProjectPassSocketDefinition target,
        GraphLink link)
    {
        if (!string.Equals(source.ResourceKind, target.ResourceKind, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"Graph link {link.FromNode}.{link.FromSocket} -> {link.ToNode}.{link.ToSocket} " +
                $"has resource kind {source.ResourceKind}, expected {target.ResourceKind}.");
        }
        if (!string.Equals(source.Format, "Unknown", StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(target.Format, "Unknown", StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(source.Format, target.Format, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"Graph link {link.FromNode}.{link.FromSocket} -> {link.ToNode}.{link.ToSocket} " +
                $"has format {source.Format}, expected {target.Format}.");
        }
        if (!string.Equals(source.ElementType, "Unknown", StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(target.ElementType, "Unknown", StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(source.ElementType, target.ElementType, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"Graph link {link.FromNode}.{link.FromSocket} -> {link.ToNode}.{link.ToSocket} " +
                $"has buffer element type {source.ElementType}, expected {target.ElementType}.");
        }
    }

    private static bool SocketMetadataCompatible(
        ProjectPassSocketDefinition source,
        ProjectPassSocketDefinition target)
        => string.Equals(source.ResourceKind, target.ResourceKind, StringComparison.OrdinalIgnoreCase) &&
           MetadataCompatible(source.Format, target.Format) &&
           MetadataCompatible(source.ElementType, target.ElementType);

    private static bool MetadataCompatible(string left, string right)
        => string.Equals(left, "Unknown", StringComparison.OrdinalIgnoreCase) ||
           string.Equals(right, "Unknown", StringComparison.OrdinalIgnoreCase) ||
           string.Equals(left, right, StringComparison.OrdinalIgnoreCase);

    private static bool IsTexture(string resourceKind)
        => string.Equals(resourceKind, "Texture2D", StringComparison.OrdinalIgnoreCase);

    private static bool IsBuffer(string resourceKind)
        => string.Equals(resourceKind, "Buffer", StringComparison.OrdinalIgnoreCase);
}

internal sealed record ResolvedGraphResource(
    string ResourceKind,
    string Format,
    string ElementType,
    object Handle);

internal static class BufferElementTypeNames
{
    public static string NormalizeManifest(string value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);
        var normalized = value.Trim();
        while (normalized.StartsWith("global::", StringComparison.Ordinal))
        {
            normalized = normalized.Substring("global::".Length);
        }
        if (normalized.Length == 0)
        {
            throw new InvalidDataException("Buffer element type must be a non-empty type name.");
        }

        return normalized.ToLowerInvariant() switch
        {
            "unknown" => "Unknown",
            "sbyte" or "system.sbyte" => "sbyte",
            "byte" or "system.byte" => "byte",
            "short" or "int16" or "system.int16" => "short",
            "ushort" or "uint16" or "system.uint16" => "ushort",
            "int" or "int32" or "system.int32" => "int",
            "uint" or "uint32" or "system.uint32" => "uint",
            "long" or "int64" or "system.int64" => "long",
            "ulong" or "uint64" or "system.uint64" => "ulong",
            "float" or "single" or "system.single" => "float",
            "double" or "system.double" => "double",
            "bool" or "boolean" or "system.boolean" => "bool",
            "char" or "system.char" => "char",
            "decimal" or "system.decimal" => "decimal",
            "nint" or "intptr" or "system.intptr" => "nint",
            "nuint" or "uintptr" or "system.uintptr" => "nuint",
            "feather.math.float2" => "float2",
            "feather.math.float3" => "float3",
            "feather.math.float4" => "float4",
            "feather.math.int2" => "int2",
            "feather.math.int3" => "int3",
            "feather.math.int4" => "int4",
            "feather.math.bool2" => "bool2",
            "feather.math.bool3" => "bool3",
            "feather.math.bool4" => "bool4",
            "feather.math.float2x2" => "float2x2",
            "feather.math.float3x3" => "float3x3",
            "feather.math.float4x4" => "float4x4",
            "feather.math.float2x3" => "float2x3",
            "feather.math.float3x2" => "float3x2",
            "feather.math.float2x4" => "float2x4",
            "feather.math.float4x2" => "float4x2",
            "feather.math.float3x4" => "float3x4",
            "feather.math.float4x3" => "float4x3",
            _ => normalized
        };
    }

    public static string Canonical(Type type)
    {
        ArgumentNullException.ThrowIfNull(type);
        if (type == typeof(sbyte)) return "sbyte";
        if (type == typeof(byte)) return "byte";
        if (type == typeof(short)) return "short";
        if (type == typeof(ushort)) return "ushort";
        if (type == typeof(int)) return "int";
        if (type == typeof(uint)) return "uint";
        if (type == typeof(long)) return "long";
        if (type == typeof(ulong)) return "ulong";
        if (type == typeof(float)) return "float";
        if (type == typeof(double)) return "double";
        if (type == typeof(bool)) return "bool";
        if (type == typeof(char)) return "char";
        if (type == typeof(decimal)) return "decimal";
        if (type == typeof(nint)) return "nint";
        if (type == typeof(nuint)) return "nuint";
        return NormalizeManifest((type.FullName ?? type.Name).Replace('+', '.'));
    }

    public static void RequireCompatible(string expected, Type actual, string label)
    {
        var canonicalExpected = NormalizeManifest(expected);
        if (!string.Equals(canonicalExpected, "Unknown", StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(canonicalExpected, Canonical(actual), StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"{label} declares buffer element type {canonicalExpected}, but the C# handle uses " +
                $"{Canonical(actual)}.");
        }
    }
}

internal static class PassMemberBinder
{
    internal const ulong GeometryHandleValue = 1;
    internal const ulong MaterialsHandleValue = 2;
    internal const ulong CameraHandleValue = 3;
    internal const ulong TexturesHandleValue = 4;
    internal const ulong LightsHandleValue = 5;
    internal const ulong TimeHandleValue = 6;

    public static void BindResources(
        IRenderPass pass,
        GraphNode passNode,
        ProjectPassDefinition definition,
        GraphResourceResolver resources)
    {
        var boundInputs = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var boundOutputs = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var member in InstanceMembers(pass.GetType()))
        {
            var input = member.GetCustomAttribute<InputAttribute>(inherit: false);
            var output = member.GetCustomAttribute<OutputAttribute>(inherit: false);
            if (input is null && output is null)
            {
                continue;
            }
            if (input is not null && output is not null)
            {
                throw new InvalidDataException($"Pass member '{member.Name}' cannot be both an input and output.");
            }

            var guidValue = input?.Guid ?? output!.Guid;
            if (!Guid.TryParseExact(guidValue, "D", out var guid))
            {
                throw new InvalidDataException(
                    $"Pass member '{member.Name}' has invalid socket GUID '{guidValue}'.");
            }
            var socketGuid = guid.ToString("D");
            object value;
            if (input is not null)
            {
                var socket = definition.Input(socketGuid);
                value = resources.ResolveInputHandle(passNode, socket);
                value = AdaptResourceHandle(value, MemberType(member), socket, member.Name);
                boundInputs.Add(socketGuid);
            }
            else
            {
                var socket = definition.Output(socketGuid);
                value = resources.ResolveOutputHandle(passNode, socket);
                value = AdaptResourceHandle(value, MemberType(member), socket, member.Name);
                boundOutputs.Add(socketGuid);
            }
            SetValue(pass, member, value);
        }

        if (boundInputs.Count != definition.Inputs.Length ||
            boundOutputs.Count != definition.Outputs.Length)
        {
            throw new InvalidDataException(
                $"Pass type '{definition.TypeName}' resource members do not match its manifest sockets.");
        }
    }

    public static void BindParameters(IRenderPass pass, JsonElement parametersElement)
    {
        var parameters = ReadParameterMap(parametersElement);
        foreach (var member in InstanceMembers(pass.GetType()))
        {
            var attribute = member.GetCustomAttribute<ParameterAttribute>(inherit: false);
            if (attribute is null)
            {
                continue;
            }
            var name = attribute.Name ?? member.Name;
            if (!parameters.TryGetValue(name, out var value))
            {
                continue;
            }

            var memberType = MemberType(member);
            object? converted;
            try
            {
                converted = JsonSerializer.Deserialize(value.GetRawText(), memberType, ProtocolJson.Options);
            }
            catch (JsonException exception)
            {
                throw new InvalidDataException(
                    $"Pass parameter '{name}' cannot be converted to {memberType.FullName}: {exception.Message}",
                    exception);
            }
            if (converted is null && memberType.IsValueType && Nullable.GetUnderlyingType(memberType) is null)
            {
                throw new InvalidDataException($"Pass parameter '{name}' cannot be null.");
            }
            EnforceDeclaredRange(name, attribute, converted);
            SetValue(pass, member, converted);
        }
    }

    /// <summary>
    /// Rejects numeric parameters that fall outside the range the pass declares.
    ///
    /// This matters most for iteration counts: an iterative pass dispatches its kernel once per
    /// step, and a dispatch already in flight cannot be cancelled, so an unreasonable count from a
    /// saved graph would stall the host. Enforcing the pass's own declared bound turns that into a
    /// reported error, and keeps every other declared range honest at the same time.
    /// </summary>
    private static void EnforceDeclaredRange(string name, ParameterAttribute attribute, object? value)
    {
        if (value is null || (double.IsNaN(attribute.Min) && double.IsNaN(attribute.Max)))
        {
            return;
        }
        var numeric = value switch
        {
            int intValue => intValue,
            uint uintValue => uintValue,
            float floatValue => floatValue,
            double doubleValue => doubleValue,
            _ => (double?)null,
        };
        if (numeric is not { } candidate)
        {
            return;
        }
        if (!double.IsNaN(attribute.Min) && candidate < attribute.Min)
        {
            throw new InvalidDataException(
                $"Pass parameter '{name}' is {candidate}; the pass declares a minimum of {attribute.Min}.");
        }
        if (!double.IsNaN(attribute.Max) && candidate > attribute.Max)
        {
            throw new InvalidDataException(
                $"Pass parameter '{name}' is {candidate}; the pass declares a maximum of {attribute.Max}.");
        }
    }

    private static IReadOnlyDictionary<string, JsonElement> ReadParameterMap(JsonElement element)
    {
        var result = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        if (element.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
        {
            return result;
        }
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (var property in element.EnumerateObject())
            {
                result[property.Name] = property.Value;
            }
            return result;
        }
        if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (var definition in element.EnumerateArray())
            {
                if (definition.ValueKind != JsonValueKind.Object ||
                    !definition.TryGetProperty("name", out var nameElement) ||
                    nameElement.ValueKind != JsonValueKind.String)
                {
                    continue;
                }
                if (definition.TryGetProperty("value", out var value) ||
                    definition.TryGetProperty("defaultValue", out value))
                {
                    result[nameElement.GetString()!] = value;
                }
            }
            return result;
        }

        throw new InvalidDataException("Pass parameters must be an object or parameter-definition array.");
    }

    private static IEnumerable<MemberInfo> InstanceMembers(Type type)
        => type.GetMembers(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            .Where(member => member is FieldInfo or PropertyInfo);

    private static Type MemberType(MemberInfo member)
        => member switch
        {
            FieldInfo field => field.FieldType,
            PropertyInfo property => property.PropertyType,
            _ => throw new ArgumentOutOfRangeException(nameof(member))
        };

    private static object AdaptResourceHandle(
        object value,
        Type memberType,
        ProjectPassSocketDefinition socket,
        string memberName)
    {
        if (value is not BufferHandle buffer || memberType == typeof(BufferHandle))
        {
            return value;
        }
        if (!memberType.IsGenericType ||
            memberType.GetGenericTypeDefinition() != typeof(BufferHandle<>))
        {
            return value;
        }

        var elementType = memberType.GetGenericArguments()[0];
        BufferElementTypeNames.RequireCompatible(
            socket.ElementType,
            elementType,
            $"Pass member '{memberName}'");
        return Activator.CreateInstance(memberType, buffer.Value)
            ?? throw new InvalidDataException(
                $"Unable to construct typed buffer handle for pass member '{memberName}'.");
    }

    private static void SetValue(object instance, MemberInfo member, object? value)
    {
        switch (member)
        {
            case FieldInfo field when !field.IsInitOnly:
                if (value is not null && !field.FieldType.IsInstanceOfType(value))
                {
                    throw TypeMismatch(member, field.FieldType, value.GetType());
                }
                field.SetValue(instance, value);
                break;
            case PropertyInfo property when property.SetMethod is not null:
                if (value is not null && !property.PropertyType.IsInstanceOfType(value))
                {
                    throw TypeMismatch(member, property.PropertyType, value.GetType());
                }
                property.SetValue(instance, value);
                break;
            default:
                throw new InvalidDataException($"Pass member '{member.Name}' must be writable.");
        }
    }

    private static InvalidDataException TypeMismatch(MemberInfo member, Type expected, Type actual)
        => new($"Pass member '{member.Name}' has type {expected.FullName}; host supplied {actual.FullName}.");
}

internal sealed class ProjectRenderContextBackend : IRenderContextBackend
{
    private readonly SceneGeometry geometry;
    private readonly SceneMaterialTable materials;
    private readonly SceneTextureTable textures;
    private readonly RenderCamera camera;
    private readonly SceneLightTable lights;
    private readonly RenderTime time;
    private readonly GraphResourceResolver resources;
    private readonly Dictionary<ulong, RenderedFrame> frames = new();

    // GPU-resident outputs live alongside the CPU frames rather than replacing them, so software
    // passes that publish arrays keep working and can be mixed with GPU passes in one graph.
    private readonly Dictionary<ulong, IGpuTexture2D> gpuTextures = new();
    private readonly Dictionary<ulong, DispatchPath> gpuDispatchPaths = new();
    private readonly Dictionary<ulong, GraphBufferData> buffers = new();
    private readonly GraphTexturePool texturePool;

    public ProjectRenderContextBackend(
        RenderSceneResources scene,
        int width,
        int height,
        SampleCount sampleCount,
        float4x4 viewProjection,
        GraphResourceResolver resources,
        IReadOnlyDictionary<string, RenderedFrame> history,
        GraphTexturePool texturePool)
    {
        this.texturePool = texturePool;
        geometry = new SceneGeometry(scene.Geometry.Vertices, scene.Geometry.Indices, scene.Geometry.Submeshes);
        materials = scene.Materials;
        textures = scene.Textures;
        camera = new RenderCamera(viewProjection);
        lights = scene.Lights;
        time = scene.Time;
        this.resources = resources;
        Width = width;
        Height = height;
        SampleCount = sampleCount;

        foreach (var (key, handle) in resources.HistoryReadHandles)
        {
            RenderedFrame frame;
            if (history.TryGetValue(key, out var previous))
            {
                if (previous.Width != width || previous.Height != height)
                {
                    throw new InvalidDataException(
                        $"History resource '{key}' dimensions do not match the current render request.");
                }
                frame = previous;
            }
            else
            {
                var pixels = new Rgba8[checked(width * height)];
                Array.Fill(pixels, new Rgba8(0, 0, 0, 255));
                frame = new RenderedFrame(width, height, pixels, DispatchPath.None);
            }
            frames.Add(handle.Value, frame);
        }
    }

    public int Width { get; }
    public int Height { get; }
    public SampleCount SampleCount { get; }
    public double GpuReadbackMilliseconds { get; private set; }

    public void ReportGpuReadback(TimeSpan elapsed)
        => GpuReadbackMilliseconds += elapsed.TotalMilliseconds;

    public SceneGeometry GetSceneGeometry(SceneGeometryHandle handle)
        => handle.Value == PassMemberBinder.GeometryHandleValue
            ? geometry
            : throw new KeyNotFoundException($"Unknown scene geometry handle {handle.Value}.");

    public RenderCamera GetCamera(CameraHandle handle)
        => handle.Value == PassMemberBinder.CameraHandleValue
            ? camera
            : throw new KeyNotFoundException($"Unknown camera handle {handle.Value}.");

    public SceneMaterialTable GetMaterials(MaterialTableHandle handle)
        => handle.Value == PassMemberBinder.MaterialsHandleValue
            ? materials
            : throw new KeyNotFoundException($"Unknown material table handle {handle.Value}.");

    public SceneTextureTable GetTextures(TextureTableHandle handle)
        => handle.Value == PassMemberBinder.TexturesHandleValue
            ? textures
            : throw new KeyNotFoundException($"Unknown texture table handle {handle.Value}.");

    public SceneLightTable GetLights(LightTableHandle handle)
        => handle.Value == PassMemberBinder.LightsHandleValue
            ? lights
            : throw new KeyNotFoundException($"Unknown light table handle {handle.Value}.");

    public RenderTime GetTime(TimeHandle handle)
        => handle.Value == PassMemberBinder.TimeHandleValue
            ? time
            : throw new KeyNotFoundException($"Unknown time handle {handle.Value}.");

    public ReadOnlyMemory<Rgba8> GetColorInput(TextureHandle handle)
    {
        if (frames.TryGetValue(handle.Value, out var frame))
        {
            return frame.Pixels;
        }
        if (gpuTextures.ContainsKey(handle.Value))
        {
            throw new InvalidDataException(
                $"Texture handle {handle.Value} holds a GPU-resident texture. Read it with " +
                "GetTextureInput to keep the data on the GPU, or have the producing pass publish " +
                "an RGBA8 frame instead.");
        }
        throw new KeyNotFoundException(
            $"Texture handle {handle.Value} has not been produced by an upstream pass.");
    }

    public GpuTexture2D<TPixel, TValue> GetOrCreateGraphTexture<TPixel, TValue>(
        TextureHandle handle,
        int width,
        int height,
        PixelFormat format)
        where TPixel : unmanaged
        where TValue : unmanaged
        => texturePool.GetOrCreate<TPixel, TValue>(
            resources.TextureIdentity(handle),
            width,
            height,
            format);

    public IGpuTexture2D GetTextureInput(TextureHandle handle)
    {
        if (gpuTextures.TryGetValue(handle.Value, out var texture))
        {
            return texture;
        }
        if (frames.TryGetValue(handle.Value, out var frame))
        {
            // The producer was a software pass. Uploading here is the one implicit CPU-to-GPU
            // transfer in the graph; it lets mixed software and GPU graphs work.
            var uploaded = texturePool.GetOrCreate<Rgba8, float4>(
                $"upload|{resources.TextureIdentity(handle)}",
                Width,
                Height,
                PixelFormat.Rgba8);
            uploaded.Upload(frame.Pixels);
            gpuTextures[handle.Value] = uploaded;
            return uploaded;
        }
        throw new KeyNotFoundException(
            $"Texture handle {handle.Value} has not been produced by an upstream pass.");
    }

    public void SetTextureOutput(
        TextureHandle handle,
        IGpuTexture2D texture,
        DispatchPath dispatchPath)
    {
        ArgumentNullException.ThrowIfNull(texture);
        if (!resources.IsWritable(handle))
        {
            throw new KeyNotFoundException($"Texture handle {handle.Value} is not a graph output.");
        }
        if (gpuTextures.ContainsKey(handle.Value) || frames.ContainsKey(handle.Value))
        {
            throw new InvalidOperationException(
                $"Texture handle {handle.Value} has already been written this execution.");
        }
        gpuTextures.Add(handle.Value, texture);
        gpuDispatchPaths[handle.Value] = dispatchPath;
    }

    public ReadOnlyMemory<T> GetBufferInput<T>(BufferHandle<T> handle)
        where T : unmanaged
    {
        if (!buffers.TryGetValue(handle.Value, out var buffer))
        {
            throw new KeyNotFoundException(
                $"Buffer handle {handle.Value} has not been produced by an upstream pass.");
        }
        if (buffer.ElementType != typeof(T) || buffer.Values is not T[] values)
        {
            throw new InvalidDataException(
                $"Buffer handle {handle.Value} contains {BufferElementTypeNames.Canonical(buffer.ElementType)}, " +
                $"not {BufferElementTypeNames.Canonical(typeof(T))}.");
        }
        return values;
    }

    public void SetColorOutput(
        TextureHandle handle,
        Rgba8[] pixels,
        DispatchPath dispatchPath)
    {
        if (!resources.IsWritable(handle))
        {
            throw new KeyNotFoundException($"Unknown color output handle {handle.Value}.");
        }
        if (frames.ContainsKey(handle.Value))
        {
            throw new InvalidOperationException(
                $"Texture handle {handle.Value} was submitted more than once.");
        }
        if (pixels.Length != checked(Width * Height))
        {
            throw new InvalidDataException("The pass color output has the wrong pixel count.");
        }
        frames.Add(handle.Value, new RenderedFrame(Width, Height, pixels, dispatchPath));
    }

    public void SetBufferOutput<T>(
        BufferHandle<T> handle,
        T[] values,
        DispatchPath dispatchPath)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(values);
        var untyped = handle.Untyped;
        if (!resources.IsWritable(untyped))
        {
            throw new KeyNotFoundException($"Unknown buffer output handle {handle.Value}.");
        }
        BufferElementTypeNames.RequireCompatible(
            resources.BufferElementType(untyped),
            typeof(T),
            $"Buffer output {handle.Value}");
        if (buffers.ContainsKey(handle.Value))
        {
            throw new InvalidOperationException(
                $"Buffer handle {handle.Value} was submitted more than once.");
        }
        buffers.Add(handle.Value, new GraphBufferData(typeof(T), values, dispatchPath));
    }

    /// <summary>
    /// Produces the CPU frame that gets published to the viewport. This is the only place the
    /// graph reads back from the GPU, so intermediate passes never pay for a transfer.
    /// </summary>
    public RenderedFrame TakeFrame(TextureHandle handle)
    {
        if (frames.TryGetValue(handle.Value, out var frame))
        {
            return frame;
        }
        if (!gpuTextures.TryGetValue(handle.Value, out var texture))
        {
            throw new InvalidDataException(
                $"The selected graph texture {handle.Value} was not produced.");
        }
        if (texture is not GpuTexture2D<Rgba8, float4> displayTexture)
        {
            throw new InvalidDataException(
                $"The selected graph output uses {texture.Format}; the viewport contract requires " +
                "an RGBA8 texture. Add a pass that converts the result to RGBA8 before the " +
                "Feather Output node.");
        }

        var pixels = new Rgba8[checked(Width * Height)];
        var readback = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            displayTexture.Read(pixels);
        }
        finally
        {
            readback.Stop();
            ReportGpuReadback(readback.Elapsed);
        }
        var path = gpuDispatchPaths.TryGetValue(handle.Value, out var dispatchPath)
            ? dispatchPath
            : DispatchPath.None;
        return new RenderedFrame(Width, Height, pixels, path);
    }

    public IReadOnlyDictionary<string, RenderedFrame> CaptureHistory()
    {
        var result = new Dictionary<string, RenderedFrame>(StringComparer.Ordinal);
        foreach (var (key, handle) in resources.HistoryWriteSources)
        {
            if (!frames.TryGetValue(handle.Value, out var frame))
            {
                throw new InvalidDataException(
                    $"History Write '{key}' references texture {handle.Value}, which was not produced.");
            }
            result.Add(key, frame);
        }
        return result;
    }
}

internal sealed record GraphBufferData(
    Type ElementType,
    Array Values,
    DispatchPath DispatchPath);
