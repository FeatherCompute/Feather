using System.Reflection;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Feather.Math;
using Feather.RenderGraph;

namespace Feather.Blender.RenderHost;

internal sealed class ProjectPassAssemblyManager : IDisposable
{
    private PassAssemblyGeneration? current;
    private bool disposed;

    internal WeakReference? LastUnloadedContextForTesting { get; private set; }

    public ProjectPassExecutionResult Execute(
        string manifestPath,
        RenderGraphExecution graph,
        RenderGeometry geometry,
        int width,
        int height,
        float4x4 viewProjection)
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

        return current!.Execute(graph, geometry, width, height, viewProjection, reloaded);
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
    bool Reloaded);

internal sealed class PassAssemblyGeneration : IDisposable
{
    private readonly ProjectPassLoadContext loadContext;
    private readonly Assembly assembly;
    private readonly ProjectPassManifest manifest;
    private readonly Dictionary<string, Type> passTypes;
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
        RenderGeometry geometry,
        int width,
        int height,
        float4x4 viewProjection,
        bool reloaded)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        var definition = manifest.Passes.SingleOrDefault(pass =>
            string.Equals(pass.PassGuid, graph.Pass.PassGuid, StringComparison.OrdinalIgnoreCase))
            ?? throw new InvalidDataException(
                $"Manifest build '{manifest.BuildId}' does not define pass GUID {graph.Pass.PassGuid}.");
        definition.ValidateMinimalRasterSockets();
        if (!string.Equals(definition.TypeName, graph.Pass.TypeName, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Graph pass type '{graph.Pass.TypeName}' does not match manifest type '{definition.TypeName}'. " +
                "Refresh the Feather pass nodes after building.");
        }

        var type = passTypes[definition.PassGuid];
        var instance = (IRenderPass)(Activator.CreateInstance(type)
            ?? throw new InvalidDataException($"Unable to create pass type '{definition.TypeName}'."));
        var backend = new ProjectRenderContextBackend(
            geometry,
            width,
            height,
            graph.SampleCount,
            viewProjection);
        try
        {
            PassMemberBinder.BindResources(instance);
            PassMemberBinder.BindParameters(instance, graph.Pass.Parameters);
            instance.Execute(new RenderContext(backend));
            return new ProjectPassExecutionResult(
                backend.TakeFrame(),
                manifest.BuildId,
                definition.TypeName,
                reloaded);
        }
        finally
        {
            (instance as IDisposable)?.Dispose();
        }
    }

    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

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
                ReadSocketGuids(pass, "inputs"),
                ReadSocketGuids(pass, "outputs")));
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
        var definition = Passes.SingleOrDefault(pass =>
            string.Equals(pass.PassGuid, graph.Pass.PassGuid, StringComparison.OrdinalIgnoreCase))
            ?? throw new InvalidDataException(
                $"Manifest build '{BuildId}' does not define pass GUID {graph.Pass.PassGuid}.");
        definition.ValidateMinimalRasterSockets();
        if (!string.Equals(definition.TypeName, graph.Pass.TypeName, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Graph pass type '{graph.Pass.TypeName}' does not match manifest type '{definition.TypeName}'. " +
                "Refresh the Feather pass nodes after building.");
        }
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

    private static string[] ReadSocketGuids(JsonObject pass, string name)
    {
        if (pass[name] is not JsonArray sockets)
        {
            throw new InvalidDataException($"Pass manifest {name} must be an array.");
        }

        var result = new List<string>(sockets.Count);
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
            result.Add(normalized);
        }
        return result.ToArray();
    }
}

internal sealed record ProjectPassDefinition(
    string PassGuid,
    string TypeName,
    string[] Inputs,
    string[] Outputs)
{
    public void ValidateMinimalRasterSockets()
    {
        foreach (var guid in new[]
                 {
                     RenderGraphDocument.GeometryInputSocketGuid,
                     RenderGraphDocument.MaterialsInputSocketGuid,
                     RenderGraphDocument.CameraInputSocketGuid
                 })
        {
            if (!Inputs.Contains(guid, StringComparer.OrdinalIgnoreCase))
            {
                throw new InvalidDataException($"Project MinimalRaster pass is missing input socket {guid}.");
            }
        }
        if (!Outputs.Contains(RenderGraphDocument.ColorOutputSocketGuid, StringComparer.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"Project MinimalRaster pass is missing color output socket {RenderGraphDocument.ColorOutputSocketGuid}.");
        }
    }
}

internal static class PassMemberBinder
{
    private const ulong GeometryHandleValue = 1;
    private const ulong MaterialsHandleValue = 2;
    private const ulong CameraHandleValue = 3;
    internal const ulong ColorHandleValue = 4;

    public static void BindResources(IRenderPass pass)
    {
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

            var guid = (input?.Guid ?? output!.Guid).ToLowerInvariant();
            object value = guid switch
            {
                RenderGraphDocument.GeometryInputSocketGuid => new SceneGeometryHandle(GeometryHandleValue),
                RenderGraphDocument.MaterialsInputSocketGuid => new MaterialTableHandle(MaterialsHandleValue),
                RenderGraphDocument.CameraInputSocketGuid => new CameraHandle(CameraHandleValue),
                RenderGraphDocument.ColorOutputSocketGuid => new TextureHandle(ColorHandleValue),
                _ => throw new InvalidDataException(
                    $"The MinimalRaster host cannot bind pass resource member '{member.Name}' ({guid}).")
            };
            SetValue(pass, member, value);
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
            SetValue(pass, member, converted);
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
    private readonly RenderCamera camera;
    private RenderedFrame? frame;

    public ProjectRenderContextBackend(
        RenderGeometry geometry,
        int width,
        int height,
        SampleCount sampleCount,
        float4x4 viewProjection)
    {
        this.geometry = new SceneGeometry(geometry.Vertices, geometry.Indices);
        camera = new RenderCamera(viewProjection);
        Width = width;
        Height = height;
        SampleCount = sampleCount;
    }

    public int Width { get; }
    public int Height { get; }
    public SampleCount SampleCount { get; }

    public SceneGeometry GetSceneGeometry(SceneGeometryHandle handle)
        => handle.Value == 1
            ? geometry
            : throw new KeyNotFoundException($"Unknown scene geometry handle {handle.Value}.");

    public RenderCamera GetCamera(CameraHandle handle)
        => handle.Value == 3
            ? camera
            : throw new KeyNotFoundException($"Unknown camera handle {handle.Value}.");

    public void SetColorOutput(
        TextureHandle handle,
        Rgba8[] pixels,
        DispatchPath dispatchPath)
    {
        if (handle.Value != PassMemberBinder.ColorHandleValue)
        {
            throw new KeyNotFoundException($"Unknown color output handle {handle.Value}.");
        }
        if (frame is not null)
        {
            throw new InvalidOperationException("The pass submitted its color output more than once.");
        }
        if (pixels.Length != checked(Width * Height))
        {
            throw new InvalidDataException("The pass color output has the wrong pixel count.");
        }
        frame = new RenderedFrame(Width, Height, pixels, dispatchPath);
    }

    public RenderedFrame TakeFrame()
        => frame ?? throw new InvalidDataException(
            "The project pass completed without calling RenderContext.SetColorOutput.");
}
