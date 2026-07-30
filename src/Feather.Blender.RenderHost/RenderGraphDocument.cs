using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Feather.Blender.RenderHost;

internal sealed class RenderGraphDocument
{
    public const int CurrentSchemaVersion = 1;
    public const string MinimalRasterPassGuid = "01c671a1-9b4e-5cab-b7e1-c101348af596";
    public const string MinimalRasterPassType = "Feather.Generated.MinimalRasterPass";
    public const string SceneGeometrySocketGuid = "b5db545a-ec06-557c-8b3e-2bc38c8193ef";
    public const string SceneMaterialsSocketGuid = "f4fe7a75-0c26-56d1-af67-01ac7638fe16";
    public const string SceneTexturesSocketGuid = "67190a80-e48b-5bf3-a467-fe84e657e7e6";
    public const string SceneCameraSocketGuid = "6078325d-ed5e-5aa7-a103-1b3292605c40";
    public const string SceneLightsSocketGuid = "d62b1dd6-d641-5ee4-be1e-111d44773721";
    public const string SceneTimeSocketGuid = "427e79d1-aa2f-56de-880c-20102c03acb9";
    public const string GeometryInputSocketGuid = "6d6eb2d5-bb7a-55a4-a85a-c58e36715c53";
    public const string MaterialsInputSocketGuid = "a6eed590-b632-5f91-a69d-09b6eb4bb5ac";
    public const string CameraInputSocketGuid = "cc78191c-ac9a-57b6-bcac-91cce5e298f5";
    public const string ColorOutputSocketGuid = "bd711ea6-36f9-56cd-863a-cfec58727a46";
    public const string OutputColorSocketGuid = "082faef8-760d-5062-9766-2d627d8c42f8";
    public const string HistoryReadSocketGuid = "b85a7129-ad17-5d67-b06b-60e15ce071d0";
    public const string HistoryWriteSocketGuid = "8d513f8b-7212-557b-bcec-2f88ed212c21";
    public const int MaximumScheduledSamples = 1_000_000_000;

    public int SchemaVersion { get; init; }
    public string GenerationId { get; init; } = "";
    public string GraphId { get; init; } = "";
    public string ViewId { get; init; } = "";
    public string ViewKind { get; init; } = "";
    public string ExecutionMode { get; init; } = "";
    public float ResolutionScale { get; init; }
    public int SampleCount { get; init; }
    public int TargetSamples { get; init; }
    public int SamplesPerIteration { get; init; }
    public int PreviewEverySamples { get; init; }
    public GraphNode[] Nodes { get; init; } = [];
    public GraphLink[] Links { get; init; } = [];
    public string[] TopologicalOrder { get; init; } = [];
    public GraphOutput Output { get; init; } = new();

    public static RenderGraphExecution Load(string path)
    {
        RenderGraphDocument graph;
        try
        {
            using var stream = new FileStream(
                path,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read | FileShare.Delete);
            using var buffer = new MemoryStream();
            stream.CopyTo(buffer);
            var bytes = buffer.ToArray();
            graph = JsonSerializer.Deserialize<RenderGraphDocument>(bytes, ProtocolJson.Options)
                ?? throw new InvalidDataException("Render graph JSON contains null.");
            return graph.Validate(ContentFingerprint(bytes));
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException($"Render graph JSON is invalid: {exception.Message}", exception);
        }
    }

    private static string ContentFingerprint(byte[] bytes)
    {
        var document = JsonNode.Parse(bytes)?.AsObject()
            ?? throw new InvalidDataException("Render graph JSON contains null.");
        if (!document.Remove("generationId"))
        {
            throw new InvalidDataException("Render graph generationId is missing.");
        }
        var content = JsonSerializer.SerializeToUtf8Bytes(document, ProtocolJson.Options);
        return "sha256:" + Convert.ToHexString(SHA256.HashData(content)).ToLowerInvariant();
    }

    public static RenderGraphExecution LoadMinimalRaster(string path)
    {
        var graph = Load(path);
        graph.RequireLegacyMinimalRaster();
        return graph;
    }

    private RenderGraphExecution Validate(string graphFingerprint)
    {
        if (SchemaVersion != CurrentSchemaVersion)
        {
            throw new InvalidDataException($"Unsupported render graph schema version: {SchemaVersion}.");
        }

        if (!Guid.TryParse(GenerationId, out _))
        {
            throw new InvalidDataException("Render graph generationId must be a GUID.");
        }
        RequireNonEmpty(GraphId, "graphId");
        RequireNonEmpty(ViewId, "viewId");
        var executionMode = ExecutionMode switch
        {
            "REALTIME" => RenderExecutionMode.Realtime,
            "ON_DEMAND" => RenderExecutionMode.OnDemand,
            "PROGRESSIVE" => RenderExecutionMode.Progressive,
            "OFFLINE" => RenderExecutionMode.Offline,
            _ => (RenderExecutionMode?)null
        };
        if (executionMode is null)
        {
            throw new InvalidDataException($"Unsupported graph executionMode: '{ExecutionMode}'.");
        }
        if (ResolutionScale is < 0.1f or > 2.0f || !float.IsFinite(ResolutionScale))
        {
            throw new InvalidDataException("Render graph resolutionScale must be between 0.1 and 2.0.");
        }
        if (SampleCount is not (1 or 2 or 4 or 8 or 16))
        {
            throw new InvalidDataException("Render graph sampleCount must be 1, 2, 4, 8, or 16.");
        }
        if (TargetSamples is < 0 or > MaximumScheduledSamples)
        {
            throw new InvalidDataException(
                $"Render graph targetSamples must be between 0 and {MaximumScheduledSamples}.");
        }
        if (SamplesPerIteration is < 0 or > MaximumScheduledSamples)
        {
            throw new InvalidDataException(
                $"Render graph samplesPerIteration must be between 0 and {MaximumScheduledSamples}.");
        }
        if (PreviewEverySamples is < 0 or > MaximumScheduledSamples)
        {
            throw new InvalidDataException(
                $"Render graph previewEverySamples must be between 0 and {MaximumScheduledSamples}.");
        }
        if (Output is null)
        {
            throw new InvalidDataException("Render graph output is required.");
        }
        RequireNonEmpty(Output.NodeId, "output.nodeId");
        RequireNonEmpty(Output.SocketGuid, "output.socketGuid");
        if (Nodes is null || Nodes.Length == 0)
        {
            throw new InvalidDataException("Render graph contains no nodes.");
        }
        if (Links is null || TopologicalOrder is null)
        {
            throw new InvalidDataException("Render graph links and topologicalOrder are required.");
        }

        var nodeIds = new HashSet<string>(StringComparer.Ordinal);
        foreach (var node in Nodes)
        {
            if (node is null)
            {
                throw new InvalidDataException("Render graph node entries cannot be null.");
            }
            RequireNonEmpty(node.NodeId, "nodes.nodeId");
            RequireNonEmpty(node.Kind, "nodes.kind");
            if (node.Kind is not ("scene" or "pass" or "output" or "history-read" or "history-write"
                    or "texture"))
            {
                throw new InvalidDataException(
                    $"Render graph node '{node.NodeId}' has unsupported kind '{node.Kind}'.");
            }
            if (!nodeIds.Add(node.NodeId))
            {
                throw new InvalidDataException($"Render graph contains duplicate node ID '{node.NodeId}'.");
            }
            if (node.Kind == "pass")
            {
                RequireNonEmpty(node.PassGuid, "nodes.passGuid");
                if (!Guid.TryParseExact(node.PassGuid, "D", out _))
                {
                    throw new InvalidDataException(
                        $"Render graph pass '{node.NodeId}' has invalid passGuid '{node.PassGuid}'.");
                }
                RequireNonEmpty(node.TypeName, "nodes.typeName");
            }
            else if (node.Kind is "history-read" or "history-write")
            {
                RequireNonEmpty(node.HistoryKey, "nodes.historyKey");
                if (node.HistoryKey.Length > 128 || node.HistoryKey.Any(char.IsControl))
                {
                    throw new InvalidDataException(
                        $"Render graph history key '{node.HistoryKey}' is invalid.");
                }
            }
            else if (node.Kind == "texture")
            {
                RequireNonEmpty(node.TextureKey, "nodes.textureKey");
                if (node.TextureKey.Length > 128 || node.TextureKey.Any(char.IsControl))
                {
                    throw new InvalidDataException(
                        $"Render graph texture key '{node.TextureKey}' is invalid.");
                }
                if (node.Source is not ("IMAGE" or "COMPUTE"))
                {
                    throw new InvalidDataException(
                        $"Render graph texture '{node.TextureKey}' has unsupported source '{node.Source}'.");
                }
                if (!node.MatchRenderSize && (node.Width < 1 || node.Height < 1 ||
                    node.Width > 8192 || node.Height > 8192))
                {
                    throw new InvalidDataException(
                        $"Render graph texture '{node.TextureKey}' has an invalid explicit size.");
                }
            }
        }

        if (!nodeIds.Contains(Output.NodeId))
        {
            throw new InvalidDataException("Render graph output.nodeId does not reference a node.");
        }

        var linkedInputs = new HashSet<(string NodeId, string SocketGuid)>();
        foreach (var link in Links)
        {
            if (!nodeIds.Contains(link.FromNode) || !nodeIds.Contains(link.ToNode))
            {
                throw new InvalidDataException("Render graph link references an unknown node.");
            }
            RequireNonEmpty(link.FromSocket, "links.fromSocket");
            RequireNonEmpty(link.ToSocket, "links.toSocket");
            if (!linkedInputs.Add((link.ToNode, link.ToSocket)))
            {
                throw new InvalidDataException(
                    $"Render graph input '{link.ToNode}.{link.ToSocket}' has multiple links.");
            }
        }

        if (TopologicalOrder.Length != Nodes.Length ||
            TopologicalOrder.Distinct(StringComparer.Ordinal).Count() != Nodes.Length ||
            TopologicalOrder.Any(nodeId => !nodeIds.Contains(nodeId)))
        {
            throw new InvalidDataException("Render graph topologicalOrder must contain every node exactly once.");
        }
        var topologicalIndices = TopologicalOrder
            .Select((nodeId, index) => (nodeId, index))
            .ToDictionary(item => item.nodeId, item => item.index, StringComparer.Ordinal);
        if (Links.Any(link => topologicalIndices[link.FromNode] >= topologicalIndices[link.ToNode]))
        {
            throw new InvalidDataException("Render graph topologicalOrder violates a resource link.");
        }

        var nodesById = Nodes.ToDictionary(node => node.NodeId, StringComparer.Ordinal);
        var orderedNodes = TopologicalOrder.Select(nodeId => nodesById[nodeId]).ToArray();

        var scenes = Nodes.Where(node => node.Kind == "scene").ToArray();
        if (scenes.Length != 1)
        {
            throw new InvalidDataException("Render graph execution requires exactly one scene node.");
        }

        var output = nodesById[Output.NodeId];
        if (output.Kind != "output")
        {
            throw new InvalidDataException("Render graph output.nodeId must reference an output node.");
        }

        var outputLink = Links.SingleOrDefault(link =>
            string.Equals(link.ToNode, output.NodeId, StringComparison.Ordinal) &&
            string.Equals(link.ToSocket, Output.SocketGuid, StringComparison.Ordinal));
        if (outputLink is null)
        {
            throw new InvalidDataException("The selected render graph output socket is not connected.");
        }

        var executionNodeIds = new HashSet<string>(StringComparer.Ordinal)
        {
            output.NodeId,
            scenes[0].NodeId
        };
        var pendingNodes = new Stack<string>();
        pendingNodes.Push(output.NodeId);
        foreach (var historyWrite in Nodes.Where(node => node.Kind == "history-write"))
        {
            if (Links.Any(link => string.Equals(link.ToNode, historyWrite.NodeId, StringComparison.Ordinal)))
            {
                executionNodeIds.Add(historyWrite.NodeId);
                pendingNodes.Push(historyWrite.NodeId);
            }
        }
        // A Texture node that something writes into is a destination in its own right. Without
        // seeding it the pass feeding it would be pruned whenever the output does not also read it,
        // which is exactly the case for a simulation that only shows its result a frame later.
        foreach (var textureNode in Nodes.Where(node => node.Kind == "texture"))
        {
            if (Links.Any(link => string.Equals(link.ToNode, textureNode.NodeId, StringComparison.Ordinal)))
            {
                executionNodeIds.Add(textureNode.NodeId);
                pendingNodes.Push(textureNode.NodeId);
            }
        }
        var incomingNodes = Links
            .GroupBy(link => link.ToNode, StringComparer.Ordinal)
            .ToDictionary(
                group => group.Key,
                group => group.Select(link => link.FromNode).Distinct(StringComparer.Ordinal).ToArray(),
                StringComparer.Ordinal);
        while (pendingNodes.TryPop(out var nodeId))
        {
            if (!incomingNodes.TryGetValue(nodeId, out var dependencies))
            {
                continue;
            }
            foreach (var dependency in dependencies)
            {
                if (executionNodeIds.Add(dependency))
                {
                    pendingNodes.Push(dependency);
                }
            }
        }

        var executionNodes = orderedNodes
            .Where(node => executionNodeIds.Contains(node.NodeId))
            .ToArray();
        var executionLinks = Links
            .Where(link => executionNodeIds.Contains(link.FromNode) &&
                           executionNodeIds.Contains(link.ToNode))
            .ToArray();
        var passes = executionNodes.Where(node => node.Kind == "pass").ToArray();
        if (passes.Length == 0)
        {
            throw new InvalidDataException("Render graph output does not depend on a pass node.");
        }

        if (Links.Any(link => nodesById[link.FromNode].Kind == "output"))
        {
            throw new InvalidDataException("Render graph output nodes cannot be resource sources.");
        }
        if (Links.Any(link => nodesById[link.ToNode].Kind == "scene"))
        {
            throw new InvalidDataException("Render graph scene nodes cannot have resource inputs.");
        }

        var historyReads = executionNodes.Where(node => node.Kind == "history-read").ToArray();
        var historyWrites = executionNodes.Where(node => node.Kind == "history-write").ToArray();
        RequireUniqueHistoryKeys(historyReads, "read");
        RequireUniqueHistoryKeys(historyWrites, "write");
        foreach (var read in historyReads)
        {
            if (executionLinks.Any(link => string.Equals(link.ToNode, read.NodeId, StringComparison.Ordinal)))
            {
                throw new InvalidDataException($"History Read '{read.HistoryKey}' cannot have inputs.");
            }
            if (executionLinks.Any(link =>
                    string.Equals(link.FromNode, read.NodeId, StringComparison.Ordinal) &&
                    !string.Equals(link.FromSocket, HistoryReadSocketGuid, StringComparison.OrdinalIgnoreCase)))
            {
                throw new InvalidDataException(
                    $"History Read '{read.HistoryKey}' uses an unknown output socket.");
            }
        }
        foreach (var write in historyWrites)
        {
            var incoming = executionLinks.Where(link =>
                string.Equals(link.ToNode, write.NodeId, StringComparison.Ordinal)).ToArray();
            if (incoming.Length != 1 ||
                !string.Equals(incoming[0].ToSocket, HistoryWriteSocketGuid, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidDataException(
                    $"History Write '{write.HistoryKey}' requires exactly one Current input.");
            }
            if (executionLinks.Any(link => string.Equals(link.FromNode, write.NodeId, StringComparison.Ordinal)))
            {
                throw new InvalidDataException($"History Write '{write.HistoryKey}' cannot produce current-frame links.");
            }
        }

        var selectedAov = string.IsNullOrWhiteSpace(Output.Aov) ? "Combined" : Output.Aov.Trim();
        if (selectedAov.Length > 128 || selectedAov.Any(char.IsControl))
        {
            throw new InvalidDataException("Render graph output.aov is invalid.");
        }

        var samplesPerIteration = SamplesPerIteration == 0 ? 1 : SamplesPerIteration;
        var previewEverySamples = PreviewEverySamples == 0 ? 1 : PreviewEverySamples;
        var targetSamples = TargetSamples;
        if (targetSamples == 0 && executionMode.Value != RenderExecutionMode.Progressive)
        {
            targetSamples = 1;
        }

        return new RenderGraphExecution(
            GenerationId,
            GraphId,
            ViewId,
            string.IsNullOrWhiteSpace(ViewKind) ? "CUSTOM" : ViewKind,
            executionMode.Value,
            (Feather.SampleCount)SampleCount,
            targetSamples,
            samplesPerIteration,
            previewEverySamples,
            selectedAov,
            executionNodes,
            executionLinks,
            passes,
            scenes[0],
            output,
            outputLink,
            historyReads,
            historyWrites)
        {
            GraphFingerprint = graphFingerprint
        };

        static void RequireUniqueHistoryKeys(GraphNode[] nodes, string direction)
        {
            var duplicate = nodes
                .GroupBy(node => node.HistoryKey, StringComparer.Ordinal)
                .FirstOrDefault(group => group.Count() > 1);
            if (duplicate is not null)
            {
                throw new InvalidDataException(
                    $"Render graph contains duplicate History {direction} key '{duplicate.Key}'.");
            }
        }
    }

    private static void RequireNonEmpty(string value, string name)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new InvalidDataException($"Render graph {name} is required.");
        }
    }
}

internal sealed record RenderGraphExecution(
    string GenerationId,
    string GraphId,
    string ViewId,
    string ViewKind,
    RenderExecutionMode ExecutionMode,
    Feather.SampleCount SampleCount,
    int TargetSamples,
    int SamplesPerIteration,
    int PreviewEverySamples,
    string SelectedAov,
    GraphNode[] Nodes,
    GraphLink[] Links,
    GraphNode[] Passes,
    GraphNode Scene,
    GraphNode Output,
    GraphLink OutputLink,
    GraphNode[] HistoryReads,
    GraphNode[] HistoryWrites)
{
    public string GraphFingerprint { get; init; } = "";

    public string ExecutionModeName => ExecutionMode switch
    {
        RenderExecutionMode.Realtime => "REALTIME",
        RenderExecutionMode.OnDemand => "ON_DEMAND",
        RenderExecutionMode.Progressive => "PROGRESSIVE",
        RenderExecutionMode.Offline => "OFFLINE",
        _ => throw new ArgumentOutOfRangeException(nameof(ExecutionMode))
    };

    public GraphNode Pass => Passes.Single();

    public MinimalRasterSettings Settings => MinimalRasterSettings.FromParameters(Pass.Parameters);

    public GraphLink? IncomingLink(string nodeId, string socketGuid)
        => Links.SingleOrDefault(link =>
            string.Equals(link.ToNode, nodeId, StringComparison.Ordinal) &&
            string.Equals(link.ToSocket, socketGuid, StringComparison.OrdinalIgnoreCase));

    public void RequireLegacyMinimalRaster()
    {
        if (Passes.Length != 1 || Pass.Muted)
        {
            throw new InvalidDataException(
                "Legacy MinimalRaster execution requires exactly one unmuted pass node.");
        }
        if (!string.Equals(Pass.PassGuid, RenderGraphDocument.MinimalRasterPassGuid, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"Legacy RenderHost requests support only MinimalRaster pass GUID {RenderGraphDocument.MinimalRasterPassGuid}.");
        }
        RequireLink(Scene.NodeId, RenderGraphDocument.SceneGeometrySocketGuid, Pass.NodeId,
            RenderGraphDocument.GeometryInputSocketGuid, "Geometry");
        RequireLink(Scene.NodeId, RenderGraphDocument.SceneMaterialsSocketGuid, Pass.NodeId,
            RenderGraphDocument.MaterialsInputSocketGuid, "Materials");
        RequireLink(Scene.NodeId, RenderGraphDocument.SceneCameraSocketGuid, Pass.NodeId,
            RenderGraphDocument.CameraInputSocketGuid, "Camera");
        if (!string.Equals(OutputLink.FromNode, Pass.NodeId, StringComparison.Ordinal) ||
            !string.Equals(OutputLink.FromSocket, RenderGraphDocument.ColorOutputSocketGuid, StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(OutputLink.ToSocket, RenderGraphDocument.OutputColorSocketGuid, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("Legacy MinimalRaster Color is not linked to the selected output node.");
        }
    }

    private void RequireLink(
        string fromNode,
        string fromSocket,
        string toNode,
        string toSocket,
        string resourceName)
    {
        if (!Links.Any(link =>
                string.Equals(link.FromNode, fromNode, StringComparison.Ordinal) &&
                string.Equals(link.FromSocket, fromSocket, StringComparison.OrdinalIgnoreCase) &&
                string.Equals(link.ToNode, toNode, StringComparison.Ordinal) &&
                string.Equals(link.ToSocket, toSocket, StringComparison.OrdinalIgnoreCase)))
        {
            throw new InvalidDataException(
                $"Legacy MinimalRaster {resourceName} input is not linked to the scene node.");
        }
    }
}

internal sealed class GraphNode
{
    public string NodeId { get; init; } = "";
    public string Kind { get; init; } = "";
    public string PassGuid { get; init; } = "";
    public string TypeName { get; init; } = "";
    public bool Muted { get; init; }
    public JsonElement Parameters { get; init; }
    public string HistoryKey { get; init; } = "";

    // Texture nodes declare a graph resource the user controls: a Blender image to sample, or an
    // empty target a compute pass writes into.
    public string TextureKey { get; init; } = "";
    public string Source { get; init; } = "";
    public string Format { get; init; } = "";
    public bool MatchRenderSize { get; init; } = true;
    public int Width { get; init; }
    public int Height { get; init; }
    public string ImageName { get; init; } = "";
}

internal sealed class GraphLink
{
    public string FromNode { get; init; } = "";
    public string FromSocket { get; init; } = "";
    public string ToNode { get; init; } = "";
    public string ToSocket { get; init; } = "";
}

internal sealed class GraphOutput
{
    public string NodeId { get; init; } = "";
    public string SocketGuid { get; init; } = "";
    public string Aov { get; init; } = "";
}

internal enum RenderExecutionMode
{
    Realtime,
    OnDemand,
    Progressive,
    Offline
}

internal sealed record MinimalRasterSettings(
    Feather.Math.float4 ClearColor,
    Feather.Math.float3 LightDirection,
    float Ambient)
{
    public static MinimalRasterSettings FromParameters(JsonElement parametersElement)
    {
        var parameters = ReadParameterMap(parametersElement);
        var clearColor = ReadVector(parameters, "clearColor", [0.035f, 0.045f, 0.06f, 1.0f], 4);
        var lightDirection = ReadVector(parameters, "lightDirection", [0.35f, -0.45f, 0.82f], 3);
        var ambient = ReadScalar(parameters, "ambient", 0.24f);
        if (ambient is < 0.0f or > 1.0f)
        {
            throw new InvalidDataException("MinimalRaster ambient must be between 0 and 1.");
        }

        var length = MathF.Sqrt(
            (lightDirection[0] * lightDirection[0]) +
            (lightDirection[1] * lightDirection[1]) +
            (lightDirection[2] * lightDirection[2]));
        if (!float.IsFinite(length) || length < 1e-6f)
        {
            throw new InvalidDataException("MinimalRaster lightDirection must be a finite non-zero vector.");
        }

        return new MinimalRasterSettings(
            new Feather.Math.float4(clearColor[0], clearColor[1], clearColor[2], clearColor[3]),
            new Feather.Math.float3(lightDirection[0] / length, lightDirection[1] / length, lightDirection[2] / length),
            ambient);
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

        throw new InvalidDataException("Pass parameters must be an object or an array of parameter definitions.");
    }

    private static float ReadScalar(
        IReadOnlyDictionary<string, JsonElement> parameters,
        string name,
        float defaultValue)
    {
        if (!parameters.TryGetValue(name, out var value))
        {
            return defaultValue;
        }
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetSingle(out var result) || !float.IsFinite(result))
        {
            throw new InvalidDataException($"MinimalRaster {name} must be a finite number.");
        }
        return result;
    }

    private static float[] ReadVector(
        IReadOnlyDictionary<string, JsonElement> parameters,
        string name,
        float[] defaultValue,
        int length)
    {
        if (!parameters.TryGetValue(name, out var value))
        {
            return defaultValue;
        }
        if (value.ValueKind != JsonValueKind.Array || value.GetArrayLength() != length)
        {
            throw new InvalidDataException($"MinimalRaster {name} must contain {length} numbers.");
        }

        var result = new float[length];
        var index = 0;
        foreach (var component in value.EnumerateArray())
        {
            if (component.ValueKind != JsonValueKind.Number ||
                !component.TryGetSingle(out result[index]) ||
                !float.IsFinite(result[index]))
            {
                throw new InvalidDataException($"MinimalRaster {name} must contain finite numbers.");
            }
            index++;
        }
        return result;
    }
}
