using System.Text.Json;

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

    public int SchemaVersion { get; init; }
    public string GenerationId { get; init; } = "";
    public string GraphId { get; init; } = "";
    public string ViewId { get; init; } = "";
    public string ExecutionMode { get; init; } = "";
    public float ResolutionScale { get; init; }
    public int SampleCount { get; init; }
    public GraphNode[] Nodes { get; init; } = [];
    public GraphLink[] Links { get; init; } = [];
    public string[] TopologicalOrder { get; init; } = [];
    public GraphOutput Output { get; init; } = new();

    public static RenderGraphExecution Load(string path)
    {
        RenderGraphDocument graph;
        try
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete);
            graph = JsonSerializer.Deserialize<RenderGraphDocument>(stream, ProtocolJson.Options)
                ?? throw new InvalidDataException("Render graph JSON contains null.");
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException($"Render graph JSON is invalid: {exception.Message}", exception);
        }

        return graph.Validate();
    }

    public static RenderGraphExecution LoadMinimalRaster(string path)
    {
        var graph = Load(path);
        graph.RequireLegacyMinimalRaster();
        return graph;
    }

    private RenderGraphExecution Validate()
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
        if (ExecutionMode is not ("REALTIME" or "ON_DEMAND" or "PROGRESSIVE" or "OFFLINE"))
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
            if (node.Kind is not ("scene" or "pass" or "output"))
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
        var passes = orderedNodes.Where(node => node.Kind == "pass").ToArray();
        if (passes.Length == 0)
        {
            throw new InvalidDataException("Render graph requires at least one pass node.");
        }

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

        if (Links.Any(link => nodesById[link.FromNode].Kind == "output"))
        {
            throw new InvalidDataException("Render graph output nodes cannot be resource sources.");
        }
        if (Links.Any(link => nodesById[link.ToNode].Kind == "scene"))
        {
            throw new InvalidDataException("Render graph scene nodes cannot have resource inputs.");
        }

        return new RenderGraphExecution(
            GenerationId,
            GraphId,
            ViewId,
            (Feather.SampleCount)SampleCount,
            Nodes,
            Links,
            passes,
            scenes[0],
            output,
            outputLink);
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
    Feather.SampleCount SampleCount,
    GraphNode[] Nodes,
    GraphLink[] Links,
    GraphNode[] Passes,
    GraphNode Scene,
    GraphNode Output,
    GraphLink OutputLink)
{
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
