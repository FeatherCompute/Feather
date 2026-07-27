using System.Text.Json;

namespace Feather.Blender.RenderHost;

internal sealed class RenderGraphDocument
{
    public const int CurrentSchemaVersion = 1;
    public const string MinimalRasterPassGuid = "01c671a1-9b4e-5cab-b7e1-c101348af596";
    public const string MinimalRasterPassType = "Feather.Generated.MinimalRasterPass";

    public int SchemaVersion { get; init; }
    public string GraphId { get; init; } = "";
    public string ViewId { get; init; } = "";
    public string ExecutionMode { get; init; } = "";
    public float ResolutionScale { get; init; }
    public int SampleCount { get; init; }
    public GraphNode[] Nodes { get; init; } = [];
    public GraphLink[] Links { get; init; } = [];
    public string[] TopologicalOrder { get; init; } = [];
    public GraphOutput Output { get; init; } = new();

    public static RenderGraphExecution LoadMinimalRaster(string path)
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

        return graph.ValidateMinimalRaster();
    }

    private RenderGraphExecution ValidateMinimalRaster()
    {
        if (SchemaVersion != CurrentSchemaVersion)
        {
            throw new InvalidDataException($"Unsupported render graph schema version: {SchemaVersion}.");
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
        RequireNonEmpty(Output.NodeId, "output.nodeId");
        RequireNonEmpty(Output.SocketGuid, "output.socketGuid");
        if (Nodes.Length == 0)
        {
            throw new InvalidDataException("Render graph contains no nodes.");
        }

        var nodeIds = new HashSet<string>(StringComparer.Ordinal);
        foreach (var node in Nodes)
        {
            RequireNonEmpty(node.NodeId, "nodes.nodeId");
            RequireNonEmpty(node.Kind, "nodes.kind");
            if (!nodeIds.Add(node.NodeId))
            {
                throw new InvalidDataException($"Render graph contains duplicate node ID '{node.NodeId}'.");
            }
        }

        if (!nodeIds.Contains(Output.NodeId))
        {
            throw new InvalidDataException("Render graph output.nodeId does not reference a node.");
        }

        foreach (var link in Links)
        {
            if (!nodeIds.Contains(link.FromNode) || !nodeIds.Contains(link.ToNode))
            {
                throw new InvalidDataException("Render graph link references an unknown node.");
            }
            RequireNonEmpty(link.FromSocket, "links.fromSocket");
            RequireNonEmpty(link.ToSocket, "links.toSocket");
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

        var passes = Nodes.Where(node =>
            string.Equals(node.Kind, "pass", StringComparison.Ordinal) && !node.Muted).ToArray();
        if (passes.Length != 1)
        {
            throw new InvalidDataException("MinimalRaster execution requires exactly one unmuted pass node.");
        }

        var pass = passes[0];
        if (!string.Equals(pass.PassGuid, MinimalRasterPassGuid, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"RenderHost MVP supports only MinimalRaster pass GUID {MinimalRasterPassGuid}.");
        }
        RequireNonEmpty(pass.TypeName, "nodes.typeName");

        var output = Nodes.Single(node => node.NodeId == Output.NodeId);
        if (!string.Equals(output.Kind, "output", StringComparison.Ordinal))
        {
            throw new InvalidDataException("Render graph output.nodeId must reference an output node.");
        }
        if (!Links.Any(link =>
            link.FromNode == pass.NodeId &&
            link.ToNode == output.NodeId &&
            link.ToSocket == Output.SocketGuid))
        {
            throw new InvalidDataException("MinimalRaster pass is not linked to the selected output node.");
        }

        return new RenderGraphExecution(
            ViewId,
            (Feather.SampleCount)SampleCount,
            MinimalRasterSettings.FromParameters(pass.Parameters));
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
    string ViewId,
    Feather.SampleCount SampleCount,
    MinimalRasterSettings Settings);

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
