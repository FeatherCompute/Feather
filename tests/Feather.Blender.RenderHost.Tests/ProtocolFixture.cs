using System.Buffers.Binary;
using System.Text.Json;

namespace Feather.Blender.RenderHost.Tests;

internal sealed class ProtocolFixture : IDisposable
{
    public const string GenerationId = "5ebc93da-b905-4f44-8eda-68968bb6ba2f";
    public const string PostProcessPassGuid = "20fc3bf1-eecb-41b1-bd03-d5c8a344ce6d";
    public const string PostProcessInputSocketGuid = "071e6d38-8fbc-4fe0-a736-f5d5142ac2aa";
    public const string PostProcessOutputSocketGuid = "14c671df-f382-4aa6-8cf6-3a7ced0456f8";
    public const string HistoryProbePassGuid = "3a458b99-47a9-47ac-8388-5e5146b2bc35";
    public const string HistoryProbeInputSocketGuid = "cb160cbd-52f8-4528-977e-9e0becd9cc6e";
    public const string HistoryProbeObservedOutputSocketGuid = "b5710275-26d7-48b7-846f-8075f67d48e4";
    public const string HistoryProbeNextOutputSocketGuid = "db62a9f6-7937-417e-9024-f5f308baf0a7";

    private readonly string root = Path.Combine(
        Path.GetTempPath(),
        $"feather-render-host-tests-{Guid.NewGuid():N}");

    public ProtocolFixture()
    {
        Directory.CreateDirectory(root);
    }

    public string Root => root;
    public string ScenePath => Path.Combine(root, "scene.featherscene");
    public string GraphPath => Path.Combine(root, "graph.json");
    public string RequestPath => Path.Combine(root, "render-request.json");
    public string OutputPath => Path.Combine(root, "viewport.frame");

    public void WriteScene(
        float[]? matrixWorld = null,
        bool invalidPositionsOffset = false,
        float[]? cornerNormals = null)
    {
        using var payload = new MemoryStream();
        var positions = WriteFloatArray(payload,
            [-0.75f, -0.65f, 0.5f, 0.75f, -0.65f, 0.5f, 0.0f, 0.75f, 0.5f],
            [3, 3]);
        var loopVertices = WriteUIntArray(payload, [0, 1, 2], [3]);
        cornerNormals ??= [0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f];
        var normalDescriptor = WriteFloatArray(payload,
            cornerNormals,
            [3, 3]);
        var triangleLoops = WriteUIntArray(payload, [0, 1, 2], [1, 3]);
        var triangleMaterials = WriteUIntArray(payload, [0], [1]);
        if (invalidPositionsOffset)
        {
            positions["offset"] = payload.Length + 4;
        }

        matrixWorld ??=
        [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ];
        var metadata = new
        {
            schemaVersion = 1,
            generationId = GenerationId,
            matrixLayout = "row-major",
            frame = 1,
            subframe = 0.0f,
            meshes = new[]
            {
                new
                {
                    name = "Evaluated Triangle",
                    vertexCount = 3,
                    cornerCount = 3,
                    triangleCount = 1,
                    materialSlots = Array.Empty<string>(),
                    attributes = new
                    {
                        positions,
                        loopVertexIndices = loopVertices,
                        cornerNormals = normalDescriptor,
                        triangleLoopIndices = triangleLoops,
                        triangleMaterialIndices = triangleMaterials
                    },
                    meshId = "mesh-0"
                }
            },
            instances = new[]
            {
                new
                {
                    instanceId = "instance-0",
                    name = "Triangle",
                    meshId = "mesh-0",
                    matrixWorld,
                    isInstance = false
                }
            },
            materials = Array.Empty<object>(),
            lights = Array.Empty<object>(),
            camera = (object?)null
        };
        var metadataBytes = JsonSerializer.SerializeToUtf8Bytes(metadata);
        var payloadBytes = payload.ToArray();

        using var file = File.Create(ScenePath);
        Span<byte> header = stackalloc byte[24];
        "FTHSCN01"u8.CopyTo(header);
        BinaryPrimitives.WriteUInt32LittleEndian(header[8..12], 1);
        BinaryPrimitives.WriteUInt32LittleEndian(header[12..16], checked((uint)metadataBytes.Length));
        BinaryPrimitives.WriteUInt64LittleEndian(header[16..24], checked((ulong)payloadBytes.Length));
        file.Write(header);
        file.Write(metadataBytes);
        file.Write(payloadBytes);
    }

    public void WriteGraph(
        string passGuid = RenderGraphDocument.MinimalRasterPassGuid,
        string typeName = RenderGraphDocument.MinimalRasterPassType,
        bool parameterDefinitions = false,
        int sampleCount = 1)
    {
        object parameters = parameterDefinitions
            ? Array.Empty<object>()
            : new
            {
                clearColor = new[] { 0.01f, 0.02f, 0.03f, 1.0f },
                lightDirection = new[] { 0.0f, 0.0f, 1.0f },
                ambient = 0.1f
            };
        var graph = new
        {
            schemaVersion = 1,
            generationId = GenerationId,
            graphId = "graph-1",
            viewId = "view-1",
            executionMode = "REALTIME",
            resolutionScale = 1.0f,
            sampleCount,
            nodes = new object[]
            {
                new { nodeId = "scene-1", kind = "scene" },
                new
                {
                    nodeId = "pass-1",
                    kind = "pass",
                    passGuid,
                    typeName,
                    muted = false,
                    parameters
                },
                new { nodeId = "output-1", kind = "output" }
            },
            links = new[]
            {
                new
                {
                    fromNode = "scene-1",
                    fromSocket = RenderGraphDocument.SceneGeometrySocketGuid,
                    toNode = "pass-1",
                    toSocket = RenderGraphDocument.GeometryInputSocketGuid
                },
                new
                {
                    fromNode = "scene-1",
                    fromSocket = RenderGraphDocument.SceneMaterialsSocketGuid,
                    toNode = "pass-1",
                    toSocket = RenderGraphDocument.MaterialsInputSocketGuid
                },
                new
                {
                    fromNode = "scene-1",
                    fromSocket = RenderGraphDocument.SceneCameraSocketGuid,
                    toNode = "pass-1",
                    toSocket = RenderGraphDocument.CameraInputSocketGuid
                },
                new
                {
                    fromNode = "pass-1",
                    fromSocket = RenderGraphDocument.ColorOutputSocketGuid,
                    toNode = "output-1",
                    toSocket = RenderGraphDocument.OutputColorSocketGuid
                }
            },
            topologicalOrder = new[] { "scene-1", "pass-1", "output-1" },
            output = new
            {
                nodeId = "output-1",
                socketGuid = RenderGraphDocument.OutputColorSocketGuid
            }
        };
        File.WriteAllText(GraphPath, JsonSerializer.Serialize(graph));
    }

    public void WriteTwoPassGraph(
        string firstPassType,
        string secondPassType,
        bool secondPassMuted = false)
    {
        var graph = new
        {
            schemaVersion = 1,
            generationId = GenerationId,
            graphId = "graph-1",
            viewId = "view-1",
            executionMode = "REALTIME",
            resolutionScale = 1.0f,
            sampleCount = 1,
            nodes = new object[]
            {
                new { nodeId = "scene-1", kind = "scene" },
                new
                {
                    nodeId = "pass-1",
                    kind = "pass",
                    passGuid = RenderGraphDocument.MinimalRasterPassGuid,
                    typeName = firstPassType,
                    muted = false,
                    parameters = Array.Empty<object>()
                },
                new
                {
                    nodeId = "pass-2",
                    kind = "pass",
                    passGuid = PostProcessPassGuid,
                    typeName = secondPassType,
                    muted = secondPassMuted,
                    parameters = Array.Empty<object>()
                },
                new { nodeId = "output-1", kind = "output" }
            },
            links = new[]
            {
                new
                {
                    fromNode = "scene-1",
                    fromSocket = RenderGraphDocument.SceneGeometrySocketGuid,
                    toNode = "pass-1",
                    toSocket = RenderGraphDocument.GeometryInputSocketGuid
                },
                new
                {
                    fromNode = "scene-1",
                    fromSocket = RenderGraphDocument.SceneMaterialsSocketGuid,
                    toNode = "pass-1",
                    toSocket = RenderGraphDocument.MaterialsInputSocketGuid
                },
                new
                {
                    fromNode = "scene-1",
                    fromSocket = RenderGraphDocument.SceneCameraSocketGuid,
                    toNode = "pass-1",
                    toSocket = RenderGraphDocument.CameraInputSocketGuid
                },
                new
                {
                    fromNode = "pass-1",
                    fromSocket = RenderGraphDocument.ColorOutputSocketGuid,
                    toNode = "pass-2",
                    toSocket = PostProcessInputSocketGuid
                },
                new
                {
                    fromNode = "pass-2",
                    fromSocket = PostProcessOutputSocketGuid,
                    toNode = "output-1",
                    toSocket = RenderGraphDocument.OutputColorSocketGuid
                }
            },
            topologicalOrder = new[] { "scene-1", "pass-1", "pass-2", "output-1" },
            output = new
            {
                nodeId = "output-1",
                socketGuid = RenderGraphDocument.OutputColorSocketGuid
            }
        };
        File.WriteAllText(GraphPath, JsonSerializer.Serialize(graph));
    }

    public void WriteHistoryGraph(string passType)
    {
        var graph = new
        {
            schemaVersion = 1,
            generationId = GenerationId,
            graphId = "history-graph-1",
            viewId = "view-1",
            executionMode = "REALTIME",
            resolutionScale = 1.0f,
            sampleCount = 1,
            nodes = new object[]
            {
                new { nodeId = "scene-1", kind = "scene" },
                new
                {
                    nodeId = "history-read-1",
                    kind = "history-read",
                    historyKey = "accumulation"
                },
                new
                {
                    nodeId = "pass-1",
                    kind = "pass",
                    passGuid = HistoryProbePassGuid,
                    typeName = passType,
                    muted = false,
                    parameters = Array.Empty<object>()
                },
                new { nodeId = "output-1", kind = "output" },
                new
                {
                    nodeId = "history-write-1",
                    kind = "history-write",
                    historyKey = "accumulation"
                }
            },
            links = new[]
            {
                new
                {
                    fromNode = "history-read-1",
                    fromSocket = RenderGraphDocument.HistoryReadSocketGuid,
                    toNode = "pass-1",
                    toSocket = HistoryProbeInputSocketGuid
                },
                new
                {
                    fromNode = "pass-1",
                    fromSocket = HistoryProbeObservedOutputSocketGuid,
                    toNode = "output-1",
                    toSocket = RenderGraphDocument.OutputColorSocketGuid
                },
                new
                {
                    fromNode = "pass-1",
                    fromSocket = HistoryProbeNextOutputSocketGuid,
                    toNode = "history-write-1",
                    toSocket = RenderGraphDocument.HistoryWriteSocketGuid
                }
            },
            topologicalOrder = new[]
            {
                "scene-1",
                "history-read-1",
                "pass-1",
                "output-1",
                "history-write-1"
            },
            output = new
            {
                nodeId = "output-1",
                socketGuid = RenderGraphDocument.OutputColorSocketGuid
            }
        };
        File.WriteAllText(GraphPath, JsonSerializer.Serialize(graph));
    }

    public void WriteRequest(
        string clipSpace = "vulkan",
        string? manifestPath = null,
        ulong requestId = 42,
        float[]? viewProjection = null,
        float[]? cameraPosition = null)
    {
        viewProjection ??=
        [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ];
        var request = new Dictionary<string, object?>
        {
            ["schemaVersion"] = 1,
            ["requestId"] = requestId,
            ["generationId"] = GenerationId,
            ["viewId"] = "view-1",
            ["scenePath"] = Path.GetFileName(ScenePath),
            ["graphPath"] = Path.GetFileName(GraphPath),
            ["manifestPath"] = manifestPath,
            ["outputPath"] = Path.GetFileName(OutputPath),
            ["width"] = 64,
            ["height"] = 64,
            ["matrixLayout"] = "row-major",
            ["clipSpace"] = clipSpace,
            ["viewProjection"] = viewProjection
        };
        // Only emit cameraPosition when supplied, so tests can exercise the absent-field default path.
        if (cameraPosition is not null)
        {
            request["cameraPosition"] = cameraPosition;
        }
        File.WriteAllText(RequestPath, JsonSerializer.Serialize(request));
    }

    public void Dispose()
    {
        try
        {
            Directory.Delete(root, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    private static Dictionary<string, object> WriteFloatArray(
        Stream payload,
        IReadOnlyList<float> values,
        int[] shape)
    {
        var offset = payload.Position;
        Span<byte> buffer = stackalloc byte[sizeof(float)];
        foreach (var value in values)
        {
            BinaryPrimitives.WriteInt32LittleEndian(buffer, BitConverter.SingleToInt32Bits(value));
            payload.Write(buffer);
        }
        return Descriptor(offset, values.Count * sizeof(float), "float32", shape);
    }

    private static Dictionary<string, object> WriteUIntArray(
        Stream payload,
        IReadOnlyList<uint> values,
        int[] shape)
    {
        var offset = payload.Position;
        Span<byte> buffer = stackalloc byte[sizeof(uint)];
        foreach (var value in values)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(buffer, value);
            payload.Write(buffer);
        }
        return Descriptor(offset, values.Count * sizeof(uint), "uint32", shape);
    }

    private static Dictionary<string, object> Descriptor(
        long offset,
        int byteLength,
        string componentType,
        int[] shape)
        => new()
        {
            ["offset"] = offset,
            ["byteLength"] = byteLength,
            ["componentType"] = componentType,
            ["shape"] = shape
        };
}
