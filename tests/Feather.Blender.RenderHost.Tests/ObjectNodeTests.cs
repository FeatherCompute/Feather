using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Feather.Math;
using Feather.RenderGraph;

namespace Feather.Blender.RenderHost.Tests;

/// <summary>
/// Guards the Object node: one scene object, chosen in the graph, handed to a pass as its transform
/// and its own mesh rather than the whole flattened scene.
/// </summary>
/// <remarks>
/// Asserted through a real render for the same reason the camera node is: what breaks is the wiring --
/// a node kind the schema refuses, a handle the backend cannot resolve, or a triangle range that
/// silently spills into the neighbouring object. The probe pass encodes what it was handed into pixels,
/// so the assertions read the values a real pass would have received.
///
/// The scene is two triangles under two names, deliberately sharing one material so the flattener
/// merges their submeshes into a single run. That merge is what makes narrowing to one object a
/// clipping problem instead of a filtering one.
/// </remarks>
public sealed class ObjectNodeTests
{
    private const string ProbePassGuid = "4a71c93e-8d25-4f16-b937-2e6c5a8d1b74";
    private const string ProbeObjectInputGuid = "b83e5d17-6c49-4a28-9f53-7d1b4e2a6c95";
    private const string ProbeOutputGuid = "2c9f6b48-5a13-4e87-b294-8d3c1f7a5e62";
    private const string PairPassGuid = "6e2b9a54-7f38-4c61-a825-3d9f1c7b4e26";
    private const string PairFirstInputGuid = "d5194c73-2a86-4b39-8e47-1f6c3b9a5d28";
    private const string PairSecondInputGuid = "a37e6b81-4d59-42c7-b163-8e5f2c9d7a34";
    private const string PairOutputGuid = "f14c8d29-6b73-4e15-92a8-5c7d3b1e6f48";

    [Fact]
    public void ObjectNodeHandsThePassThatObjectsTransform()
    {
        // The second object sits five along X, three up Z. The probe reads the translation column, so
        // this fails if the node resolved the wrong object or the identity fallback.
        var probe = RenderProbe(objectName: "Second");

        Assert.Equal(5.0f, probe.TranslationX, 4);
        Assert.Equal(3.0f, probe.TranslationZ, 4);
    }

    [Fact]
    public void ObjectNodeHandsOverOnlyThatObjectsTriangles()
    {
        // Both objects are one triangle each, so the whole scene is two. Receiving six indices would
        // mean the node handed over the flattened scene instead of the object.
        var probe = RenderProbe(objectName: "First");

        Assert.True(probe.Exists);
        Assert.Equal(3, probe.IndexCount);
        Assert.Equal(1, probe.SubmeshCount);
    }

    [Fact]
    public void EachObjectResolvesToItsOwnTriangle()
    {
        // The two objects share a material, so the flattener merges their submeshes into one six-index
        // run. Narrowing has to clip that run rather than take it whole.
        var first = RenderProbe(objectName: "First");
        var second = RenderProbe(objectName: "Second");

        Assert.Equal(3, first.IndexCount);
        Assert.Equal(3, second.IndexCount);
        Assert.Equal(1, first.SubmeshCount);
        Assert.Equal(1, second.SubmeshCount);
        // Every submesh must be expressed against the geometry the pass was handed, so the first index
        // of an object's only submesh is zero regardless of where it sat in the scene buffer.
        Assert.Equal(0, first.FirstSubmeshIndex);
        Assert.Equal(0, second.FirstSubmeshIndex);
    }

    [Fact]
    public void GeometryIsWorldSpaceLikeTheSceneGeometryItComesFrom()
    {
        // The second object's triangle is offset five along X. World-space positions are what a pass
        // reading the scene's geometry already receives, so an object must not arrive in local space.
        var second = RenderProbe(objectName: "Second");
        var first = RenderProbe(objectName: "First");

        Assert.Equal(5.0f, second.FirstVertexX, 4);
        Assert.Equal(0.0f, first.FirstVertexX, 4);
    }

    [Fact]
    public void NarrowedIndicesStillAddressTheWholeSceneVertexBuffer()
    {
        // Only the indices are narrowed; the vertex buffer stays the scene's so nothing has to be
        // copied or rebased. A pass therefore walks its object through Indices, never through
        // Vertices, and this pins that contract down before an effect relies on the other reading.
        var probe = RenderProbe(objectName: "Second");

        Assert.Equal(3, probe.IndexCount);
        Assert.Equal(6, probe.VertexCount);
    }

    [Fact]
    public void AMissingObjectIsReportedRatherThanThrown()
    {
        // A name typed into the graph goes stale the moment the object is renamed or hidden. Refusing
        // the frame would break the graph exactly while the user is editing it.
        var probe = RenderProbe(objectName: "Absent");

        Assert.False(probe.Exists);
        Assert.Equal(0, probe.IndexCount);
    }

    [Fact]
    public void AnEmptyObjectNameIsRejected()
    {
        using var fixture = new ProtocolFixture();
        WriteTwoObjectScene(fixture);
        WriteObjectGraph(fixture, objectName: "");

        Assert.Contains(
            "objectName",
            Assert.Throws<InvalidDataException>(
                () => RenderGraphDocument.Load(fixture.GraphPath)).Message);
    }

    [Fact]
    public void ObjectNodeCannotBeGivenInputs()
    {
        using var fixture = new ProtocolFixture();
        WriteTwoObjectScene(fixture);
        WriteObjectGraph(fixture, objectName: "First");

        var graph = JsonNode.Parse(File.ReadAllText(fixture.GraphPath))!.AsObject();
        graph["links"]!.AsArray().Add(new JsonObject
        {
            ["fromNode"] = "scene",
            ["fromSocket"] = RenderGraphDocument.SceneGeometrySocketGuid,
            ["toNode"] = "object",
            ["toSocket"] = "bad-input"
        });
        File.WriteAllText(fixture.GraphPath, graph.ToJsonString());

        Assert.Contains(
            "cannot have resource inputs",
            Assert.Throws<InvalidDataException>(
                () => RenderGraphDocument.Load(fixture.GraphPath)).Message);
    }

    [Fact]
    public void TwoObjectNodesResolveToTwoDifferentObjects()
    {
        // Two nodes must not collapse onto one handle, or a pass reading both would silently receive
        // the same object twice. Asserted in one pass so the two handles are live at the same moment.
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "object-manifest.json");
        WriteTwoObjectScene(fixture);
        WriteTwoObjectGraph(fixture);
        WritePairManifest(manifestPath);
        fixture.WriteRequest(manifestPath: manifestPath);

        using var host = new RenderHostRunner();
        _ = host.RenderOnce(fixture.RequestPath);

        var pixels = File.ReadAllBytes(fixture.OutputPath)[40..];
        Assert.Equal(0, pixels[0] - 128);
        Assert.Equal(5, pixels[1] - 128);
        // Each handle keeps its own narrowed geometry, not the flattened scene's six indices.
        Assert.Equal(3, pixels[2]);
        Assert.Equal(3, pixels[4]);
    }

    private static ProbeResult RenderProbe(string objectName)
    {
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "object-manifest.json");
        WriteTwoObjectScene(fixture);
        WriteObjectGraph(fixture, objectName);
        WriteManifest(manifestPath);
        fixture.WriteRequest(manifestPath: manifestPath);

        using var host = new RenderHostRunner();
        _ = host.RenderOnce(fixture.RequestPath);
        return ReadProbe(fixture);
    }

    private static ProbeResult ReadProbe(ProtocolFixture fixture)
    {
        // Pixels begin after the frame header, which is where the camera probe reads them from too.
        var pixels = File.ReadAllBytes(fixture.OutputPath)[40..];
        return new ProbeResult(
            Exists: pixels[0] != 0,
            IndexCount: pixels[1],
            SubmeshCount: pixels[2],
            FirstSubmeshIndex: pixels[4],
            TranslationX: pixels[5] - 128.0f,
            TranslationZ: pixels[6] - 128.0f,
            FirstVertexX: pixels[8] - 128.0f,
            VertexCount: pixels[9]);
    }

    private readonly record struct ProbeResult(
        bool Exists,
        int IndexCount,
        int SubmeshCount,
        int FirstSubmeshIndex,
        float TranslationX,
        float TranslationZ,
        float FirstVertexX,
        int VertexCount);

    private static void WriteObjectGraph(ProtocolFixture fixture, string objectName)
    {
        var nodes = new JsonArray(
            new JsonObject { ["nodeId"] = "scene", ["kind"] = "scene" },
            new JsonObject
            {
                ["nodeId"] = "object",
                ["kind"] = "object",
                ["objectName"] = objectName
            },
            ProbeNode("probe"),
            new JsonObject { ["nodeId"] = "output", ["kind"] = "output" });
        var links = new JsonArray(
            Link("object", RenderGraphDocument.ObjectNodeSocketGuid, "probe", ProbeObjectInputGuid),
            Link("probe", ProbeOutputGuid, "output", RenderGraphDocument.OutputColorSocketGuid));
        WriteGraph(fixture, nodes, links, ["scene", "object", "probe", "output"]);
    }

    private static void WriteTwoObjectGraph(ProtocolFixture fixture)
    {
        var nodes = new JsonArray(
            new JsonObject { ["nodeId"] = "scene", ["kind"] = "scene" },
            new JsonObject
            {
                ["nodeId"] = "object-a",
                ["kind"] = "object",
                ["objectName"] = "First"
            },
            new JsonObject
            {
                ["nodeId"] = "object-b",
                ["kind"] = "object",
                ["objectName"] = "Second"
            },
            new JsonObject
            {
                ["nodeId"] = "probe",
                ["kind"] = "pass",
                ["passGuid"] = PairPassGuid,
                ["typeName"] = typeof(ObjectPairProbePass).FullName,
                ["muted"] = false,
                ["parameters"] = new JsonObject()
            },
            new JsonObject { ["nodeId"] = "output", ["kind"] = "output" });
        var links = new JsonArray(
            Link("object-a", RenderGraphDocument.ObjectNodeSocketGuid, "probe", PairFirstInputGuid),
            Link("object-b", RenderGraphDocument.ObjectNodeSocketGuid, "probe", PairSecondInputGuid),
            Link("probe", PairOutputGuid, "output", RenderGraphDocument.OutputColorSocketGuid));
        WriteGraph(
            fixture,
            nodes,
            links,
            ["scene", "object-a", "object-b", "probe", "output"]);
    }

    private static JsonObject ProbeNode(string nodeId)
        => new()
        {
            ["nodeId"] = nodeId,
            ["kind"] = "pass",
            ["passGuid"] = ProbePassGuid,
            ["typeName"] = typeof(ObjectProbePass).FullName,
            ["muted"] = false,
            ["parameters"] = new JsonObject()
        };

    private static void WriteGraph(
        ProtocolFixture fixture,
        JsonArray nodes,
        JsonArray links,
        string[] topologicalOrder)
    {
        var graph = new JsonObject
        {
            ["schemaVersion"] = 1,
            ["generationId"] = ProtocolFixture.GenerationId,
            ["graphId"] = "object-graph",
            ["viewId"] = "view-1",
            ["viewKind"] = "CUSTOM",
            ["executionMode"] = "REALTIME",
            ["resolutionScale"] = 1.0f,
            ["sampleCount"] = 1,
            ["nodes"] = nodes,
            ["links"] = links,
            ["topologicalOrder"] = new JsonArray([.. topologicalOrder.Select(id => (JsonNode)id!)]),
            ["output"] = new JsonObject
            {
                ["nodeId"] = "output",
                ["socketGuid"] = RenderGraphDocument.OutputColorSocketGuid,
                ["aov"] = "Combined"
            }
        };
        File.WriteAllText(
            fixture.GraphPath,
            graph.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + Environment.NewLine);
    }

    private static JsonObject Link(string fromNode, string fromSocket, string toNode, string toSocket)
        => new()
        {
            ["fromNode"] = fromNode,
            ["fromSocket"] = fromSocket,
            ["toNode"] = toNode,
            ["toSocket"] = toSocket
        };

    /// <summary>
    /// Writes a scene of two single-triangle objects, "First" at the origin and "Second" offset along
    /// X and Z, sharing one mesh and one material.
    /// </summary>
    private static void WriteTwoObjectScene(ProtocolFixture fixture)
    {
        using var payload = new MemoryStream();
        var positions = WriteFloats(payload, [0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f], [3, 3]);
        var loopVertices = WriteUInts(payload, [0, 1, 2], [3]);
        var normals = WriteFloats(payload, [0, 0, 1, 0, 0, 1, 0, 0, 1], [3, 3]);
        var triangleLoops = WriteUInts(payload, [0, 1, 2], [1, 3]);
        var triangleMaterials = WriteUInts(payload, [0], [1]);

        var metadata = new
        {
            schemaVersion = 1,
            generationId = ProtocolFixture.GenerationId,
            matrixLayout = "row-major",
            frame = 1,
            subframe = 0.0f,
            meshes = new[]
            {
                new
                {
                    meshId = "mesh-0",
                    name = "Triangle",
                    vertexCount = 3,
                    cornerCount = 3,
                    triangleCount = 1,
                    materialSlots = Array.Empty<string>(),
                    attributes = new
                    {
                        positions,
                        loopVertexIndices = loopVertices,
                        cornerNormals = normals,
                        triangleLoopIndices = triangleLoops,
                        triangleMaterialIndices = triangleMaterials
                    }
                }
            },
            instances = new[]
            {
                new
                {
                    instanceId = "instance-0",
                    name = "First",
                    meshId = "mesh-0",
                    matrixWorld = new float[]
                    {
                        1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1
                    },
                    isInstance = false
                },
                new
                {
                    instanceId = "instance-1",
                    name = "Second",
                    meshId = "mesh-0",
                    matrixWorld = new float[]
                    {
                        1, 0, 0, 5,
                        0, 1, 0, 0,
                        0, 0, 1, 3,
                        0, 0, 0, 1
                    },
                    isInstance = false
                }
            },
            materials = Array.Empty<object>(),
            lights = Array.Empty<object>(),
            camera = (object?)null
        };

        var metadataBytes = JsonSerializer.SerializeToUtf8Bytes(metadata);
        var payloadBytes = payload.ToArray();
        using var file = File.Create(fixture.ScenePath);
        Span<byte> header = stackalloc byte[24];
        "FTHSCN01"u8.CopyTo(header);
        BinaryPrimitives.WriteUInt32LittleEndian(header[8..12], 1);
        BinaryPrimitives.WriteUInt32LittleEndian(header[12..16], checked((uint)metadataBytes.Length));
        BinaryPrimitives.WriteUInt64LittleEndian(header[16..24], checked((ulong)payloadBytes.Length));
        file.Write(header);
        file.Write(metadataBytes);
        file.Write(payloadBytes);
    }

    private static Dictionary<string, object> WriteFloats(
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

    private static Dictionary<string, object> WriteUInts(
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

    private static void WriteManifest(string path)
        => WriteManifest(
            path,
            ProbePassGuid,
            typeof(ObjectProbePass),
            ProbeOutputGuid,
            [ProbeObjectInputGuid]);

    private static void WritePairManifest(string path)
        => WriteManifest(
            path,
            PairPassGuid,
            typeof(ObjectPairProbePass),
            PairOutputGuid,
            [PairFirstInputGuid, PairSecondInputGuid]);

    private static void WriteManifest(
        string path,
        string passGuid,
        Type passType,
        string outputGuid,
        string[] objectInputGuids)
    {
        var assemblyPath = typeof(ObjectNodeTests).Assembly.Location;
        var inputs = new JsonArray();
        foreach (var socketGuid in objectInputGuids)
        {
            inputs.Add(new JsonObject
            {
                ["socketGuid"] = socketGuid,
                ["resourceKind"] = "SceneObject",
                ["format"] = "Unknown"
            });
        }
        var document = new JsonObject
        {
            ["schemaVersion"] = 1,
            ["buildId"] = string.Empty,
            ["assemblyPath"] = assemblyPath,
            ["feirDirectory"] = "",
            ["projectRoot"] = ".",
            ["passes"] = new JsonArray(
                new JsonObject
                {
                    ["passGuid"] = passGuid,
                    ["typeName"] = passType.FullName,
                    ["assemblyPath"] = assemblyPath,
                    ["inputs"] = inputs,
                    ["outputs"] = new JsonArray(
                        new JsonObject
                        {
                            ["socketGuid"] = outputGuid,
                            ["resourceKind"] = "Texture2D",
                            ["format"] = "RGBA8"
                        })
                })
        };
        var options = new JsonSerializerOptions { WriteIndented = true };
        var normalized = document.ToJsonString(options);
        using var hasher = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        hasher.AppendData(Encoding.UTF8.GetBytes(normalized));
        hasher.AppendData([0]);
        hasher.AppendData(File.ReadAllBytes(assemblyPath));
        document["buildId"] = "sha256:" + Convert.ToHexString(
            hasher.GetHashAndReset()).ToLowerInvariant();
        File.WriteAllText(path, document.ToJsonString(options) + Environment.NewLine);
    }

    /// <summary>
    /// Writes what it was handed about its object into the first pixels, so a test reads the values a
    /// real pass would have received instead of inferring them from a shaded image.
    /// </summary>
    [FeatherPass(ProbePassGuid)]
    public sealed class ObjectProbePass : IComputePass
    {
        [Input(ProbeObjectInputGuid)]
        public SceneObjectHandle Object { get; init; }

        [Output(ProbeOutputGuid, Format = TextureFormat.Rgba8)]
        public TextureHandle Color { get; init; }

        public void Execute(RenderContext context)
        {
            var target = context.GetSceneObject(Object);
            var submeshes = target.Geometry.Submeshes;
            var vertices = target.Geometry.Vertices;
            var pixels = new Rgba8[checked(context.Width * context.Height)];
            Array.Fill(pixels, new Rgba8(0, 0, 0, 255));
            pixels[0] = new Rgba8(
                (byte)(target.Exists ? 1 : 0),
                (byte)System.Math.Min(target.Geometry.Indices.Length, 255),
                (byte)System.Math.Min(submeshes.Length, 255),
                255);
            // Column three is the translation, because float4x4 is column-major.
            pixels[1] = new Rgba8(
                (byte)(submeshes.Length > 0 ? submeshes.Span[0].FirstIndex : 0),
                Encode(target.ModelMatrix.C3.X),
                Encode(target.ModelMatrix.C3.Z),
                255);
            // Read through the object's own indices, which is how a pass walks its triangles: the
            // vertex buffer is the whole scene's and the indices still address it.
            var indices = target.Geometry.Indices;
            pixels[2] = new Rgba8(
                Encode(indices.Length > 0 ? vertices.Span[(int)indices.Span[0]].Position.X : 0.0f),
                (byte)System.Math.Min(vertices.Length, 255),
                0,
                255);
            context.SetColorOutput(Color, pixels);
        }

        // Offset by 128 so a negative coordinate round-trips through an unsigned byte.
        private static byte Encode(float value)
            => (byte)System.Math.Clamp((int)MathF.Round(value) + 128, 0, 255);
    }

    /// <summary>
    /// Reads two objects at once, so a test can prove two handles stay distinct while both are live.
    /// </summary>
    [FeatherPass(PairPassGuid)]
    public sealed class ObjectPairProbePass : IComputePass
    {
        [Input(PairFirstInputGuid)]
        public SceneObjectHandle First { get; init; }

        [Input(PairSecondInputGuid)]
        public SceneObjectHandle Second { get; init; }

        [Output(PairOutputGuid, Format = TextureFormat.Rgba8)]
        public TextureHandle Color { get; init; }

        public void Execute(RenderContext context)
        {
            var first = context.GetSceneObject(First);
            var second = context.GetSceneObject(Second);
            var pixels = new Rgba8[checked(context.Width * context.Height)];
            Array.Fill(pixels, new Rgba8(0, 0, 0, 255));
            pixels[0] = new Rgba8(
                (byte)System.Math.Clamp((int)MathF.Round(first.ModelMatrix.C3.X) + 128, 0, 255),
                (byte)System.Math.Clamp((int)MathF.Round(second.ModelMatrix.C3.X) + 128, 0, 255),
                (byte)System.Math.Min(first.Geometry.Indices.Length, 255),
                255);
            pixels[1] = new Rgba8(
                (byte)System.Math.Min(second.Geometry.Indices.Length, 255),
                0,
                0,
                255);
            context.SetColorOutput(Color, pixels);
        }
    }
}
