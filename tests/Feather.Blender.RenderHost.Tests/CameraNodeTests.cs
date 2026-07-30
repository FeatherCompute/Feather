using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Feather.Math;
using Feather.RenderGraph;

namespace Feather.Blender.RenderHost.Tests;

/// <summary>
/// Guards the View Camera node: a graph-visible camera whose up axis a pass can be pointed at without
/// touching the effect's code.
/// </summary>
/// <remarks>
/// Asserted by rendering a real graph rather than by unit-testing the swap, because the thing that
/// breaks is the wiring: a node kind the schema rejects, a handle the backend cannot resolve, or a
/// convention applied twice. The probe pass writes the camera it was handed into a pixel, so the
/// assertions read the value a real pass would have received.
/// </remarks>
public sealed class CameraNodeTests
{
    private const string ProbePassGuid = "6f4c1b98-3a27-4d85-9e16-2b7c5a9d4e83";
    private const string ProbeCameraInputGuid = "1e8a4d67-9c35-4b92-8f47-6d2b3e7a5c19";
    private const string ProbeOutputGuid = "9b2f7c43-5e81-4a36-b729-4c8d1a6e3b95";

    [Fact]
    public void CameraNodeHandsThePassAYUpCamera()
    {
        // Eye ten units up Blender's Z. A Y-up node must move that height onto Y, which is where a
        // procedural shader looks for it.
        var eye = RenderProbe(upAxis: "Y", cameraPosition: [1.0f, 2.0f, 10.0f]);

        Assert.Equal(1.0f, eye.X, 4);
        Assert.Equal(10.0f, eye.Y, 4);
        Assert.Equal(2.0f, eye.Z, 4);
    }

    [Fact]
    public void CameraNodeCanHandOverBlendersOwnAxes()
    {
        var eye = RenderProbe(upAxis: "Z", cameraPosition: [1.0f, 2.0f, 10.0f]);

        Assert.Equal(1.0f, eye.X, 4);
        Assert.Equal(2.0f, eye.Y, 4);
        Assert.Equal(10.0f, eye.Z, 4);
    }

    [Fact]
    public void SceneCameraSocketStillHandsOverBlendersAxes()
    {
        // The scene node predates the camera node, so its handle must keep delivering the camera
        // exactly as Blender supplied it or every existing pass would silently change convention.
        var eye = RenderProbe(upAxis: null, cameraPosition: [1.0f, 2.0f, 10.0f]);

        Assert.Equal(1.0f, eye.X, 4);
        Assert.Equal(2.0f, eye.Y, 4);
        Assert.Equal(10.0f, eye.Z, 4);
    }

    [Fact]
    public void YUpConversionAlsoFoldsTheRayMatrix()
    {
        // The eye alone is not enough: rays come from the inverse view-projection, so it has to make
        // the same trip or a marched surface would be lit from the wrong direction.
        var camera = new RenderCamera(
            float4x4.Identity,
            float4x4.Identity,
            new float3(0.0f, 0.0f, 0.0f));

        var swapped = camera.SwapUpAxis();
        var up = swapped.InverseViewProjection * new float4(0.0f, 0.0f, 1.0f, 1.0f);

        Assert.Equal(0.0f, up.X, 4);
        Assert.Equal(1.0f, up.Y, 4);
        Assert.Equal(0.0f, up.Z, 4);
    }

    [Fact]
    public void SwappingTwiceReturnsTheOriginalCamera()
    {
        var camera = new RenderCamera(
            float4x4.Identity,
            float4x4.Identity,
            new float3(3.0f, -4.0f, 5.0f));

        var round = camera.SwapUpAxis().SwapUpAxis();

        Assert.Equal(3.0f, round.WorldPosition.X, 5);
        Assert.Equal(-4.0f, round.WorldPosition.Y, 5);
        Assert.Equal(5.0f, round.WorldPosition.Z, 5);
    }

    [Fact]
    public void RasterizerMatrixIsLeftAloneByTheSwap()
    {
        // ViewProjection carries the rasterizer's viewport fixup and is documented as not the mutual
        // inverse of InverseViewProjection. Swapping it too would corrupt the raster path.
        var raster = new float4x4(
            new float4(2.0f, 0.0f, 0.0f, 0.0f),
            new float4(0.0f, 3.0f, 0.0f, 0.0f),
            new float4(0.0f, 0.0f, 4.0f, 0.0f),
            new float4(0.0f, 0.0f, 0.0f, 1.0f));
        var camera = new RenderCamera(raster, float4x4.Identity, new float3(0.0f, 0.0f, 0.0f));

        var swapped = camera.SwapUpAxis();

        Assert.Equal(2.0f, swapped.ViewProjection.C0.X, 5);
        Assert.Equal(3.0f, swapped.ViewProjection.C1.Y, 5);
        Assert.Equal(4.0f, swapped.ViewProjection.C2.Z, 5);
    }

    [Fact]
    public void UnknownUpAxisIsRejectedRatherThanDefaulted()
    {
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "camera-manifest.json");
        fixture.WriteScene();
        WriteCameraGraph(fixture, upAxis: "Q");
        WriteManifest(manifestPath);
        fixture.WriteRequest(manifestPath: manifestPath);

        // Defaulting would render the world on its side, which is far harder to diagnose than this.
        Assert.Contains(
            "upAxis",
            Assert.Throws<InvalidDataException>(
                () => RenderGraphDocument.Load(fixture.GraphPath)).Message);
    }

    [Fact]
    public void CameraNodeCannotBeGivenInputs()
    {
        using var fixture = new ProtocolFixture();
        fixture.WriteScene();
        WriteCameraGraph(fixture, upAxis: "Y");

        var graph = JsonNode.Parse(File.ReadAllText(fixture.GraphPath))!.AsObject();
        graph["links"]!.AsArray().Add(new JsonObject
        {
            ["fromNode"] = "scene",
            ["fromSocket"] = RenderGraphDocument.SceneGeometrySocketGuid,
            ["toNode"] = "camera",
            ["toSocket"] = "bad-input"
        });
        File.WriteAllText(fixture.GraphPath, graph.ToJsonString());

        Assert.Contains(
            "cannot have resource inputs",
            Assert.Throws<InvalidDataException>(
                () => RenderGraphDocument.Load(fixture.GraphPath)).Message);
    }

    /// <summary>
    /// Renders a graph whose probe pass writes the camera eye it received, and returns that eye.
    /// </summary>
    /// <param name="upAxis">
    /// The node's up axis, or null to wire the probe to the scene node's camera socket instead.
    /// </param>
    private static float3 RenderProbe(string? upAxis, float[] cameraPosition)
    {
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "camera-manifest.json");
        fixture.WriteScene();
        WriteCameraGraph(fixture, upAxis);
        WriteManifest(manifestPath);
        fixture.WriteRequest(manifestPath: manifestPath, cameraPosition: cameraPosition);

        using var host = new RenderHostRunner();
        _ = host.RenderOnce(fixture.RequestPath);

        // The probe encodes each axis as a byte offset by 128, so a negative coordinate survives.
        var pixel = File.ReadAllBytes(fixture.OutputPath)[40..44];
        return new float3(pixel[0] - 128.0f, pixel[1] - 128.0f, pixel[2] - 128.0f);
    }

    private static void WriteCameraGraph(ProtocolFixture fixture, string? upAxis)
    {
        var nodes = new JsonArray(
            new JsonObject { ["nodeId"] = "scene", ["kind"] = "scene" });
        var links = new JsonArray();
        var order = new JsonArray("scene");

        if (upAxis is null)
        {
            links.Add(Link(
                "scene",
                RenderGraphDocument.SceneCameraSocketGuid,
                "probe",
                ProbeCameraInputGuid));
        }
        else
        {
            nodes.Add(new JsonObject
            {
                ["nodeId"] = "camera",
                ["kind"] = "camera",
                ["upAxis"] = upAxis
            });
            order.Add("camera");
            links.Add(Link(
                "camera",
                RenderGraphDocument.CameraNodeSocketGuid,
                "probe",
                ProbeCameraInputGuid));
        }

        nodes.Add(new JsonObject
        {
            ["nodeId"] = "probe",
            ["kind"] = "pass",
            ["passGuid"] = ProbePassGuid,
            ["typeName"] = typeof(CameraProbePass).FullName,
            ["muted"] = false,
            ["parameters"] = new JsonObject()
        });
        nodes.Add(new JsonObject { ["nodeId"] = "output", ["kind"] = "output" });
        order.Add("probe");
        order.Add("output");
        links.Add(Link(
            "probe",
            ProbeOutputGuid,
            "output",
            RenderGraphDocument.OutputColorSocketGuid));

        var graph = new JsonObject
        {
            ["schemaVersion"] = 1,
            ["generationId"] = ProtocolFixture.GenerationId,
            ["graphId"] = "camera-graph",
            ["viewId"] = "view-1",
            ["viewKind"] = "CUSTOM",
            ["executionMode"] = "REALTIME",
            ["resolutionScale"] = 1.0f,
            ["sampleCount"] = 1,
            ["nodes"] = nodes,
            ["links"] = links,
            ["topologicalOrder"] = order,
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

    private static void WriteManifest(string path)
    {
        var assemblyPath = typeof(CameraNodeTests).Assembly.Location;
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
                    ["passGuid"] = ProbePassGuid,
                    ["typeName"] = typeof(CameraProbePass).FullName,
                    ["assemblyPath"] = assemblyPath,
                    ["inputs"] = new JsonArray(
                        new JsonObject
                        {
                            ["socketGuid"] = ProbeCameraInputGuid,
                            ["resourceKind"] = "Camera",
                            ["format"] = "Unknown"
                        }),
                    ["outputs"] = new JsonArray(
                        new JsonObject
                        {
                            ["socketGuid"] = ProbeOutputGuid,
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
    /// Writes the camera eye it was handed into every pixel, so a test can read back the convention
    /// the graph resolved rather than inferring it from a shaded image.
    /// </summary>
    [FeatherPass(ProbePassGuid)]
    public sealed class CameraProbePass : IComputePass
    {
        [Input(ProbeCameraInputGuid)]
        public CameraHandle Camera { get; init; }

        [Output(ProbeOutputGuid, Format = TextureFormat.Rgba8)]
        public TextureHandle Color { get; init; }

        public void Execute(RenderContext context)
        {
            var eye = context.GetCamera(Camera).WorldPosition;
            var encoded = new Rgba8(
                Encode(eye.X),
                Encode(eye.Y),
                Encode(eye.Z),
                255);
            var pixels = new Rgba8[checked(context.Width * context.Height)];
            Array.Fill(pixels, encoded);
            context.SetColorOutput(Color, pixels);
        }

        // Offset by 128 so a negative coordinate round-trips through an unsigned byte.
        private static byte Encode(float value)
            => (byte)System.Math.Clamp((int)MathF.Round(value) + 128, 0, 255);
    }
}
