using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Feather.Math;
using Feather.RenderGraph;

namespace Feather.Blender.RenderHost.Tests;

public sealed class RenderHostProtocolTests
{
    private const string BufferProducerPassGuid = "a13d9b28-f076-4542-a807-3566f27f735a";
    private const string BufferEvaluatePassGuid = "be95613d-4edd-43e7-97cf-3947ceedda48";
    private const string BufferVisualizePassGuid = "dabc09ca-cbc6-4890-a886-acce544f8938";
    private const string DirectionsOutputGuid = "e1505ffa-901e-4b3f-9b7b-2ae44757a2fc";
    private const string WeightsOutputGuid = "1df28dd5-196c-40a8-941d-a7aac353fe3f";
    private const string DirectionsInputGuid = "8354dcfd-fe3c-46e8-b172-c065c28fbae5";
    private const string WeightsInputGuid = "3aaddfe4-d900-4215-a20f-04ed5f07ca68";
    private const string ScoresOutputGuid = "0bc16d56-6333-4b8e-91b5-28d7fa40f1fc";
    private const string ScoresInputGuid = "bf7f71a7-f86b-4769-8af3-a5e338c6a856";
    private const string BufferColorOutputGuid = "aeaea2ce-b9da-484c-921a-c41e788656d6";

    [Fact]
    public void PublicationGenerationDoesNotChangeSceneOrGraphContentIdentity()
    {
        using var fixture = new ProtocolFixture();
        fixture.WriteScene();
        fixture.WriteGraph();
        var firstScene = SceneSnapshot.Load(fixture.ScenePath);
        var firstGraph = RenderGraphDocument.Load(fixture.GraphPath);
        const string replacementGeneration = "e53e3108-9276-4440-b503-94c45e094e3f";

        var sceneBytes = File.ReadAllBytes(fixture.ScenePath);
        var generationBytes = Encoding.UTF8.GetBytes(ProtocolFixture.GenerationId);
        var generationOffset = sceneBytes.AsSpan().IndexOf(generationBytes);
        Assert.True(generationOffset >= 0);
        Encoding.UTF8.GetBytes(replacementGeneration).CopyTo(sceneBytes, generationOffset);
        File.WriteAllBytes(fixture.ScenePath, sceneBytes);

        var graphJson = JsonNode.Parse(File.ReadAllText(fixture.GraphPath))!.AsObject();
        graphJson["generationId"] = replacementGeneration;
        File.WriteAllText(fixture.GraphPath, graphJson.ToJsonString());
        var secondScene = SceneSnapshot.Load(fixture.ScenePath);
        var secondGraph = RenderGraphDocument.Load(fixture.GraphPath);

        Assert.Equal(firstScene.ContentFingerprint, secondScene.ContentFingerprint);
        Assert.Equal(firstGraph.GraphFingerprint, secondGraph.GraphFingerprint);
    }

    [Fact]
    public void BlenderSceneV1BuildsIndexedEvaluatedGeometryWithInstanceTransform()
    {
        using var fixture = new ProtocolFixture();
        fixture.WriteScene(
        [
            2, 0, 0, 3,
            0, 2, 0, 4,
            0, 0, 2, 5,
            0, 0, 0, 1
        ]);

        var snapshot = SceneSnapshot.Load(fixture.ScenePath);
        var geometry = SceneGeometryBuilder.Build(snapshot);

        Assert.Equal([0u, 1u, 2u], geometry.Indices);
        Assert.Equal(3, geometry.Vertices.Length);
        Assert.Equal(new float3(1.5f, 2.7f, 6.0f), geometry.Vertices[0].Position);
        Assert.Equal(new float3(0.0f, 0.0f, 1.0f), geometry.Vertices[0].Normal);
    }

    [Fact]
    public void SceneArrayOutsidePayloadIsRejectedBeforeRendering()
    {
        using var fixture = new ProtocolFixture();
        fixture.WriteScene(invalidPositionsOffset: true);

        var snapshot = SceneSnapshot.Load(fixture.ScenePath);
        var exception = Assert.Throws<InvalidDataException>(() => SceneGeometryBuilder.Build(snapshot));

        Assert.Contains("outside the snapshot payload", exception.Message);
    }

    [Fact]
    public void NonUniformInstanceScaleUsesInverseTransposeNormalMatrix()
    {
        using var fixture = new ProtocolFixture();
        fixture.WriteScene(
        [
            2, 0, 0, 0,
            0, 4, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ],
        cornerNormals:
        [
            1, 1, 0,
            1, 1, 0,
            1, 1, 0
        ]);

        var geometry = SceneGeometryBuilder.Build(SceneSnapshot.Load(fixture.ScenePath));

        Assert.Equal(0.8944272f, geometry.Vertices[0].Normal.X, 5);
        Assert.Equal(0.4472136f, geometry.Vertices[0].Normal.Y, 5);
        Assert.Equal(0.0f, geometry.Vertices[0].Normal.Z, 5);
    }

    [Fact]
    public void BlenderOpenGlViewProjectionIsConvertedToVulkanClipSpace()
    {
        using var fixture = new ProtocolFixture();
        fixture.WriteRequest(clipSpace: "blender-opengl");

        var request = RenderRequest.Load(fixture.RequestPath);

        Assert.Equal(-1.0f, request.ViewProjection.M11);
        Assert.Equal(0.5f, request.ViewProjection.M22);
        Assert.Equal(0.5f, request.ViewProjection.M23);
        Assert.Equal(1.0f, request.ViewProjection.M33);
    }

    [Fact]
    public void UnsupportedPassGraphIsRejected()
    {
        using var fixture = new ProtocolFixture();
        fixture.WriteGraph(passGuid: "00000000-0000-0000-0000-000000000000");

        var exception = Assert.Throws<InvalidDataException>(
            () => RenderGraphDocument.LoadMinimalRaster(fixture.GraphPath));

        Assert.Contains("only MinimalRaster", exception.Message);
    }

    [Fact]
    public void BlenderParameterDefinitionArrayAndProjectSpecificTypeAreAccepted()
    {
        using var fixture = new ProtocolFixture();
        fixture.WriteGraph(
            typeName: "TerrainExperiment.Passes.MinimalRasterPass",
            parameterDefinitions: true);

        var graph = RenderGraphDocument.LoadMinimalRaster(fixture.GraphPath);

        Assert.Equal("view-1", graph.ViewId);
        Assert.Equal(SampleCount.X1, graph.SampleCount);
        Assert.Equal(0.24f, graph.Settings.Ambient);
    }

    [Fact]
    public void GraphTopologicalOrderMustRespectLinks()
    {
        using var fixture = new ProtocolFixture();
        fixture.WriteGraph();
        var json = File.ReadAllText(fixture.GraphPath);
        File.WriteAllText(
            fixture.GraphPath,
            json.Replace(
                "[\"scene-1\",\"pass-1\",\"output-1\"]",
                "[\"scene-1\",\"output-1\",\"pass-1\"]",
                StringComparison.Ordinal));

        var exception = Assert.Throws<InvalidDataException>(
            () => RenderGraphDocument.LoadMinimalRaster(fixture.GraphPath));

        Assert.Contains("violates a resource link", exception.Message);
    }

    [Fact]
    public void MinimalRasterRequiresExactSceneAndColorSocketLinks()
    {
        using var fixture = new ProtocolFixture();
        fixture.WriteGraph();
        var graph = JsonNode.Parse(File.ReadAllText(fixture.GraphPath))!.AsObject();
        graph["links"]!.AsArray().RemoveAt(0);
        File.WriteAllText(fixture.GraphPath, graph.ToJsonString());

        var exception = Assert.Throws<InvalidDataException>(
            () => RenderGraphDocument.LoadMinimalRaster(fixture.GraphPath));

        Assert.Contains("Geometry input", exception.Message);
    }

    [Fact]
    public void GraphRejectsMultipleLinksToOneInput()
    {
        using var fixture = new ProtocolFixture();
        fixture.WriteGraph();
        var graph = JsonNode.Parse(File.ReadAllText(fixture.GraphPath))!.AsObject();
        var links = graph["links"]!.AsArray();
        links.Add(links[0]!.DeepClone());
        File.WriteAllText(fixture.GraphPath, graph.ToJsonString());

        var exception = Assert.Throws<InvalidDataException>(
            () => RenderGraphDocument.LoadMinimalRaster(fixture.GraphPath));

        Assert.Contains("multiple links", exception.Message);
    }

    [Fact]
    public void RunnerRejectsMixedRequestGenerationBeforeGpuWork()
    {
        using var fixture = new ProtocolFixture();
        fixture.WriteScene();
        fixture.WriteGraph();
        fixture.WriteRequest();
        var request = JsonNode.Parse(File.ReadAllText(fixture.RequestPath))!.AsObject();
        request["generationId"] = "e53e3108-9276-4440-b503-94c45e094e3f";
        File.WriteAllText(fixture.RequestPath, request.ToJsonString());
        using var host = new RenderHostRunner();

        var exception = Assert.Throws<InvalidDataException>(() => host.RenderOnce(fixture.RequestPath));

        Assert.Contains("does not match graph generation", exception.Message);
        Assert.False(File.Exists(fixture.OutputPath));
    }

    [Theory]
    [InlineData(null, 100)]
    [InlineData("INTERACTIVE", 100)]
    [InlineData("FINAL", 200)]
    public void RequestPurposeReachesThePassAndTheHostResult(string? purpose, byte expected)
    {
        // A pass could not tell a viewport preview from the frame the user keeps, so any pass wanting
        // to trade quality for latency had to pick one and be wrong half the time. The purpose rides
        // on the request rather than the graph because one View serves both paths.
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "pass-manifest.json");
        fixture.WriteScene();
        fixture.WriteGraph(typeName: typeof(PurposeProbeCpuPass).FullName!);
        WritePassManifest(manifestPath, typeof(PurposeProbeCpuPass));
        fixture.WriteRequest(manifestPath: manifestPath, purpose: purpose);
        using var host = new RenderHostRunner();

        var result = host.RenderOnce(fixture.RequestPath);
        var frame = File.ReadAllBytes(fixture.OutputPath);

        // Red carries Purpose and green carries IsInteractive; equal bytes mean the two agree.
        Assert.Equal(expected, frame[40]);
        Assert.Equal(expected, frame[41]);
        Assert.Equal(expected == 200 ? "FINAL" : "INTERACTIVE", result.Purpose);
    }

    [Fact]
    public void UnknownRequestPurposeIsRejectedRatherThanTreatedAsAPreview()
    {
        // Defaulting an unrecognised purpose would mean a future host silently rendering a saved
        // frame at preview quality, which is the failure this whole field exists to prevent.
        using var fixture = new ProtocolFixture();
        fixture.WriteScene();
        fixture.WriteGraph();
        fixture.WriteRequest(purpose: "BATCH");
        using var host = new RenderHostRunner();

        var exception = Assert.Throws<InvalidDataException>(() => host.RenderOnce(fixture.RequestPath));

        Assert.Contains("Unsupported render request purpose", exception.Message);
        Assert.False(File.Exists(fixture.OutputPath));
    }

    [Fact]
    public void ProjectPassAssemblyLoadsReloadsAndUnloadsWithoutGpuWork()
    {
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "pass-manifest.json");
        fixture.WriteScene();
        fixture.WriteGraph(typeName: typeof(RedCpuPass).FullName!);
        WritePassManifest(manifestPath, typeof(RedCpuPass));
        var firstManifest = File.ReadAllText(manifestPath);
        fixture.WriteRequest(manifestPath: manifestPath);
        using var host = new RenderHostRunner();

        var first = host.RenderOnce(fixture.RequestPath);
        var firstFrame = File.ReadAllBytes(fixture.OutputPath);

        Assert.True(first.PassReloaded);
        Assert.Equal(typeof(RedCpuPass).FullName, first.PassType);
        Assert.Equal(1, first.PassCount);
        Assert.StartsWith("sha256:", first.BuildId, StringComparison.Ordinal);
        Assert.Equal(new byte[] { 240, 20, 30, 255 }, firstFrame[40..44]);

        var brokenManifest = JsonNode.Parse(firstManifest)!.AsObject();
        brokenManifest["assemblyPath"] = "missing-pass-assembly.dll";
        brokenManifest["passes"]![0]!["assemblyPath"] = "missing-pass-assembly.dll";
        File.WriteAllText(manifestPath, brokenManifest.ToJsonString());
        Assert.Throws<FileNotFoundException>(() => host.RenderOnce(fixture.RequestPath));
        File.WriteAllText(manifestPath, firstManifest);
        var recovered = host.RenderOnce(fixture.RequestPath);
        Assert.False(recovered.PassReloaded);
        Assert.Equal(first.BuildId, recovered.BuildId);

        fixture.WriteGraph(typeName: typeof(BlueCpuPass).FullName!);
        WritePassManifest(manifestPath, typeof(BlueCpuPass));
        var second = host.RenderOnce(fixture.RequestPath);
        var secondFrame = File.ReadAllBytes(fixture.OutputPath);

        Assert.True(second.PassReloaded);
        Assert.NotEqual(first.BuildId, second.BuildId);
        Assert.Equal(typeof(BlueCpuPass).FullName, second.PassType);
        Assert.Equal(1, second.PassCount);
        Assert.Equal(new byte[] { 20, 30, 240, 255 }, secondFrame[40..44]);
        Assert.NotNull(host.LastUnloadedPassContextForTesting);
        AssertEventuallyUnloaded(host.LastUnloadedPassContextForTesting!);

        var third = host.RenderOnce(fixture.RequestPath);
        Assert.False(third.PassReloaded);
        Assert.Equal(second.BuildId, third.BuildId);
    }

    [Fact]
    public void TwoPassGraphPropagatesColorResourceAndReportsExecutedPassCount()
    {
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "pass-manifest.json");
        fixture.WriteScene();
        fixture.WriteTwoPassGraph(
            typeof(RedCpuPass).FullName!,
            typeof(SwapRedBlueCpuPass).FullName!);
        WritePassManifest(
            manifestPath,
            MinimalRasterManifestPass(typeof(RedCpuPass)),
            PostProcessManifestPass(typeof(SwapRedBlueCpuPass)));
        fixture.WriteRequest(manifestPath: manifestPath);
        using var host = new RenderHostRunner();

        var result = host.RenderOnce(fixture.RequestPath);
        var frame = File.ReadAllBytes(fixture.OutputPath);

        Assert.Equal(2, result.PassCount);
        Assert.Equal(typeof(SwapRedBlueCpuPass).FullName, result.PassType);
        Assert.Equal(new byte[] { 30, 20, 240, 255 }, frame[40..44]);
    }

    [Fact]
    public void TypedBufferGraphTransfersFloat3IntAndFloatArraysBetweenPasses()
    {
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "pass-manifest.json");
        fixture.WriteScene();
        WriteTypedBufferGraph(fixture);
        WritePassManifest(manifestPath, TypedBufferManifestPasses());
        fixture.WriteRequest(manifestPath: manifestPath);
        using var host = new RenderHostRunner();

        var result = host.RenderOnce(fixture.RequestPath);
        var frame = File.ReadAllBytes(fixture.OutputPath);

        Assert.Equal(3, result.PassCount);
        Assert.Equal(typeof(BufferVisualizeCpuPass).FullName, result.PassType);
        Assert.Equal(new byte[] { 60, 150, 2, 255 }, frame[40..44]);
    }

    [Fact]
    public void TypedBufferGraphAcceptsCanonicalElementTypeAliases()
    {
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "pass-manifest.json");
        fixture.WriteScene();
        WriteTypedBufferGraph(fixture);
        var passes = TypedBufferManifestPasses();
        passes[0] = passes[0] with
        {
            Outputs =
            [
                passes[0].Outputs[0] with { ElementType = "global::Feather.Math.float3" },
                passes[0].Outputs[1] with { ElementType = "System.Int32" }
            ]
        };
        passes[1] = passes[1] with
        {
            Inputs =
            [
                passes[1].Inputs[0] with { ElementType = "Feather.Math.float3" },
                passes[1].Inputs[1] with { ElementType = "int32" }
            ],
            Outputs = [passes[1].Outputs[0] with { ElementType = "System.Single" }]
        };
        passes[2] = passes[2] with
        {
            Inputs = [passes[2].Inputs[0] with { ElementType = "single" }]
        };
        WritePassManifest(manifestPath, passes);
        fixture.WriteRequest(manifestPath: manifestPath);
        using var host = new RenderHostRunner();

        var result = host.RenderOnce(fixture.RequestPath);
        var frame = File.ReadAllBytes(fixture.OutputPath);

        Assert.Equal(3, result.PassCount);
        Assert.Equal(new byte[] { 60, 150, 2, 255 }, frame[40..44]);
    }

    [Fact]
    public void TypedBufferGraphRejectsMismatchedElementTypesBeforeExecution()
    {
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "pass-manifest.json");
        fixture.WriteScene();
        WriteTypedBufferGraph(fixture);
        var passes = TypedBufferManifestPasses();
        passes[1] = passes[1] with
        {
            Inputs =
            [
                passes[1].Inputs[0] with { ElementType = "float" },
                passes[1].Inputs[1]
            ]
        };
        WritePassManifest(manifestPath, passes);
        fixture.WriteRequest(manifestPath: manifestPath);
        using var host = new RenderHostRunner();

        var exception = Assert.Throws<InvalidDataException>(
            () => host.RenderOnce(fixture.RequestPath));

        Assert.Contains("element type float3, expected float", exception.Message);
        Assert.False(File.Exists(fixture.OutputPath));
    }

    [Fact]
    public void RenderHostHistoryReadsPreviousOutputAndResetsWhenViewProjectionChanges()
    {
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "pass-manifest.json");
        fixture.WriteScene();
        fixture.WriteHistoryGraph(typeof(HistoryProbeCpuPass).FullName!);
        WritePassManifest(manifestPath, HistoryProbeManifestPass(typeof(HistoryProbeCpuPass)));
        fixture.WriteRequest(manifestPath: manifestPath);
        using var host = new RenderHostRunner();

        var first = host.RenderOnce(fixture.RequestPath);
        var firstPixel = File.ReadAllBytes(fixture.OutputPath)[40..44];

        Assert.Equal(new byte[] { 0, 0, 0, 255 }, firstPixel);
        Assert.True(first.HistoryReset);
        Assert.Equal(1, first.ResetCount);
        Assert.Equal(1, first.Iteration);
        Assert.True(first.PassReloaded);

        fixture.WriteRequest(manifestPath: manifestPath, requestId: 43);
        var second = host.RenderOnce(fixture.RequestPath);
        var secondPixel = File.ReadAllBytes(fixture.OutputPath)[40..44];

        Assert.Equal(new byte[] { 37, 73, 109, 255 }, secondPixel);
        Assert.False(second.HistoryReset);
        Assert.Equal(1, second.ResetCount);
        Assert.Equal(2, second.Iteration);
        Assert.False(second.PassReloaded);

        fixture.WriteRequest(
            manifestPath: manifestPath,
            requestId: 44,
            viewProjection:
            [
                2, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
            ]);
        var reset = host.RenderOnce(fixture.RequestPath);
        var resetPixel = File.ReadAllBytes(fixture.OutputPath)[40..44];

        Assert.Equal(new byte[] { 0, 0, 0, 255 }, resetPixel);
        Assert.True(reset.HistoryReset);
        Assert.Equal(2, reset.ResetCount);
        Assert.Equal(1, reset.Iteration);
        Assert.False(reset.PassReloaded);
    }

    [Fact]
    public void MutedTexturePassBypassesItsInputAndIsNotCountedAsExecuted()
    {
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "pass-manifest.json");
        fixture.WriteScene();
        fixture.WriteTwoPassGraph(
            typeof(RedCpuPass).FullName!,
            typeof(SwapRedBlueCpuPass).FullName!,
            secondPassMuted: true);
        WritePassManifest(
            manifestPath,
            MinimalRasterManifestPass(typeof(RedCpuPass)),
            PostProcessManifestPass(typeof(SwapRedBlueCpuPass)));
        fixture.WriteRequest(manifestPath: manifestPath);
        using var host = new RenderHostRunner();

        var result = host.RenderOnce(fixture.RequestPath);
        var frame = File.ReadAllBytes(fixture.OutputPath);

        Assert.Equal(1, result.PassCount);
        Assert.Equal(typeof(RedCpuPass).FullName, result.PassType);
        Assert.Equal(new byte[] { 240, 20, 30, 255 }, frame[40..44]);
    }

    [Theory]
    [InlineData("SceneGeometry", "RGBA8", "resource kind")]
    [InlineData("Texture2D", "R8", "format")]
    public void GraphRejectsIncompatiblePassResourceLinks(
        string postInputKind,
        string postInputFormat,
        string expectedMessage)
    {
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "pass-manifest.json");
        fixture.WriteScene();
        fixture.WriteTwoPassGraph(
            typeof(RedCpuPass).FullName!,
            typeof(SwapRedBlueCpuPass).FullName!);
        WritePassManifest(
            manifestPath,
            MinimalRasterManifestPass(typeof(RedCpuPass)),
            PostProcessManifestPass(
                typeof(SwapRedBlueCpuPass),
                postInputKind,
                postInputFormat));
        fixture.WriteRequest(manifestPath: manifestPath);
        using var host = new RenderHostRunner();

        var exception = Assert.Throws<InvalidDataException>(
            () => host.RenderOnce(fixture.RequestPath));

        Assert.Contains(expectedMessage, exception.Message, StringComparison.OrdinalIgnoreCase);
        Assert.False(File.Exists(fixture.OutputPath));
    }

    [Fact]
    public void GraphRejectsUnconnectedRequiredPassInput()
    {
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "pass-manifest.json");
        fixture.WriteScene();
        fixture.WriteTwoPassGraph(
            typeof(RedCpuPass).FullName!,
            typeof(SwapRedBlueCpuPass).FullName!);
        var graph = JsonNode.Parse(File.ReadAllText(fixture.GraphPath))!.AsObject();
        var links = graph["links"]!.AsArray();
        links.Remove(links.Single(link =>
            string.Equals(
                link!["toSocket"]!.GetValue<string>(),
                ProtocolFixture.PostProcessInputSocketGuid,
                StringComparison.Ordinal)));
        File.WriteAllText(fixture.GraphPath, graph.ToJsonString());
        WritePassManifest(
            manifestPath,
            MinimalRasterManifestPass(typeof(RedCpuPass)),
            PostProcessManifestPass(typeof(SwapRedBlueCpuPass)));
        fixture.WriteRequest(manifestPath: manifestPath);
        using var host = new RenderHostRunner();

        var exception = Assert.Throws<InvalidDataException>(
            () => host.RenderOnce(fixture.RequestPath));

        Assert.Contains("is not connected", exception.Message);
        Assert.False(File.Exists(fixture.OutputPath));
    }

    [Fact]
    public void DisconnectedDraftPassDoesNotInvalidateTheExecutableGraph()
    {
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "pass-manifest.json");
        fixture.WriteScene();
        fixture.WriteGraph(typeName: typeof(RedCpuPass).FullName!);
        var graph = JsonNode.Parse(File.ReadAllText(fixture.GraphPath))!.AsObject();
        graph["nodes"]!.AsArray().Add(new JsonObject
        {
            ["nodeId"] = "disconnected-draft",
            ["kind"] = "pass",
            ["passGuid"] = ProtocolFixture.PostProcessPassGuid,
            ["typeName"] = typeof(SwapRedBlueCpuPass).FullName,
            ["muted"] = false,
            ["parameters"] = new JsonObject()
        });
        graph["topologicalOrder"]!.AsArray().Add("disconnected-draft");
        File.WriteAllText(fixture.GraphPath, graph.ToJsonString());
        WritePassManifest(
            manifestPath,
            MinimalRasterManifestPass(typeof(RedCpuPass)),
            PostProcessManifestPass(typeof(SwapRedBlueCpuPass)));
        fixture.WriteRequest(manifestPath: manifestPath);
        using var host = new RenderHostRunner();

        var result = host.RenderOnce(fixture.RequestPath);

        Assert.Equal(1, result.PassCount);
        Assert.Equal(typeof(RedCpuPass).FullName, result.PassType);
        Assert.Equal(
            new byte[] { 240, 20, 30, 255 },
            File.ReadAllBytes(fixture.OutputPath)[40..44]);
    }

    [Fact]
    public void FrameWriterMatchesBlenderFrameV1AndLeavesNoTemporaryFile()
    {
        using var fixture = new ProtocolFixture();
        var pixels = new[]
        {
            new Rgba8(1, 2, 3, 4),
            new Rgba8(5, 6, 7, 8)
        };

        FrameFileWriter.WriteAtomic(
            fixture.OutputPath,
            99,
            new RenderedFrame(2, 1, pixels, DispatchPath.TypedEasyGpu));

        var bytes = File.ReadAllBytes(fixture.OutputPath);
        Assert.Equal("FTHRFRM1"u8.ToArray(), bytes[..8]);
        Assert.Equal((ushort)1, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(8, 2)));
        Assert.Equal((ushort)40, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(10, 2)));
        Assert.Equal((ushort)1, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(12, 2)));
        Assert.Equal((ushort)2, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(14, 2)));
        Assert.Equal(2u, BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(16, 4)));
        Assert.Equal(1u, BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(20, 4)));
        Assert.Equal(99ul, BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(32, 8)));
        Assert.Equal(new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 }, bytes[40..]);
        Assert.Empty(Directory.GetFiles(fixture.Root, "*.tmp"));
    }

    private static void WritePassManifest(string path, Type passType)
        => WritePassManifest(path, MinimalRasterManifestPass(passType));

    private static void WriteTypedBufferGraph(ProtocolFixture fixture)
    {
        var graph = new JsonObject
        {
            ["schemaVersion"] = 1,
            ["generationId"] = ProtocolFixture.GenerationId,
            ["graphId"] = "typed-buffer-graph",
            ["viewId"] = "view-1",
            ["executionMode"] = "REALTIME",
            ["resolutionScale"] = 1.0f,
            ["sampleCount"] = 1,
            ["nodes"] = new JsonArray(
                new JsonObject { ["nodeId"] = "scene-1", ["kind"] = "scene" },
                new JsonObject
                {
                    ["nodeId"] = "produce",
                    ["kind"] = "pass",
                    ["passGuid"] = BufferProducerPassGuid,
                    ["typeName"] = typeof(BufferProducerCpuPass).FullName,
                    ["muted"] = false,
                    ["parameters"] = new JsonObject()
                },
                new JsonObject
                {
                    ["nodeId"] = "evaluate",
                    ["kind"] = "pass",
                    ["passGuid"] = BufferEvaluatePassGuid,
                    ["typeName"] = typeof(BufferEvaluateCpuPass).FullName,
                    ["muted"] = false,
                    ["parameters"] = new JsonObject()
                },
                new JsonObject
                {
                    ["nodeId"] = "visualize",
                    ["kind"] = "pass",
                    ["passGuid"] = BufferVisualizePassGuid,
                    ["typeName"] = typeof(BufferVisualizeCpuPass).FullName,
                    ["muted"] = false,
                    ["parameters"] = new JsonObject()
                },
                new JsonObject { ["nodeId"] = "output-1", ["kind"] = "output" }),
            ["links"] = new JsonArray(
                Link("produce", DirectionsOutputGuid, "evaluate", DirectionsInputGuid),
                Link("produce", WeightsOutputGuid, "evaluate", WeightsInputGuid),
                Link("evaluate", ScoresOutputGuid, "visualize", ScoresInputGuid),
                Link("visualize", BufferColorOutputGuid, "output-1", RenderGraphDocument.OutputColorSocketGuid)),
            ["topologicalOrder"] = new JsonArray(
                "scene-1", "produce", "evaluate", "visualize", "output-1"),
            ["output"] = new JsonObject
            {
                ["nodeId"] = "output-1",
                ["socketGuid"] = RenderGraphDocument.OutputColorSocketGuid
            }
        };
        File.WriteAllText(fixture.GraphPath, graph.ToJsonString());

        static JsonNode Link(string fromNode, string fromSocket, string toNode, string toSocket)
            => new JsonObject
            {
                ["fromNode"] = fromNode,
                ["fromSocket"] = fromSocket,
                ["toNode"] = toNode,
                ["toSocket"] = toSocket
            };
    }

    private static void WritePassManifest(string path, params ManifestPass[] passes)
    {
        var assemblyPath = passes[0].PassType.Assembly.Location;
        if (passes.Any(pass => !string.Equals(
                pass.PassType.Assembly.Location,
                assemblyPath,
                StringComparison.Ordinal)))
        {
            throw new ArgumentException("Test manifest pass types must come from one assembly.", nameof(passes));
        }
        var document = new JsonObject
        {
            ["schemaVersion"] = 1,
            ["buildId"] = string.Empty,
            ["assemblyPath"] = assemblyPath,
            ["feirDirectory"] = "",
            ["projectRoot"] = ".",
            ["passes"] = new JsonArray(passes.Select(pass =>
                (JsonNode)new JsonObject
                {
                    ["passGuid"] = pass.PassGuid,
                    ["typeName"] = pass.PassType.FullName,
                    ["assemblyPath"] = assemblyPath,
                    ["inputs"] = new JsonArray(pass.Inputs.Select(SocketJson).ToArray()),
                    ["outputs"] = new JsonArray(pass.Outputs.Select(SocketJson).ToArray())
                }).ToArray())
        };
        var options = new JsonSerializerOptions { WriteIndented = true };
        var hashInput = document.ToJsonString(options);
        using var hasher = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        hasher.AppendData(Encoding.UTF8.GetBytes(hashInput));
        hasher.AppendData([0]);
        hasher.AppendData(File.ReadAllBytes(assemblyPath));
        document["buildId"] = "sha256:" + Convert.ToHexString(
            hasher.GetHashAndReset()).ToLowerInvariant();
        File.WriteAllText(path, document.ToJsonString(options) + Environment.NewLine);
    }

    private static ManifestPass MinimalRasterManifestPass(Type passType)
        => new(
            passType,
            RenderGraphDocument.MinimalRasterPassGuid,
            [
                new(RenderGraphDocument.GeometryInputSocketGuid, "SceneGeometry", "Unknown"),
                new(RenderGraphDocument.MaterialsInputSocketGuid, "MaterialTable", "Unknown"),
                new(RenderGraphDocument.CameraInputSocketGuid, "Camera", "Unknown")
            ],
            [new(RenderGraphDocument.ColorOutputSocketGuid, "Texture2D", "RGBA8")]);

    private static ManifestPass PostProcessManifestPass(
        Type passType,
        string inputKind = "Texture2D",
        string inputFormat = "RGBA8")
        => new(
            passType,
            ProtocolFixture.PostProcessPassGuid,
            [new(ProtocolFixture.PostProcessInputSocketGuid, inputKind, inputFormat)],
            [new(ProtocolFixture.PostProcessOutputSocketGuid, "Texture2D", "RGBA8")]);

    private static ManifestPass HistoryProbeManifestPass(Type passType)
        => new(
            passType,
            ProtocolFixture.HistoryProbePassGuid,
            [new(ProtocolFixture.HistoryProbeInputSocketGuid, "Texture2D", "RGBA8")],
            [
                new(ProtocolFixture.HistoryProbeObservedOutputSocketGuid, "Texture2D", "RGBA8"),
                new(ProtocolFixture.HistoryProbeNextOutputSocketGuid, "Texture2D", "RGBA8")
            ]);

    private static ManifestPass[] TypedBufferManifestPasses()
        =>
        [
            new(
                typeof(BufferProducerCpuPass),
                BufferProducerPassGuid,
                [],
                [
                    new(DirectionsOutputGuid, "Buffer", "Unknown", "float3"),
                    new(WeightsOutputGuid, "Buffer", "Unknown", "int")
                ]),
            new(
                typeof(BufferEvaluateCpuPass),
                BufferEvaluatePassGuid,
                [
                    new(DirectionsInputGuid, "Buffer", "Unknown", "float3"),
                    new(WeightsInputGuid, "Buffer", "Unknown", "int")
                ],
                [new(ScoresOutputGuid, "Buffer", "Unknown", "float")]),
            new(
                typeof(BufferVisualizeCpuPass),
                BufferVisualizePassGuid,
                [new(ScoresInputGuid, "Buffer", "Unknown", "float")],
                [new(BufferColorOutputGuid, "Texture2D", "RGBA8")])
        ];

    private static JsonNode SocketJson(ManifestSocket socket)
    {
        var result = new JsonObject
        {
            ["socketGuid"] = socket.SocketGuid,
            ["resourceKind"] = socket.ResourceKind,
            ["format"] = socket.Format
        };
        if (string.Equals(socket.ResourceKind, "Buffer", StringComparison.OrdinalIgnoreCase))
        {
            result["elementType"] = socket.ElementType;
        }
        return result;
    }

    private static void AssertEventuallyUnloaded(WeakReference reference)
    {
        for (var attempt = 0; attempt < 10 && reference.IsAlive; attempt++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }
        Assert.False(reference.IsAlive, "The previous project pass AssemblyLoadContext remained alive.");
    }

    [FeatherPass(RenderGraphDocument.MinimalRasterPassGuid)]
    public sealed class RedCpuPass : CpuPassBase
    {
        protected override Rgba8 Color => new(240, 20, 30, 255);
    }

    [FeatherPass(RenderGraphDocument.MinimalRasterPassGuid)]
    public sealed class BlueCpuPass : CpuPassBase
    {
        protected override Rgba8 Color => new(20, 30, 240, 255);
    }

    /// <summary>
    /// Paints the render purpose into the frame, which is how a pass would branch on it for real:
    /// cheap work for a frame the user is navigating through, expensive work for the one they keep.
    /// </summary>
    [FeatherPass(RenderGraphDocument.MinimalRasterPassGuid)]
    public sealed class PurposeProbeCpuPass : IRasterPass
    {
        [Input(RenderGraphDocument.GeometryInputSocketGuid)]
        public SceneGeometryHandle Geometry { get; init; }

        [Input(RenderGraphDocument.MaterialsInputSocketGuid)]
        public MaterialTableHandle Materials { get; init; }

        [Input(RenderGraphDocument.CameraInputSocketGuid)]
        public CameraHandle Camera { get; init; }

        [Output(RenderGraphDocument.ColorOutputSocketGuid, Format = TextureFormat.Rgba8)]
        public TextureHandle Output { get; init; }

        public void Execute(RenderContext context)
        {
            // Both the enum and the shorthand, because IsInteractive is the form nearly every pass
            // will actually write and a wrong inversion in it would otherwise go unnoticed.
            var purpose = (byte)(context.Purpose == RenderPurpose.Final ? 200 : 100);
            var shorthand = (byte)(context.IsInteractive ? 100 : 200);
            var pixels = new Rgba8[checked(context.Width * context.Height)];
            Array.Fill(pixels, new Rgba8(purpose, shorthand, 0, 255));
            context.SetColorOutput(Output, pixels);
        }
    }

    [FeatherPass(ProtocolFixture.PostProcessPassGuid)]
    public sealed class SwapRedBlueCpuPass : IComputePass
    {
        [Input(ProtocolFixture.PostProcessInputSocketGuid, Format = TextureFormat.Rgba8)]
        public TextureHandle Input { get; init; }

        [Output(ProtocolFixture.PostProcessOutputSocketGuid, Format = TextureFormat.Rgba8)]
        public TextureHandle Output { get; init; }

        public void Execute(RenderContext context)
        {
            var input = context.GetColorInput(Input).Span;
            var output = new Rgba8[input.Length];
            for (var index = 0; index < input.Length; index++)
            {
                var pixel = input[index];
                output[index] = new Rgba8(pixel.B, pixel.G, pixel.R, pixel.A);
            }
            context.SetColorOutput(Output, output);
        }
    }

    [FeatherPass(ProtocolFixture.HistoryProbePassGuid)]
    public sealed class HistoryProbeCpuPass : IComputePass
    {
        [Input(ProtocolFixture.HistoryProbeInputSocketGuid, Format = TextureFormat.Rgba8)]
        public TextureHandle History { get; init; }

        [Output(ProtocolFixture.HistoryProbeObservedOutputSocketGuid, Format = TextureFormat.Rgba8)]
        public TextureHandle Observed { get; init; }

        [Output(ProtocolFixture.HistoryProbeNextOutputSocketGuid, Format = TextureFormat.Rgba8)]
        public TextureHandle Next { get; init; }

        public void Execute(RenderContext context)
        {
            context.SetColorOutput(Observed, context.GetColorInput(History).Span);
            var next = new Rgba8[checked(context.Width * context.Height)];
            Array.Fill(next, new Rgba8(37, 73, 109, 255));
            context.SetColorOutput(Next, next);
        }
    }

    [FeatherPass(BufferProducerPassGuid)]
    public sealed class BufferProducerCpuPass : IComputePass
    {
        [Output(DirectionsOutputGuid)]
        public BufferHandle<float3> Directions { get; init; }

        [Output(WeightsOutputGuid)]
        public BufferHandle<int> Weights { get; init; }

        public void Execute(RenderContext context)
        {
            context.SetBufferOutput(
                Directions,
                new[] { new float3(1, 2, 2), new float3(3, 0, 4) });
            context.SetBufferOutput(Weights, new[] { 2, 3 });
        }
    }

    [FeatherPass(BufferEvaluatePassGuid)]
    public sealed class BufferEvaluateCpuPass : IComputePass
    {
        [Input(DirectionsInputGuid)]
        public BufferHandle<float3> Directions { get; init; }

        [Input(WeightsInputGuid)]
        public BufferHandle<int> Weights { get; init; }

        [Output(ScoresOutputGuid)]
        public BufferHandle<float> Scores { get; init; }

        public void Execute(RenderContext context)
        {
            var directions = context.GetBufferInput(Directions).Span;
            var weights = context.GetBufferInput(Weights).Span;
            if (directions.Length != weights.Length)
            {
                throw new InvalidDataException("Typed buffer test inputs have different lengths.");
            }

            var scores = new float[directions.Length];
            for (var index = 0; index < scores.Length; index++)
            {
                var direction = directions[index];
                scores[index] = MathF.Sqrt(
                    direction.X * direction.X +
                    direction.Y * direction.Y +
                    direction.Z * direction.Z) * weights[index];
            }
            context.SetBufferOutput(Scores, scores);
        }
    }

    [FeatherPass(BufferVisualizePassGuid)]
    public sealed class BufferVisualizeCpuPass : IComputePass
    {
        [Input(ScoresInputGuid)]
        public BufferHandle<float> Scores { get; init; }

        [Output(BufferColorOutputGuid, Format = TextureFormat.Rgba8)]
        public TextureHandle Output { get; init; }

        public void Execute(RenderContext context)
        {
            var scores = context.GetBufferInput(Scores).Span;
            var pixels = new Rgba8[checked(context.Width * context.Height)];
            var color = new Rgba8(
                checked((byte)MathF.Round(scores[0] * 10)),
                checked((byte)MathF.Round(scores[1] * 10)),
                checked((byte)scores.Length),
                255);
            Array.Fill(pixels, color);
            context.SetColorOutput(Output, pixels);
        }
    }

    public abstract class CpuPassBase : IRasterPass
    {
        [Input(RenderGraphDocument.GeometryInputSocketGuid)]
        public SceneGeometryHandle Geometry { get; init; }

        [Input(RenderGraphDocument.MaterialsInputSocketGuid)]
        public MaterialTableHandle Materials { get; init; }

        [Input(RenderGraphDocument.CameraInputSocketGuid)]
        public CameraHandle Camera { get; init; }

        [Output(RenderGraphDocument.ColorOutputSocketGuid, Format = TextureFormat.Rgba8)]
        public TextureHandle Output { get; init; }

        protected abstract Rgba8 Color { get; }

        public void Execute(RenderContext context)
        {
            _ = context.GetSceneGeometry(Geometry);
            _ = context.GetCamera(Camera);
            var pixels = new Rgba8[checked(context.Width * context.Height)];
            Array.Fill(pixels, Color);
            context.SetColorOutput(Output, pixels);
        }
    }

    private sealed record ManifestPass(
        Type PassType,
        string PassGuid,
        ManifestSocket[] Inputs,
        ManifestSocket[] Outputs);

    private sealed record ManifestSocket(
        string SocketGuid,
        string ResourceKind,
        string Format,
        string ElementType = "Unknown");
}
