using System.Buffers.Binary;
using System.Text.Json.Nodes;
using Feather.Math;

namespace Feather.Blender.RenderHost.Tests;

public sealed class RenderHostProtocolTests
{
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

        Assert.Contains("supports only", exception.Message);
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

    [Fact]
    public void FrameWriterMatchesBlenderFrameV1AndLeavesNoTemporaryFile()
    {
        using var fixture = new ProtocolFixture();
        var pixels = new[]
        {
            new Rgba32(1, 2, 3, 4),
            new Rgba32(5, 6, 7, 8)
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
}
