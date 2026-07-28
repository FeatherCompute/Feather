using System.Buffers.Binary;

namespace Feather.Blender.RenderHost.Tests;

public sealed class RenderHostGpuTests
{
    [Theory]
    [Trait("Category", "Gpu")]
    [InlineData(1)]
    [InlineData(4)]
    public void RenderOnceRasterizesSceneMeshThroughPublicFeatherApi(int sampleCount)
    {
        using var fixture = new ProtocolFixture();
        fixture.WriteScene();
        fixture.WriteGraph(sampleCount: sampleCount);
        fixture.WriteRequest();
        using var host = new RenderHostRunner();

        var result = host.RenderOnce(fixture.RequestPath);

        Assert.Equal("TypedEasyGpu", result.DispatchPath);
        Assert.Equal(1, result.TriangleCount);
        Assert.Equal(3, result.VertexCount);
        var frame = File.ReadAllBytes(fixture.OutputPath);
        Assert.Equal(42ul, BinaryPrimitives.ReadUInt64LittleEndian(frame.AsSpan(32, 8)));
        var pixels = frame.AsSpan(40);
        var brightPixelCount = 0;
        var backgroundPixelCount = 0;
        for (var offset = 0; offset < pixels.Length; offset += 4)
        {
            if (pixels[offset] > 200 && pixels[offset + 1] > 200 && pixels[offset + 2] > 200)
            {
                brightPixelCount++;
            }
            if (pixels[offset] < 10 && pixels[offset + 1] < 10 && pixels[offset + 2] < 20)
            {
                backgroundPixelCount++;
            }
        }

        Assert.True(brightPixelCount > 300, $"Expected rendered triangle pixels, found {brightPixelCount}.");
        Assert.True(backgroundPixelCount > 300, $"Expected background pixels, found {backgroundPixelCount}.");
    }
}
