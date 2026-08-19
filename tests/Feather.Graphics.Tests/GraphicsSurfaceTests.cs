using Feather.Graphics;

namespace Feather.Graphics.Tests;

public class GraphicsSurfaceTests
{
    [Fact]
    public void GraphicsPipelineDescDefaultsToTriangleListWithoutDepth()
    {
        var desc = new GraphicsPipelineDesc();

        Assert.Equal(PrimitiveTopology.TriangleList, desc.Topology);
        Assert.Equal(SampleCount.X1, desc.SampleCount);
        Assert.False(desc.DepthTest);
        Assert.False(desc.DepthWrite);
    }

    [Fact]
    public void GraphicsEnumsExposeEasyGpuTopologyAndSampleCoverage()
    {
        Assert.Equal(5u, (uint)PrimitiveTopology.TriangleFan);
        Assert.Equal(16u, (uint)SampleCount.X16);
    }

    [Fact]
    public void BackendCapsExposeAdvancedRasterFeatureFlags()
    {
        var caps = new BackendCaps(
            BackendType.Vulkan,
            1,
            1,
            1,
            SupportsGraphics: true,
            SupportsAD: false,
            SupportsNN: false,
            SupportsDepthClamp: true,
            SupportsNonFillPolygonMode: true);

        Assert.True(caps.SupportsDepthClamp);
        Assert.True(caps.SupportsNonFillPolygonMode);
    }

    [Fact]
    public void BackendDeviceInfoCarriesStablePipelineIdentity()
    {
        var info = new BackendDeviceInfo(
            NativeAbiVersion: 1,
            MaxTextureDimension2D: 8192,
            SupportsTimestampQueries: true,
            AdapterName: "test-adapter",
            DriverVersion: "test-driver",
            BackendVersion: "1.3");

        Assert.Equal("test-adapter", info.AdapterName);
        Assert.Equal("test-driver", info.DriverVersion);
        Assert.Equal(8192u, info.MaxTextureDimension2D);
        Assert.True(info.SupportsTimestampQueries);
    }
}
