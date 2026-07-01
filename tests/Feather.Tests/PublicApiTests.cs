using Feather.Diagnostics;
using Feather.Graphics;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

namespace Feather.Tests;

public class PublicApiTests
{
    [Fact]
    public void ThreadGroupDefaultsMatchSpecification()
    {
        Assert.Equal((256, 1, 1), AttributeValues(new ThreadGroupSizeAttribute(DefaultThreadGroupSizes.X)));
        Assert.Equal((16, 16, 1), AttributeValues(new ThreadGroupSizeAttribute(DefaultThreadGroupSizes.XY)));
        Assert.Equal((8, 8, 4), AttributeValues(new ThreadGroupSizeAttribute(DefaultThreadGroupSizes.XYZ)));
    }

    [Fact]
    public void DiagnosticCatalogIncludesRequiredIds()
    {
        var ids = DiagnosticDescriptors.All.Select(diagnostic => diagnostic.Id).ToHashSet();

        foreach (var id in Enumerable.Range(1, 25).Select(value => $"FE{value:0000}"))
        {
            Assert.Contains(id, ids);
        }
    }

    [Fact]
    public void ShaderMathSupportsCpuEquivalentSmokeOperations()
    {
        var a = new float3(1, 2, 3);
        var b = new float3(4, 5, 6);

        Assert.Equal(32, ShaderMath.Dot(a, b));
        Assert.Equal(new float3(-3, 6, -3), ShaderMath.Cross(a, b));
        Assert.Equal(2, ShaderMath.Clamp(4, -2, 2));
        Assert.Equal(2.5f, ShaderMath.Lerp(2, 4, 0.25f));
    }

    [Fact]
    public void KernelDescriptorCanRepresentComputeResources()
    {
        var descriptor = new KernelDescriptor(
            KernelDimension.One,
            new int3(256, 1, 1),
            [new ResourceDescriptor(0, ResourceKind.Buffer, ResourceAccess.ReadWrite, typeof(float), "values")],
            [],
            BoundsCheck: true,
            AutoDiff: false,
            DebugName: "Smoke");

        Assert.Equal(KernelDimension.One, descriptor.Dimension);
        Assert.Equal(ResourceKind.Buffer, descriptor.Resources[0].Kind);
    }

    [Fact]
    public void SamplerDescriptorsExposeExpectedDefaults()
    {
        Assert.Equal(new SamplerDesc(SamplerFilter.Linear, SamplerFilter.Linear, SamplerAddressMode.Repeat, SamplerAddressMode.Repeat), SamplerDesc.LinearRepeat);
        Assert.Equal(new SamplerDesc(SamplerFilter.Nearest, SamplerFilter.Nearest, SamplerAddressMode.ClampToEdge, SamplerAddressMode.ClampToEdge), SamplerDesc.NearestClamp);
        Assert.Equal(SamplerMipmapMode.Linear, SamplerDesc.LinearRepeat.MipmapMode);
        Assert.Equal(SamplerMipmapMode.Nearest, SamplerDesc.NearestClamp.MipmapMode);
        Assert.False(SamplerDesc.LinearRepeat.AnisotropyEnabled);
        Assert.False(SamplerDesc.LinearRepeat.CompareEnabled);
        Assert.Equal(SamplerCompareOp.Always, SamplerDesc.LinearRepeat.CompareOp);
        Assert.Equal(SamplerBorderColor.FloatOpaqueBlack, SamplerDesc.LinearRepeat.BorderColor);
    }

    [Fact]
    public void GraphicsConvenienceResourcesExposeExpectedModes()
    {
        using var renderTarget = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(1, 1, PixelFormat.Rgba8);
        using var depth = GPU.CreateDepthTexture2D(1, 1);
        using var indices = GPU.CreateIndexBuffer<uint>([0, 1, 2]);

        Assert.Equal(TextureAccess.RenderTarget, renderTarget.Access);
        Assert.Equal(TextureAccess.DepthStencil, depth.Access);
        Assert.Equal(BufferAccess.ReadOnly, indices.Access);
    }

    [Fact]
    public void TextureCanCreateSampledShaderView()
    {
        using var texture = GPU.CreateTexture2D<Rgba32, Rgba32>(2, 3, PixelFormat.Rgba8, TextureAccess.Sampled);

        var view = texture.AsSampled();

        Assert.Equal(new int2(2, 3), view.Size);
    }

    [Fact]
    public void TextureExposesMipLevelMetadata()
    {
        using var texture = GPU.CreateTexture2D<Rgba32, Rgba32>(8, 4, 4, PixelFormat.Rgba8, TextureAccess.Sampled);

        Assert.Equal(4, texture.MipLevels);
        Assert.Equal(PixelFormat.Rgba8, texture.Format);
        Assert.Equal(TextureAccess.Sampled, texture.Access);
    }

    [Fact]
    public void Texture3DExposesSizeAndShaderViews()
    {
        using var texture = GPU.CreateTexture3D<Rgba32, Rgba32>(4, 3, 2, 2, PixelFormat.Rgba8);

        var view = texture.AsReadWrite();

        Assert.Equal(new int3(4, 3, 2), texture.Size);
        Assert.Equal(new int3(4, 3, 2), view.Size);
        Assert.Equal(2, texture.MipLevels);
    }

    [Fact]
    public void UniformStoresCurrentCpuValueForGeneratedPushConstants()
    {
        var uniform = new Uniform<float4>(new float4(1, 2, 3, 4));

        Assert.Equal(new float4(1, 2, 3, 4), uniform.Value);
    }

    [Fact]
    public void GpuBarriersAreShaderOnlyMarkers()
    {
        Assert.Throws<InvalidOperationException>(GpuBarrier.Workgroup);
        Assert.Throws<InvalidOperationException>(GpuBarrier.Memory);
        Assert.Throws<InvalidOperationException>(GpuBarrier.Full);
    }

    [Fact]
    public void GpuAtomicsAreShaderOnlyMarkers()
    {
        var value = 0;

        Assert.Throws<InvalidOperationException>(() => GpuAtomic.Add(ref value, 1));
        Assert.Throws<InvalidOperationException>(() => GpuAtomic.Sub(ref value, 1));
        Assert.Throws<InvalidOperationException>(() => GpuAtomic.Min(ref value, 1));
        Assert.Throws<InvalidOperationException>(() => GpuAtomic.Max(ref value, 1));
        Assert.Throws<InvalidOperationException>(() => GpuAtomic.And(ref value, 1));
        Assert.Throws<InvalidOperationException>(() => GpuAtomic.Or(ref value, 1));
        Assert.Throws<InvalidOperationException>(() => GpuAtomic.Xor(ref value, 1));
        Assert.Throws<InvalidOperationException>(() => GpuAtomic.Exchange(ref value, 1));
        Assert.Throws<InvalidOperationException>(() => GpuAtomic.CompareExchange(ref value, 0, 1));
    }

    [Fact]
    public void SharedMemoryTracksDeclaredLength()
    {
        var shared = new SharedMemory<float>(256);

        Assert.Equal(256, shared.Length);
        Assert.Throws<ArgumentOutOfRangeException>(() => new SharedMemory<int>(0));
    }

    [Fact]
    public void GpuProfilerExposesProcessWideControls()
    {
        try
        {
            GpuProfiler.SetEnabled(true);
            GpuProfiler.Clear();

            var missing = GpuProfiler.Query("MissingKernel");

            Assert.True(GpuProfiler.IsEnabled);
            Assert.Equal(0UL, missing.Count);
            Assert.Equal("MissingKernel", missing.Name);
            Assert.Equal(0.0, GpuProfiler.GetTotalTimeMs());
            Assert.Contains("No GPU commands recorded", GpuProfiler.GetFormattedReport(), StringComparison.Ordinal);
        }
        finally
        {
            GpuProfiler.Clear();
            GpuProfiler.SetEnabled(false);
        }
    }

    [Fact]
    public void GraphicsShaderIdsExposeSpecificationTypes()
    {
        Assert.Equal(typeof(int), typeof(VertexIds).GetProperty(nameof(VertexIds.Index))?.PropertyType);
        Assert.Equal(typeof(int), typeof(VertexIds).GetProperty(nameof(VertexIds.Instance))?.PropertyType);
        Assert.Equal(typeof(float4), typeof(FragmentIds).GetProperty(nameof(FragmentIds.Coord))?.PropertyType);
    }

    private static (int X, int Y, int Z) AttributeValues(ThreadGroupSizeAttribute attribute)
        => (attribute.X, attribute.Y, attribute.Z);

    private readonly record struct Rgba32(byte R, byte G, byte B, byte A);
}
