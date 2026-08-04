using Feather.Interop;
using Feather.Math;
using Feather.Resources;

namespace Feather.Integration.Tests;

public class LuisaBackendResourceDispatchTests
{
    [Fact]
    [Trait("Category", "Gpu")]
    public void FullIntegerAtomicMatrixMatchesEasyGpu()
    {
        using var input = GPU.CreateBuffer<int>([7]);
        int[] initial = [10, 10, 10, 1, 0xFFFF, 0x10, 0xF0, -1, 0];
        using var easy = GPU.CreateBuffer<int>(initial);
        using var luisa = GPU.CreateBuffer<int>(initial);
        GPU.Dispatch(new AtomicOpsKernel(input.AsReadOnly(), easy.AsReadWrite()), 1);
        DispatchLuisa(new AtomicOpsKernel(input.AsReadOnly(), luisa.AsReadWrite()), new(1, 1, 1));
        Assert.Equal(easy.ToArray(), luisa.ToArray());
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void TwoAndThreeDimensionalDispatchIdsMatchEasyGpu()
    {
        using var easy2D = GPU.CreateBuffer<int>(6);
        using var luisa2D = GPU.CreateBuffer<int>(6);
        GPU.Dispatch(new ThreadIdsXyLinearIndexKernel(easy2D.AsReadWrite()), new int2(3, 2));
        DispatchLuisa(new ThreadIdsXyLinearIndexKernel(luisa2D.AsReadWrite()), new(3, 2, 1));
        Assert.Equal(easy2D.ToArray(), luisa2D.ToArray());

        using var easy3D = GPU.CreateBuffer<int>(8);
        using var luisa3D = GPU.CreateBuffer<int>(8);
        GPU.Dispatch(new ThreadIdsXyzLinearIndexKernel(easy3D.AsReadWrite()), new int3(2, 2, 2));
        DispatchLuisa(new ThreadIdsXyzLinearIndexKernel(luisa3D.AsReadWrite()), new(2, 2, 2));
        Assert.Equal(easy3D.ToArray(), luisa3D.ToArray());
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void Texture2DAndTexture3DLoadStoreMatchEasyGpu()
    {
        float4[] pixels =
        [
            new(1, 2, 3, 4), new(5, 6, 7, 8),
            new(9, 10, 11, 12), new(13, 14, 15, 16)
        ];
        using var input2D = GPU.CreateTexture2D<float4, float4>(2, 2, PixelFormat.Rgba32Float, TextureAccess.ReadOnly);
        using var easy2D = GPU.CreateTexture2D<float4, float4>(2, 2, PixelFormat.Rgba32Float, TextureAccess.ReadWrite);
        using var luisa2D = GPU.CreateTexture2D<float4, float4>(2, 2, PixelFormat.Rgba32Float, TextureAccess.ReadWrite);
        input2D.Upload(pixels);
        GPU.Dispatch(new TextureFloat4CopyKernel(input2D.AsReadOnly(), easy2D.AsReadWrite()), new int2(2, 2));
        DispatchLuisa(new TextureFloat4CopyKernel(input2D.AsReadOnly(), luisa2D.AsReadWrite()), new(2, 2, 1));
        var easyPixels = new float4[4];
        var luisaPixels = new float4[4];
        easy2D.Read(easyPixels);
        luisa2D.Read(luisaPixels);
        Assert.Equal(easyPixels, luisaPixels);

        float4[] voxels =
        [
            new(0, 0, 0, 1), new(1, 0, 1, 1), new(0, 1, 10, 1), new(1, 1, 11, 1),
            new(0, 0, 100, 1), new(1, 0, 101, 1), new(0, 1, 110, 1), new(1, 1, 111, 1)
        ];
        using var input3D = GPU.CreateTexture3D<float4, float4>(2, 2, 2, PixelFormat.Rgba32Float, TextureAccess.ReadOnly);
        using var easy3D = GPU.CreateTexture3D<float4, float4>(2, 2, 2, PixelFormat.Rgba32Float, TextureAccess.ReadWrite);
        using var luisa3D = GPU.CreateTexture3D<float4, float4>(2, 2, 2, PixelFormat.Rgba32Float, TextureAccess.ReadWrite);
        input3D.Upload(voxels);
        GPU.Dispatch(new Texture3DCopyKernel(input3D.AsReadOnly(), easy3D.AsReadWrite()), new int3(2, 2, 2));
        DispatchLuisa(new Texture3DCopyKernel(input3D.AsReadOnly(), luisa3D.AsReadWrite()), new(2, 2, 2));
        var easyVoxels = new float4[8];
        var luisaVoxels = new float4[8];
        easy3D.Read(easyVoxels);
        luisa3D.Read(luisaVoxels);
        Assert.Equal(easyVoxels, luisaVoxels);
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void TextureSamplingAndMixedResourceOrderMatchEasyGpu()
    {
        using var texture = GPU.CreateTexture2D<float4, float4>(2, 2, PixelFormat.Rgba32Float, TextureAccess.Sampled);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var easy = GPU.CreateBuffer<float4>(2);
        using var luisa = GPU.CreateBuffer<float4>(2);
        texture.Upload([
            new(1, 2, 3, 4), new(5, 6, 7, 8),
            new(9, 10, 11, 12), new(13, 14, 15, 16)
        ]);

        GPU.Dispatch(new LuisaTextureSamplingKernel(texture.AsSampled(), sampler, easy.AsReadWrite()), 1);
        DispatchLuisa(new LuisaTextureSamplingKernel(texture.AsSampled(), sampler, luisa.AsReadWrite()), new(1, 1, 1));
        Assert.Equal(easy.ToArray(), luisa.ToArray());
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void TextureSampleGradExecutesThroughLuisaXir()
    {
        using var texture = GPU.CreateTexture2D<float4, float4>(2, 2, PixelFormat.Rgba32Float, TextureAccess.Sampled);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var output = GPU.CreateBuffer<float4>(1);
        texture.Upload([
            new(1, 2, 3, 4), new(5, 6, 7, 8),
            new(9, 10, 11, 12), new(13, 14, 15, 16)
        ]);

        DispatchLuisa(new LuisaTextureSampleGradKernel(texture.AsSampled(), sampler, output.AsReadWrite()), new(1, 1, 1));
        Assert.Equal(new float4(13, 14, 15, 16), output.ToArray()[0]);
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void NestedScalarCallablesMatchEasyGpu()
    {
        using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var easy = GPU.CreateBuffer<float>(4);
        using var luisa = GPU.CreateBuffer<float>(4);
        GPU.Dispatch(new NestedCallableKernel(input.AsReadOnly(), easy.AsReadWrite()), 4);
        DispatchLuisa(new NestedCallableKernel(input.AsReadOnly(), luisa.AsReadWrite()), new(4, 1, 1));
        Assert.Equal(easy.ToArray(), luisa.ToArray());
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void ShaderLibraryBufferCallablesMatchEasyGpu()
    {
        using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var easy = GPU.CreateBuffer<float>(4);
        using var luisa = GPU.CreateBuffer<float>(4);
        GPU.Dispatch(new ReadWriteBufferCallableKernel(input.AsReadOnly(), easy.AsReadWrite()), 4);
        DispatchLuisa(new ReadWriteBufferCallableKernel(input.AsReadOnly(), luisa.AsReadWrite()), new(4, 1, 1));
        Assert.Equal(easy.ToArray(), luisa.ToArray());
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void ShaderLibraryTextureAndSamplerCallablesMatchEasyGpu()
    {
        using var sampled = GPU.CreateTexture2D<Rgba32, Rgba32>(1, 1, PixelFormat.Rgba8, TextureAccess.Sampled);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var easySample = GPU.CreateBuffer<float>(1);
        using var luisaSample = GPU.CreateBuffer<float>(1);
        sampled.Upload([new Rgba32(128, 0, 0, 255)]);
        GPU.Dispatch(new TextureCallableKernel(sampled.AsSampled(), sampler, easySample.AsReadWrite()), 1);
        DispatchLuisa(new TextureCallableKernel(sampled.AsSampled(), sampler, luisaSample.AsReadWrite()), new(1, 1, 1));
        Assert.Equal(easySample.ToArray(), luisaSample.ToArray());

        using var storage = GPU.CreateTexture2D<Rgba32, Rgba32>(1, 1, PixelFormat.Rgba8, TextureAccess.ReadOnly);
        using var easyLoad = GPU.CreateBuffer<float>(1);
        using var luisaLoad = GPU.CreateBuffer<float>(1);
        storage.Upload([new Rgba32(64, 0, 0, 255)]);
        GPU.Dispatch(new ReadOnlyTextureCallableKernel(storage.AsReadOnly(), easyLoad.AsReadWrite()), 1);
        DispatchLuisa(new ReadOnlyTextureCallableKernel(storage.AsReadOnly(), luisaLoad.AsReadWrite()), new(1, 1, 1));
        Assert.Equal(easyLoad.ToArray(), luisaLoad.ToArray());
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void MutableGpuStructCallablesAndWritebackMatchEasyGpu()
    {
        MutableCounter[] counters =
        [
            new() { Value = 10, Hits = 1, Nested = new MutableCounterNested { Inner = 5 } },
            new() { Value = -3, Hits = 4, Nested = new MutableCounterNested { Inner = 1 } }
        ];
        using var easyCounters = GPU.CreateBuffer<MutableCounter>(counters);
        using var luisaCounters = GPU.CreateBuffer<MutableCounter>(counters);
        using var input = GPU.CreateBuffer<float>([2, 5]);
        using var easy = GPU.CreateBuffer<float>(2);
        using var luisa = GPU.CreateBuffer<float>(2);
        GPU.Dispatch(new GpuStructMutatingInstanceCallableKernel(
            easyCounters.AsReadWrite(), input.AsReadOnly(), easy.AsReadWrite()), 2);
        DispatchLuisa(new GpuStructMutatingInstanceCallableKernel(
            luisaCounters.AsReadWrite(), input.AsReadOnly(), luisa.AsReadWrite()), new(2, 1, 1));
        Assert.Equal(easy.ToArray(), luisa.ToArray());
        var expected = easyCounters.ToArray();
        var actual = luisaCounters.ToArray();
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].Value, actual[i].Value);
            Assert.Equal(expected[i].Hits, actual[i].Hits);
            Assert.Equal(expected[i].Nested.Inner, actual[i].Nested.Inner);
        }
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void StructArraysAndNestedWritebackMatchEasyGpu()
    {
        var first = new ArrayScene { Weight = 5 };
        first.Directions[0] = new(1, 2, 3);
        first.Directions[1] = new(4, 5, 6);
        first.Directions[2] = new(7, 8, 9);
        first.Directions[3] = new(10, 11, 12);
        var second = new ArrayScene { Weight = 20 };
        second.Directions[0] = new(2, 4, 6);
        second.Directions[1] = new(8, 10, 12);
        second.Directions[2] = new(14, 16, 18);
        second.Directions[3] = new(20, 22, 24);
        using var input = GPU.CreateBuffer<ArrayScene>([first, second]);
        using var easyValues = GPU.CreateBuffer<float>(2);
        using var luisaValues = GPU.CreateBuffer<float>(2);
        using var easy = GPU.CreateBuffer<NestedArrayScene>(2);
        using var luisa = GPU.CreateBuffer<NestedArrayScene>(2);
        GPU.Dispatch(new StructArrayReadKernel(input.AsReadOnly(), easyValues.AsReadWrite()), 2);
        DispatchLuisa(new StructArrayReadKernel(input.AsReadOnly(), luisaValues.AsReadWrite()), new(2, 1, 1));
        GPU.Dispatch(new StructArrayWriteKernel(easyValues.AsReadOnly(), easy.AsReadWrite()), 2);
        DispatchLuisa(new StructArrayWriteKernel(luisaValues.AsReadOnly(), luisa.AsReadWrite()), new(2, 1, 1));
        Assert.Equal(easyValues.ToArray(), luisaValues.ToArray());
        var expected = easy.ToArray();
        var actual = luisa.ToArray();
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].Weight, actual[i].Weight);
            for (var item = 0; item < 3; item++)
            {
                Assert.Equal(expected[i].Items[item].Value, actual[i].Items[item].Value);
            }
        }
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void NonDivisibleLogicalBoundsMatchEasyGpu()
    {
        using var easy = GPU.CreateBuffer<int>(5);
        using var luisa = GPU.CreateBuffer<int>(5);
        GPU.Dispatch(new BoundsCheckedWriteKernel(easy.AsReadWrite()), 5);
        DispatchLuisa(new BoundsCheckedWriteKernel(luisa.AsReadWrite()), new(5, 1, 1));
        Assert.Equal(easy.ToArray(), luisa.ToArray());
    }

    private static void DispatchLuisa<TKernel>(TKernel kernel, GpuDispatchSize size)
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        using var compiled = GpuKernel.Create<TKernel>(GPU.Context, GpuExecutionBackend.Luisa);
        GpuKernel.Dispatch(GPU.Context, compiled, kernel, size, wait: true);
        Assert.Equal(DispatchPath.Luisa, compiled.LastDispatchPath);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct LuisaTextureSamplingKernel(
    SampledTexture2D<float4> texture,
    SamplerState sampler,
    ReadWriteBuffer<float4> output) : IKernel1D
{
    public void Execute()
    {
        float2 uv = new(0.75f, 0.75f);
        output[0] = texture.Sample(sampler, uv);
        output[1] = texture.SampleLevel(sampler, uv, 0.0f);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct LuisaTextureSampleGradKernel(
    SampledTexture2D<float4> texture,
    SamplerState sampler,
    ReadWriteBuffer<float4> output) : IKernel1D
{
    public void Execute()
    {
        float2 uv = new(0.75f, 0.75f);
        output[0] = texture.SampleGrad(sampler, uv, new float2(0.5f, 0.0f), new float2(0.0f, 0.5f));
    }
}
