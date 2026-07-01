using Feather.Native;

namespace Feather.Resources;

public readonly record struct SamplerDesc(
    SamplerFilter MinFilter,
    SamplerFilter MagFilter,
    SamplerMipmapMode MipmapMode,
    SamplerAddressMode AddressU,
    SamplerAddressMode AddressV,
    SamplerAddressMode AddressW,
    float MipLodBias,
    float MinLod,
    float MaxLod,
    bool AnisotropyEnabled,
    float MaxAnisotropy,
    bool CompareEnabled,
    SamplerCompareOp CompareOp,
    SamplerBorderColor BorderColor)
{
    public SamplerDesc(SamplerFilter minFilter, SamplerFilter magFilter, SamplerAddressMode addressU, SamplerAddressMode addressV)
        : this(
            minFilter,
            magFilter,
            minFilter == SamplerFilter.Linear ? SamplerMipmapMode.Linear : SamplerMipmapMode.Nearest,
            addressU,
            addressV,
            SamplerAddressMode.ClampToEdge,
            0.0f,
            0.0f,
            1000.0f,
            false,
            1.0f,
            false,
            SamplerCompareOp.Always,
            SamplerBorderColor.FloatOpaqueBlack)
    {
    }

    public static SamplerDesc NearestClamp => new(SamplerFilter.Nearest, SamplerFilter.Nearest, SamplerAddressMode.ClampToEdge, SamplerAddressMode.ClampToEdge);
    public static SamplerDesc LinearClamp => new(SamplerFilter.Linear, SamplerFilter.Linear, SamplerAddressMode.ClampToEdge, SamplerAddressMode.ClampToEdge);
    public static SamplerDesc NearestRepeat => new(SamplerFilter.Nearest, SamplerFilter.Nearest, SamplerAddressMode.Repeat, SamplerAddressMode.Repeat);
    public static SamplerDesc LinearRepeat => new(SamplerFilter.Linear, SamplerFilter.Linear, SamplerAddressMode.Repeat, SamplerAddressMode.Repeat);
    public static SamplerDesc LinearMirroredRepeat => new(SamplerFilter.Linear, SamplerFilter.Linear, SamplerAddressMode.MirroredRepeat, SamplerAddressMode.MirroredRepeat);
}

public enum SamplerFilter : uint
{
    Nearest,
    Linear
}

public enum SamplerMipmapMode : uint
{
    Nearest,
    Linear
}

public enum SamplerAddressMode : uint
{
    ClampToEdge,
    Repeat,
    MirroredRepeat,
    ClampToBorder
}

public enum SamplerCompareOp : uint
{
    Never,
    Less,
    Equal,
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always
}

public enum SamplerBorderColor : uint
{
    FloatTransparentBlack,
    IntTransparentBlack,
    FloatOpaqueBlack,
    IntOpaqueBlack,
    FloatOpaqueWhite,
    IntOpaqueWhite
}

public readonly struct SamplerState : IDisposable, IGpuSamplerBinding
{
    internal SamplerState(FeSamplerHandle handle, SamplerDesc desc)
    {
        Handle = handle;
        Desc = desc;
    }

    internal FeSamplerHandle Handle { get; }
    public SamplerDesc Desc { get; }
    FeSamplerHandle IGpuSamplerBinding.NativeSamplerHandle => Handle;

    internal static SamplerState Create(GpuContext context, SamplerDesc desc)
    {
        var nativeDesc = new FeSamplerDesc(
            (uint)desc.MinFilter,
            (uint)desc.MagFilter,
            (uint)desc.MipmapMode,
            (uint)desc.AddressU,
            (uint)desc.AddressV,
            (uint)desc.AddressW,
            desc.MipLodBias,
            desc.MinLod,
            desc.MaxLod,
            desc.AnisotropyEnabled ? 1u : 0u,
            desc.MaxAnisotropy,
            desc.CompareEnabled ? 1u : 0u,
            (uint)desc.CompareOp,
            (uint)desc.BorderColor);
        NativeMethods.ThrowIfFailed(NativeMethods.fe_sampler_create(context.Handle, in nativeDesc, out var handle));
        return new SamplerState(handle, desc);
    }

    public void Dispose()
    {
        Handle.Dispose();
    }
}
