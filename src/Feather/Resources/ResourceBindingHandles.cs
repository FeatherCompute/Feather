namespace Feather.Resources;

/// <summary>
/// Marks a shader-facing GPU buffer binding.
/// </summary>
public interface IGpuBufferBinding
{
}

/// <summary>
/// Marks a shader-facing GPU texture binding.
/// </summary>
public interface IGpuTextureBinding
{
}

/// <summary>
/// Marks a shader-facing GPU sampler binding.
/// </summary>
public interface IGpuSamplerBinding
{
}

internal interface INativeBufferBinding
{
    Native.FeBufferHandle NativeBufferHandle { get; }
}

internal interface INativeTextureBinding
{
    Native.FeTextureHandle NativeTextureHandle { get; }
}

internal interface INativeSamplerBinding
{
    Native.FeSamplerHandle NativeSamplerHandle { get; }
}
