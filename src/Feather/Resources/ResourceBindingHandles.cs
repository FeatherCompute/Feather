using Feather.Native;

namespace Feather.Resources;

/// <summary>
/// Exposes the native buffer handle needed by generated Feather binding code.
/// </summary>
public interface IGpuBufferBinding
{
    /// <summary>
    /// Gets the native buffer handle associated with the shader-facing buffer view.
    /// </summary>
    FeBufferHandle NativeBufferHandle { get; }
}

/// <summary>
/// Exposes the native texture handle needed by generated Feather binding code.
/// </summary>
public interface IGpuTextureBinding
{
    /// <summary>
    /// Gets the native texture handle associated with the shader-facing texture view.
    /// </summary>
    FeTextureHandle NativeTextureHandle { get; }
}

/// <summary>
/// Exposes the native sampler handle needed by generated Feather binding code.
/// </summary>
public interface IGpuSamplerBinding
{
    /// <summary>
    /// Gets the native sampler handle associated with the sampler state.
    /// </summary>
    FeSamplerHandle NativeSamplerHandle { get; }
}
