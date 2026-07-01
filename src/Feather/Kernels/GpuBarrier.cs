namespace Feather;

/// <summary>
/// Provides shader-only synchronization barriers for generated compute kernels.
/// </summary>
public static class GpuBarrier
{
    /// <summary>
    /// Synchronizes invocations in the current workgroup.
    /// </summary>
    public static void Workgroup()
        => ThrowShaderOnly();

    /// <summary>
    /// Synchronizes memory visibility for shader memory operations.
    /// </summary>
    public static void Memory()
        => ThrowShaderOnly();

    /// <summary>
    /// Synchronizes memory visibility and invocations in the current workgroup.
    /// </summary>
    public static void Full()
        => ThrowShaderOnly();

    private static void ThrowShaderOnly()
        => throw new InvalidOperationException("GPU barriers are shader markers and can only be used inside Feather-generated GPU code.");
}
