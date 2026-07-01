namespace Feather;

/// <summary>
/// Provides shader-only atomic operation markers for generated compute kernels.
/// </summary>
public static class GpuAtomic
{
    /// <summary>
    /// Atomically adds <paramref name="value" /> to <paramref name="target" /> and returns the original value.
    /// </summary>
    public static int Add(ref int target, int value)
        => ThrowShaderOnly<int>();

    /// <summary>
    /// Atomically subtracts <paramref name="value" /> from <paramref name="target" /> and returns the original value.
    /// </summary>
    public static int Sub(ref int target, int value)
        => ThrowShaderOnly<int>();

    /// <summary>
    /// Atomically writes the minimum of <paramref name="target" /> and <paramref name="value" /> and returns the original value.
    /// </summary>
    public static int Min(ref int target, int value)
        => ThrowShaderOnly<int>();

    /// <summary>
    /// Atomically writes the maximum of <paramref name="target" /> and <paramref name="value" /> and returns the original value.
    /// </summary>
    public static int Max(ref int target, int value)
        => ThrowShaderOnly<int>();

    /// <summary>
    /// Atomically applies bitwise AND with <paramref name="value" /> and returns the original value.
    /// </summary>
    public static int And(ref int target, int value)
        => ThrowShaderOnly<int>();

    /// <summary>
    /// Atomically applies bitwise OR with <paramref name="value" /> and returns the original value.
    /// </summary>
    public static int Or(ref int target, int value)
        => ThrowShaderOnly<int>();

    /// <summary>
    /// Atomically applies bitwise XOR with <paramref name="value" /> and returns the original value.
    /// </summary>
    public static int Xor(ref int target, int value)
        => ThrowShaderOnly<int>();

    /// <summary>
    /// Atomically exchanges <paramref name="target" /> with <paramref name="value" /> and returns the original value.
    /// </summary>
    public static int Exchange(ref int target, int value)
        => ThrowShaderOnly<int>();

    /// <summary>
    /// Atomically compares <paramref name="target" /> with <paramref name="compare" />, writes <paramref name="value" /> on match, and returns the original value.
    /// </summary>
    public static int CompareExchange(ref int target, int compare, int value)
        => ThrowShaderOnly<int>();

    private static T ThrowShaderOnly<T>()
        => throw new InvalidOperationException("GPU atomics are shader markers and can only be used inside Feather-generated GPU code.");
}
