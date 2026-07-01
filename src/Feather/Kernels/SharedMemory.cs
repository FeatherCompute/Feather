namespace Feather;

/// <summary>
/// Represents shader-only workgroup shared memory.
/// </summary>
/// <typeparam name="T">The unmanaged element type stored in shared memory.</typeparam>
public readonly struct SharedMemory<T>
    where T : unmanaged
{
    /// <summary>
    /// Initializes a shared-memory declaration marker.
    /// </summary>
    /// <param name="length">The number of elements in the shared-memory array.</param>
    public SharedMemory(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        Length = length;
    }

    /// <summary>
    /// Gets the declared element count.
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// Gets the shader value at an element index by reference.
    /// </summary>
    public ref T this[int index] => ref ShaderRuntimeMarker<T>.RefValue;
}
