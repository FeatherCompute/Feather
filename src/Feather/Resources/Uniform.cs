namespace Feather.Resources;

/// <summary>
/// Represents a small CPU-side value that is uploaded as push-constant data when a generated shader is bound.
/// </summary>
/// <typeparam name="T">The unmanaged value type stored by the uniform.</typeparam>
public readonly struct Uniform<T>
    where T : unmanaged
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Uniform{T}"/> struct.
    /// </summary>
    /// <param name="value">The value to expose to generated GPU code.</param>
    public Uniform(T value)
    {
        Value = value;
    }

    /// <summary>
    /// Gets the current CPU-side value for this uniform.
    /// </summary>
    public T Value { get; }
}
