using Feather.Resources;

namespace Feather.NN;

/// <summary>
/// Describes a non-owning view into a tensor buffer.
/// </summary>
/// <typeparam name="T">The unmanaged element type stored by the tensor.</typeparam>
public readonly record struct TensorView<T>
    where T : unmanaged
{
    /// <summary>
    /// Initializes a tensor view.
    /// </summary>
    public TensorView(Tensor<T> tensor, TensorShape shape, int offset = 0)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        if (offset + shape.ElementCount > tensor.ElementCount)
        {
            throw new ArgumentException("Tensor view range exceeds the source tensor.", nameof(shape));
        }

        Tensor = tensor;
        Shape = shape;
        Offset = offset;
    }

    /// <summary>
    /// Gets the tensor that owns the backing buffer.
    /// </summary>
    public Tensor<T> Tensor { get; }

    /// <summary>
    /// Gets the logical view shape.
    /// </summary>
    public TensorShape Shape { get; }

    /// <summary>
    /// Gets the flat element offset inside <see cref="Tensor"/>.
    /// </summary>
    public int Offset { get; }

    /// <summary>
    /// Gets the number of elements visible through the view.
    /// </summary>
    public int ElementCount => Shape.ElementCount;
}

/// <summary>
/// A typed rank-2 tensor wrapper for matrix-shaped NN values.
/// </summary>
/// <typeparam name="T">The unmanaged element type stored by the tensor.</typeparam>
public sealed class Tensor2D<T> : IDisposable
    where T : unmanaged
{
    /// <summary>
    /// Initializes a new rank-2 tensor over an existing GPU buffer.
    /// </summary>
    public Tensor2D(int rows, int columns, GpuBuffer<T> buffer, bool requiresGrad = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);
        Tensor = new Tensor<T>(new TensorShape(rows, columns), buffer, requiresGrad);
        Rows = rows;
        Columns = columns;
    }

    /// <summary>
    /// Gets the number of matrix rows.
    /// </summary>
    public int Rows { get; }

    /// <summary>
    /// Gets the number of matrix columns.
    /// </summary>
    public int Columns { get; }

    /// <summary>
    /// Gets the underlying tensor.
    /// </summary>
    public Tensor<T> Tensor { get; }

    /// <summary>
    /// Gets the matrix tensor shape.
    /// </summary>
    public TensorShape Shape => Tensor.Shape;

    /// <summary>
    /// Gets the backing GPU buffer.
    /// </summary>
    public GpuBuffer<T> Buffer => Tensor.Buffer;

    /// <summary>
    /// Creates a non-owning view over this matrix tensor.
    /// </summary>
    public TensorView<T> AsView() => new(Tensor, Shape);

    /// <inheritdoc />
    public void Dispose() => Tensor.Dispose();
}
