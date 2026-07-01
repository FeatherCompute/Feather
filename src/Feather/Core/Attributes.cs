namespace Feather;

[AttributeUsage(AttributeTargets.Struct)]
public sealed class KernelAttribute : Attribute
{
    public bool BoundsCheck { get; set; } = true;
}

[AttributeUsage(AttributeTargets.Struct)]
public sealed class AutoDiffAttribute : Attribute;

[AttributeUsage(AttributeTargets.Struct)]
public sealed class ThreadGroupSizeAttribute : Attribute
{
    public ThreadGroupSizeAttribute(int x) : this(x, 1, 1)
    {
    }

    public ThreadGroupSizeAttribute(int x, int y) : this(x, y, 1)
    {
    }

    public ThreadGroupSizeAttribute(int x, int y, int z)
    {
        X = x;
        Y = y;
        Z = z;
    }

    public ThreadGroupSizeAttribute(DefaultThreadGroupSizes size)
    {
        (X, Y, Z) = size switch
        {
            DefaultThreadGroupSizes.X => (256, 1, 1),
            DefaultThreadGroupSizes.XY => (16, 16, 1),
            DefaultThreadGroupSizes.XYZ => (8, 8, 4),
            _ => throw new ArgumentOutOfRangeException(nameof(size), size, null)
        };
    }

    public int X { get; }
    public int Y { get; }
    public int Z { get; }
}

[AttributeUsage(AttributeTargets.Struct)]
public sealed class VertexShaderAttribute : Attribute;

[AttributeUsage(AttributeTargets.Struct)]
public sealed class FragmentShaderAttribute : Attribute;

[AttributeUsage(AttributeTargets.Method)]
public sealed class EntryAttribute : Attribute;

[AttributeUsage(AttributeTargets.Method)]
public sealed class ShaderFunctionAttribute : Attribute;

/// <summary>
/// Marks a static or instance method inside a kernel or shader struct as a GPU callable function.
/// Callable methods are compiled into the shader module and can be invoked from the entry-point
/// method or from other callables within the same module.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item>Callables must be defined inside the kernel or shader struct.</item>
/// <item>The generator builds a call graph and emits each callable exactly once, even when referenced from multiple call sites.</item>
/// <item>Recursion is not supported by current EasyGPU backends and will produce a diagnostic.</item>
/// <item>Parameters may be scalars, vectors, matrices, structs, or resource references where the backend allows.</item>
/// <item><c>ref</c> / <c>inout</c> parameters are accepted only where EasyGPU supports writable arguments.</item>
/// </list>
/// </remarks>
[AttributeUsage(AttributeTargets.Method)]
public sealed class CallableAttribute : Attribute;

[AttributeUsage(AttributeTargets.Struct)]
public sealed class GpuStructAttribute(GpuLayout layout = GpuLayout.Std430) : Attribute
{
    public GpuLayout Layout { get; } = layout;
}

[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
public sealed class PositionAttribute : Attribute;

[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
public sealed class ColorAttribute(uint index) : Attribute
{
    public uint Index { get; } = index;
}

[AttributeUsage(AttributeTargets.Parameter | AttributeTargets.Field)]
public sealed class BindingAttribute(uint index) : Attribute
{
    public uint Index { get; } = index;
}
