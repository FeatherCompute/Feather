namespace Feather.RenderGraph;

/// <summary>
/// Marks a class as a Feather render-graph pass with a persistent project identity.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class FeatherPassAttribute(string guid) : Attribute
{
    public string Guid { get; } = guid;

    public string? Name { get; set; }

    public string Category { get; set; } = "Uncategorized";

    public int Version { get; set; } = 1;
}

/// <summary>
/// Declares a pass resource input and its persistent socket identity.
/// </summary>
[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false)]
public sealed class InputAttribute(string guid) : Attribute
{
    public string Guid { get; } = guid;

    public string? Name { get; set; }

    public TextureFormat Format { get; set; } = TextureFormat.Unknown;
}

/// <summary>
/// Declares a pass resource output and its persistent socket identity.
/// </summary>
[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false)]
public sealed class OutputAttribute(string guid) : Attribute
{
    public string Guid { get; } = guid;

    public string? Name { get; set; }

    public TextureFormat Format { get; set; } = TextureFormat.Unknown;
}

/// <summary>
/// Declares an editable pass parameter and its persistent identity.
/// </summary>
[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false)]
public sealed class ParameterAttribute(string guid) : Attribute
{
    public string Guid { get; } = guid;

    public string? Name { get; set; }

    public object? DefaultValue { get; set; }

    public double Min { get; set; } = double.NaN;

    public double Max { get; set; } = double.NaN;
}

/// <summary>
/// Base contract for passes executed by a Feather render host.
/// </summary>
public interface IRenderPass
{
    void Execute(RenderContext context);
}

/// <summary>
/// Identifies a pass that records raster graphics work.
/// </summary>
public interface IRasterPass : IRenderPass;

/// <summary>
/// Identifies a pass that records compute work.
/// </summary>
public interface IComputePass : IRenderPass;

public enum TextureFormat
{
    Unknown = 0,
    R8 = 1,
    Rg8 = 2,
    Rgba8 = 3,
    Bgra8 = 4,
    R16Float = 5,
    Rg16Float = 6,
    Rgba16Float = 7,
    R32Float = 8,
    Rg32Float = 9,
    Rgba32Float = 10,
    Depth24Stencil8 = 100,
    Depth32Float = 101
}

public readonly record struct ResourceHandle(ulong Value);

public readonly record struct BufferHandle(ulong Value)
{
    /// <summary>
    /// Treats this untyped handle as a buffer containing <typeparamref name="T"/> elements.
    /// </summary>
    public BufferHandle<T> As<T>()
        where T : unmanaged
        => new(Value);
}

/// <summary>
/// Identifies a render-graph buffer whose logical elements have type <typeparamref name="T"/>.
/// </summary>
/// <typeparam name="T">The unmanaged buffer element type.</typeparam>
public readonly record struct BufferHandle<T>(ulong Value)
    where T : unmanaged
{
    /// <summary>
    /// Gets the same graph resource without its compile-time element type.
    /// </summary>
    public BufferHandle Untyped => new(Value);

    public static implicit operator BufferHandle(BufferHandle<T> handle) => handle.Untyped;

    public static explicit operator BufferHandle<T>(BufferHandle handle) => new(handle.Value);
}

public readonly record struct TextureHandle(ulong Value);

public readonly record struct SceneGeometryHandle(ulong Value);

public readonly record struct MaterialTableHandle(ulong Value);

public readonly record struct TextureTableHandle(ulong Value);

public readonly record struct CameraHandle(ulong Value);

public readonly record struct LightTableHandle(ulong Value);

public readonly record struct TimeHandle(ulong Value);
