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

    public double Step { get; set; } = double.NaN;

    public string? Unit { get; set; }

    public string? Description { get; set; }

    public string? Group { get; set; }

    public int Order { get; set; }

    public string? EditorHint { get; set; }

    public ParameterMutability Mutability { get; set; } = ParameterMutability.Dynamic;

    public ParameterBindingTargets Bindings { get; set; } = ParameterBindingTargets.Instance;

    public ParameterRedaction Redaction { get; set; } = ParameterRedaction.Public;
}

/// <summary>
/// Gives a Studio-visible enum a persistent nominal identity. Enum symbols and source files may
/// be renamed without changing this value.
/// </summary>
[AttributeUsage(AttributeTargets.Enum, AllowMultiple = false, Inherited = false)]
public sealed class FeatherEnumAttribute(string guid) : Attribute
{
    public string Guid { get; } = guid;

    /// <summary>Allows an unnamed numeric value to be retained by a normal enum.</summary>
    public bool AllowUnknownNumeric { get; set; }

    /// <summary>Allows a flags value to retain bits outside the declared member mask.</summary>
    public bool AllowUnknownBits { get; set; }
}

/// <summary>
/// Gives one Studio-visible enum member a persistent identity independent of its symbol,
/// declaration order, display label, and numeric representation.
/// </summary>
[AttributeUsage(AttributeTargets.Field, AllowMultiple = false, Inherited = false)]
public sealed class FeatherEnumMemberAttribute(string guid) : Attribute
{
    public string Guid { get; } = guid;

    public string? Name { get; set; }

    public string? Description { get; set; }

    public int Order { get; set; }

    public bool Deprecated { get; set; }

    public string? ReplacementMemberGuid { get; set; }
}

public enum ParameterMutability
{
    Dynamic = 0,
    Specialization = 1,
    ResourceShape = 2,
    CompileTime = 3,
}

[Flags]
public enum ParameterBindingTargets
{
    None = 0,
    Instance = 1 << 0,
    GraphValue = 1 << 1,
    RuntimeProperty = 1 << 2,
    Timeline = 1 << 3,
    Public = 1 << 4,
}

public enum ParameterRedaction
{
    Public = 0,
    MetadataOnly = 1,
    Secret = 2,
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

/// <summary>
/// Identifies one scene object a render graph selected by name, as distinct from the whole scene.
/// </summary>
/// <remarks>
/// A host normally hands a pass the entire scene, which is what a rasterizer wants and what an
/// effect aimed at a single object cannot use: an ocean surface, a cloth patch or a fluid domain each
/// need one object's mesh and placement, and picking it inside the pass would mean naming a Blender
/// object in C#. This handle is what lets the graph make that choice instead.
/// </remarks>
public readonly record struct SceneObjectHandle(ulong Value);
