namespace Feather;

public enum BufferAccess : uint
{
    ReadOnly = 1,
    WriteOnly = 2,
    ReadWrite = 3
}

public enum TextureAccess : uint
{
    ReadOnly = 1,
    WriteOnly = 2,
    ReadWrite = 3,
    Sampled = 4,
    RenderTarget = 5,
    DepthStencil = 6
}

public enum PixelFormat : uint
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

public enum ResourceKind : uint
{
    Buffer,
    Texture2D,
    Texture3D,
    Sampler,
    Uniform,
    UniformBuffer,
    PushConstant,
    SharedMemory
}

public enum ResourceAccess : uint
{
    Read,
    Write,
    ReadWrite,
    Sample,
    RenderTarget,
    Depth
}

public enum KernelDimension : uint
{
    One = 1,
    Two = 2,
    Three = 3
}

public enum DefaultThreadGroupSizes
{
    X,
    XY,
    XYZ
}

public enum PrimitiveTopology : uint
{
    TriangleList,
    TriangleStrip,
    LineList,
    LineStrip,
    PointList,
    TriangleFan
}

public enum SampleCount : uint
{
    X1 = 1,
    X2 = 2,
    X4 = 4,
    X8 = 8,
    X16 = 16
}

public enum GpuLayout
{
    Std430
}

/// <summary>
/// Identifies the native route used by the most recent dispatch or draw.
/// </summary>
public enum DispatchPath : uint
{
    None = 0,
    TypedEasyGpu = 1,
    CpuReferenceFallback = 2,
    GraphicsFallback = 3,
    Rejected = 4
}
