using Feather.Graphics;
using Feather.Math;

namespace Feather.Interop;

public interface IGeneratedKernel<TKernel>
    where TKernel : struct
{
    static abstract ReadOnlySpan<byte> IR { get; }
    static abstract KernelDescriptor Descriptor { get; }
    static abstract void Bind(in TKernel kernel, GpuKernelCommand command);
}

public interface IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
    where TVertexShader : struct
    where TFragmentShader : struct
    where TVaryings : unmanaged
{
    static abstract ReadOnlySpan<byte> IR { get; }
    static abstract ReadOnlySpan<byte> VertexIR { get; }
    static abstract ReadOnlySpan<byte> FragmentIR { get; }
    static abstract GraphicsPipelineDescriptor Descriptor { get; }
    static abstract void BindVertex(in TVertexShader shader, GpuGraphicsCommand command);
    static abstract void BindFragment(in TFragmentShader shader, GpuGraphicsCommand command);
}

public readonly record struct KernelDescriptor(
    KernelDimension Dimension,
    int3 ThreadGroupSize,
    ResourceDescriptor[] Resources,
    PushConstantDescriptor[] PushConstants,
    bool BoundsCheck,
    bool AutoDiff,
    string DebugName);

public readonly record struct GraphicsPipelineDescriptor(
    ResourceDescriptor[] Resources,
    PushConstantDescriptor[] PushConstants,
    string VertexShaderName,
    string FragmentShaderName);

public readonly record struct ResourceDescriptor(
    uint Binding,
    ResourceKind Kind,
    ResourceAccess Access,
    Type ElementType,
    string Name);

public readonly record struct PushConstantDescriptor(
    uint Offset,
    uint Size,
    Type Type,
    string Name);
