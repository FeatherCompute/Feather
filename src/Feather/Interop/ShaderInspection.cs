using System.Reflection;

namespace Feather.Interop;

public static class ShaderInspection
{
    /// <summary>
    /// Returns the serialized Feather IR payload for a generated kernel as hexadecimal text.
    /// </summary>
    /// <typeparam name="TKernel">The generated compute kernel type.</typeparam>
    /// <returns>The generated kernel IR encoded as uppercase hexadecimal text.</returns>
    public static string GetIR<TKernel>()
        where TKernel : struct, IGeneratedKernel<TKernel>
        => Convert.ToHexString(TKernel.IR);

    /// <summary>
    /// Builds a generated kernel through the EasyGPU IR module bridge and returns the unoptimized GLSL source.
    /// </summary>
    /// <typeparam name="TKernel">The generated compute kernel type.</typeparam>
    /// <returns>The GLSL source produced after EasyGPU lowers the generated module.</returns>
    public static string GetGLSL<TKernel>()
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        using var kernel = GpuKernel.Create<TKernel>(GPU.Context);
        return kernel.GetGLSL();
    }

    /// <summary>
    /// Builds a generated kernel through the EasyGPU IR module bridge and returns the backend-optimized GLSL inspection dump.
    /// </summary>
    /// <typeparam name="TKernel">The generated compute kernel type.</typeparam>
    /// <returns>The optimized GLSL produced by the active EasyGPU backend.</returns>
    public static string GetOptimizedGLSL<TKernel>()
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        using var kernel = GpuKernel.Create<TKernel>(GPU.Context);
        return kernel.GetOptimizedGLSL();
    }

    /// <summary>
    /// Builds a generated kernel through the active backend and returns the optimized target IR.
    /// </summary>
    /// <typeparam name="TKernel">The generated compute kernel type.</typeparam>
    /// <returns>Backend-specific optimized target IR, such as SPIR-V assembly on Vulkan.</returns>
    public static string GetOptimizedIR<TKernel>()
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        using var kernel = GpuKernel.Create<TKernel>(GPU.Context);
        return kernel.GetOptimizedIR();
    }

    /// <summary>
    /// Builds a generated kernel through the active backend and returns its structured optimizer decisions.
    /// </summary>
    /// <typeparam name="TKernel">The generated compute kernel type.</typeparam>
    /// <returns>A versioned backend-owned JSON optimization report.</returns>
    public static string GetOptimizationReport<TKernel>()
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        using var kernel = GpuKernel.Create<TKernel>(GPU.Context);
        return kernel.GetOptimizationReport();
    }

    /// <summary>
    /// Inspects shaders that are currently loaded by the default GPU context for one assembly.
    /// This method does not discover types or create pipelines merely to produce inspection output.
    /// </summary>
    /// <param name="assembly">The generated pass assembly whose loaded shaders should be returned.</param>
    /// <returns>An immutable snapshot of exact backend inputs and available optimized outputs.</returns>
    public static IReadOnlyList<LoadedShaderSource> GetLoadedShaders(Assembly assembly)
        => GPU.Context.GetLoadedShaders(assembly);

    /// <summary>
    /// Returns the generated resource table for a compute kernel.
    /// </summary>
    /// <typeparam name="TKernel">The generated compute kernel type.</typeparam>
    /// <returns>The resource descriptors emitted by the Roslyn generator.</returns>
    public static ResourceDescriptor[] GetResources<TKernel>()
        where TKernel : struct, IGeneratedKernel<TKernel>
        => TKernel.Descriptor.Resources;

    /// <summary>
    /// Returns generated graphics shader inspection payloads for a pipeline pair.
    /// </summary>
    /// <typeparam name="TVS">The generated vertex shader type.</typeparam>
    /// <typeparam name="TFS">The generated fragment shader type.</typeparam>
    /// <typeparam name="TVaryings">The varying struct shared by both stages.</typeparam>
    /// <returns>The generated graphics shader source payloads currently available to Feather.</returns>
    public static GraphicsShaderSource GetGraphicsSource<TVS, TFS, TVaryings>()
        where TVS : struct, IGeneratedGraphicsPipeline<TVS, TFS, TVaryings>
        where TFS : struct, IGeneratedGraphicsPipeline<TVS, TFS, TVaryings>
        where TVaryings : unmanaged
        => new(
            Convert.ToHexString(TVS.IR),
            Convert.ToHexString(TVS.VertexIR),
            Convert.ToHexString(TVS.FragmentIR),
            string.Empty,
            string.Empty);
}

public readonly record struct GraphicsShaderSource(string IR, string VertexIR, string FragmentIR, string VertexGLSL, string FragmentGLSL);

/// <summary>
/// Identifies a generated shader stage loaded by Feather.
/// </summary>
public enum ShaderStage
{
    Compute,
    Vertex,
    Fragment
}

/// <summary>
/// Identifies the exact target-binary format used to create a loaded shader module.
/// </summary>
public enum ShaderBinaryFormat
{
    Unavailable,
    SpirV
}

/// <summary>
/// Exact shader material captured from a live Feather kernel or graphics pipeline.
/// Optimized fields are empty when the active backend cannot expose that representation.
/// </summary>
public sealed record LoadedShaderSource(
    Type SourceType,
    ShaderStage Stage,
    bool AutoDiff,
    string BackendInputGLSL,
    string OptimizedBackendGLSL,
    string OptimizedTargetIR,
    string OptimizationReportJson,
    ShaderBinaryFormat TargetBinaryFormat,
    ReadOnlyMemory<byte> TargetBinary);

internal interface ILoadedGraphicsShaderInspection
{
    IReadOnlyList<LoadedShaderSource> InspectLoadedShaders();
}
