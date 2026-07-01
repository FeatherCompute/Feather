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
