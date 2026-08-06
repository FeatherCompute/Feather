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
    /// Returns the generated resource table for a compute kernel.
    /// </summary>
    /// <typeparam name="TKernel">The generated compute kernel type.</typeparam>
    /// <returns>The resource descriptors emitted by the Roslyn generator.</returns>
    public static ResourceDescriptor[] GetResources<TKernel>()
        where TKernel : struct, IGeneratedKernel<TKernel>
        => TKernel.Descriptor.Resources;

    /// <summary>
    /// Returns generated graphics FEIR payloads for a pipeline pair.
    /// </summary>
    /// <typeparam name="TVS">The generated vertex shader type.</typeparam>
    /// <typeparam name="TFS">The generated fragment shader type.</typeparam>
    /// <typeparam name="TVaryings">The varying struct shared by both stages.</typeparam>
    /// <returns>The generated graphics IR payloads.</returns>
    public static GraphicsShaderIr GetGraphicsIR<TVS, TFS, TVaryings>()
        where TVS : struct, IGeneratedGraphicsPipeline<TVS, TFS, TVaryings>
        where TFS : struct, IGeneratedGraphicsPipeline<TVS, TFS, TVaryings>
        where TVaryings : unmanaged
        => new(
            Convert.ToHexString(TVS.IR),
            Convert.ToHexString(TVS.VertexIR),
            Convert.ToHexString(TVS.FragmentIR));
}

public readonly record struct GraphicsShaderIr(string IR, string VertexIR, string FragmentIR);
