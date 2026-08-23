using Feather.Interop;
using Feather.Math;

namespace Feather;

/// <summary>
/// One context-scoped diagnostic substitution. Implementations own their scratch kernel and
/// buffers; ordinary cached kernels never implement or retain this interface.
/// </summary>
internal interface IGpuDiagnosticCapture
{
    bool TryGetOrCreateKernel<TKernel>(bool autoDiff, out GpuKernel diagnosticKernel)
        where TKernel : struct, IGeneratedKernel<TKernel>;

    void Bind(
        GpuKernel dispatchKernel,
        GpuKernelCommand command,
        GpuDispatchSize logicalSize,
        int3 threadGroupSize);

    void DisposeForContextShutdown();
}
