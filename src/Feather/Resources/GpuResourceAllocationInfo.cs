namespace Feather.Resources;

/// <summary>
/// Exact physical allocation evidence reported by the active GPU backend.
/// </summary>
/// <param name="Available">
/// <see langword="true"/> only when the resource has been materialized and the backend exposes exact evidence.
/// </param>
/// <param name="PhysicalBytes">The exact byte size of the physical allocation represented by <paramref name="AllocationGroup"/>.</param>
/// <param name="AllocationGroup">
/// An opaque backend-owned identifier. Equal non-zero values identify the same physical allocation;
/// the value is never a native handle or pointer.
/// </param>
public readonly record struct GpuResourceAllocationInfo(
    bool Available,
    ulong PhysicalBytes,
    ulong AllocationGroup);

/// <summary>
/// Exposes side-effect-free physical allocation evidence for a managed GPU resource.
/// Querying an unmaterialized resource does not cause a GPU allocation, enqueue work,
/// or acquire the managed queue-recording gate.
/// </summary>
public interface IGpuResourceAllocation
{
    GpuResourceAllocationInfo AllocationInfo { get; }
}
