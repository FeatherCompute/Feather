using Feather.Native;

namespace Feather;

/// <summary>
/// Provides process-wide profiling controls and aggregate timings for Feather GPU commands.
/// </summary>
/// <remarks>
/// The current native backend records successful compute dispatches and graphics draw calls. Basic generated compute
/// kernels use the typed EasyGPU backend path; compatibility and graphics fallback commands are recorded with the same
/// aggregate counters for diagnostics.
/// </remarks>
public static class GpuProfiler
{
    /// <summary>
    /// Gets a value indicating whether profiler recording is enabled.
    /// </summary>
    public static bool IsEnabled
    {
        get
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_profiler_is_enabled(out var enabled));
            return enabled;
        }
    }

    /// <summary>
    /// Enables or disables process-wide GPU command profiling.
    /// </summary>
    /// <param name="enabled">True to collect future command timings; false to stop collecting timings.</param>
    public static void SetEnabled(bool enabled)
        => NativeMethods.ThrowIfFailed(NativeMethods.fe_profiler_set_enabled(enabled));

    /// <summary>
    /// Removes all recorded profiler entries and aggregate statistics.
    /// </summary>
    public static void Clear()
        => NativeMethods.ThrowIfFailed(NativeMethods.fe_profiler_clear());

    /// <summary>
    /// Gets the total elapsed time, in milliseconds, across all recorded command names.
    /// </summary>
    /// <returns>The total recorded elapsed time in milliseconds.</returns>
    public static double GetTotalTimeMs()
    {
        NativeMethods.ThrowIfFailed(NativeMethods.fe_profiler_get_total_time(out var totalTimeMs));
        return totalTimeMs;
    }

    /// <summary>
    /// Queries aggregate profiler statistics for a command name.
    /// </summary>
    /// <param name="name">The generated kernel name, graphics pipeline name, or explicit debug name to query.</param>
    /// <returns>Aggregate timing data. Missing names return a result with a zero count.</returns>
    public static GpuProfilerQuery Query(string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        NativeMethods.ThrowIfFailed(NativeMethods.fe_profiler_query(name, out var result));
        return new GpuProfilerQuery(
            name,
            result.Count,
            result.MinTimeMs,
            result.MaxTimeMs,
            result.AverageTimeMs,
            result.TotalTimeMs);
    }

    /// <summary>
    /// Gets a formatted aggregate profiler report suitable for logs and diagnostics.
    /// </summary>
    /// <returns>A formatted profiler report.</returns>
    public static string GetFormattedReport()
        => NativeStringCall.GetString((IntPtr buffer, UIntPtr length, out UIntPtr required) =>
            NativeMethods.fe_profiler_get_formatted(buffer, length, out required));
}

/// <summary>
/// Represents aggregate profiler timings for a single GPU command name.
/// </summary>
/// <param name="Name">The generated kernel name, graphics pipeline name, or explicit debug name.</param>
/// <param name="Count">The number of recorded executions.</param>
/// <param name="MinTimeMs">The minimum recorded elapsed time in milliseconds.</param>
/// <param name="MaxTimeMs">The maximum recorded elapsed time in milliseconds.</param>
/// <param name="AverageTimeMs">The average recorded elapsed time in milliseconds.</param>
/// <param name="TotalTimeMs">The total recorded elapsed time in milliseconds.</param>
public readonly record struct GpuProfilerQuery(
    string Name,
    ulong Count,
    double MinTimeMs,
    double MaxTimeMs,
    double AverageTimeMs,
    double TotalTimeMs);
