using System.Diagnostics;
using Feather.Native;

namespace Feather.Resources;

/// <summary>
/// Describes the observable lifecycle of an asynchronous texture readback.
/// </summary>
public enum ReadbackOperationState
{
    Pending,
    Completed,
    Mapped,
    Consumed,
    Cancelled,
    Faulted,
    Disposed
}

/// <summary>
/// Owns one asynchronous texture-to-staging submission and its exactly-once mapping.
/// </summary>
public sealed class ReadbackOperation : IDisposable, IAsyncDisposable
{
    private const ulong WaitSliceNanoseconds = 10_000_000;

    private readonly object gate = new();
    private readonly GpuContext context;
    private readonly FeReadbackHandle handle;
    private readonly int expectedByteLength;
    private readonly int expectedRowPitch;
    private NativeHandleLease? textureLease;
    private NativeHandleLease? stagingLease;
    private ReadbackOperationState state = ReadbackOperationState.Pending;
    private Exception? fault;
    private IntPtr mappedData;
    private long nextMappingToken;
    private long activeMappingToken;
    private bool cleanupClaimed;

    private ReadbackOperation(
        GpuContext context,
        FeReadbackHandle handle,
        NativeHandleLease textureLease,
        NativeHandleLease stagingLease,
        int expectedByteLength,
        int expectedRowPitch)
    {
        this.context = context;
        this.handle = handle;
        this.textureLease = textureLease;
        this.stagingLease = stagingLease;
        this.expectedByteLength = expectedByteLength;
        this.expectedRowPitch = expectedRowPitch;
    }

    ~ReadbackOperation()
    {
        DisposeWithoutContextLease(throwOnError: false);
    }

    /// <summary>
    /// Gets the current operation lifecycle state.
    /// </summary>
    public ReadbackOperationState State
    {
        get
        {
            lock (gate)
            {
                return state;
            }
        }
    }

    /// <summary>
    /// Polls the native submission and records completion when it is observed.
    /// </summary>
    public bool IsCompleted
    {
        get
        {
            using var contextOperation = context.EnterOperation();
            lock (gate)
            {
                ThrowIfUnavailableLocked();
                if (state is ReadbackOperationState.Completed or ReadbackOperationState.Mapped or ReadbackOperationState.Consumed)
                {
                    return true;
                }

                try
                {
                    NativeMethods.ThrowIfFailed(NativeMethods.fe_readback_is_complete(handle, out var completed));
                    if (completed)
                    {
                        state = ReadbackOperationState.Completed;
                    }
                    return completed;
                }
                catch (Exception exception)
                {
                    RecordFaultLocked(exception);
                    throw;
                }
            }
        }
    }

    internal static ReadbackOperation Begin(
        GpuContext context,
        FeTextureHandle texture,
        int textureWidth,
        int textureHeight,
        int textureMipLevels,
        PixelFormat format,
        GpuBuffer<byte> staging,
        int x,
        int y,
        int width,
        int height,
        long stagingByteOffset,
        int mipLevel)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(texture);
        ArgumentNullException.ThrowIfNull(staging);
        ArgumentOutOfRangeException.ThrowIfNegative(x);
        ArgumentOutOfRangeException.ThrowIfNegative(y);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegative(stagingByteOffset);
        ArgumentOutOfRangeException.ThrowIfNegative(mipLevel);

        if (!ReferenceEquals(context, staging.Context))
        {
            throw new ArgumentException("The texture and staging buffer must belong to the same GPU context.", nameof(staging));
        }
        if (mipLevel >= textureMipLevels)
        {
            throw new ArgumentOutOfRangeException(nameof(mipLevel), "Texture readback mip level exceeds the allocated mip chain.");
        }

        var mipWidth = System.Math.Max(1, textureWidth >> mipLevel);
        var mipHeight = System.Math.Max(1, textureHeight >> mipLevel);
        if (x > mipWidth || width > mipWidth - x || y > mipHeight || height > mipHeight - y)
        {
            throw new ArgumentOutOfRangeException(nameof(width), "Texture readback region exceeds the selected mip dimensions.");
        }

        var bytesPerPixel = GetColorPixelSize(format);
        var rowPitch = checked(width * bytesPerPixel);
        var byteLength = checked(rowPitch * height);
        var stagingCapacity = staging.SizeInBytes;
        if (stagingByteOffset > stagingCapacity || byteLength > stagingCapacity - stagingByteOffset)
        {
            throw new ArgumentOutOfRangeException(nameof(stagingByteOffset), "Texture readback range exceeds the staging buffer.");
        }
        if ((stagingByteOffset % 4) != 0 || (stagingByteOffset % bytesPerPixel) != 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(stagingByteOffset),
                "Texture readback staging offset must satisfy four-byte and pixel alignment.");
        }

        NativeHandleLease? textureLease = null;
        NativeHandleLease? stagingLease = null;
        FeReadbackHandle? readbackHandle = null;
        try
        {
            textureLease = new NativeHandleLease(texture);
            var stagingHandle = staging.GetNativeHandle();
            stagingLease = new NativeHandleLease(stagingHandle);

            using var contextOperation = context.EnterOperation();
            lock (context.QueueGate)
            {
                NativeMethods.ThrowIfFailed(NativeMethods.fe_texture2d_begin_readback_mip(
                    context.Handle,
                    texture,
                    stagingHandle,
                    (uint)mipLevel,
                    (uint)x,
                    (uint)y,
                    (uint)width,
                    (uint)height,
                    (ulong)stagingByteOffset,
                    out readbackHandle));
            }

            var result = new ReadbackOperation(
                context,
                readbackHandle,
                textureLease,
                stagingLease,
                byteLength,
                rowPitch);
            context.RegisterReadback(result);
            textureLease = null;
            stagingLease = null;
            readbackHandle = null;
            return result;
        }
        catch
        {
            if (readbackHandle is not null)
            {
                _ = readbackHandle.ReleaseReadback();
                readbackHandle.Dispose();
            }
            stagingLease?.Dispose();
            textureLease?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Waits until the readback completes.
    /// </summary>
    public void Wait()
        => _ = Wait(Timeout.InfiniteTimeSpan);

    /// <summary>
    /// Waits for completion up to the supplied timeout.
    /// </summary>
    public bool Wait(TimeSpan timeout)
        => WaitCore(timeout, CancellationToken.None);

    /// <summary>
    /// Asynchronously waits until the readback completes without blocking the caller.
    /// </summary>
    public async ValueTask WaitAsync(CancellationToken cancellationToken = default)
    {
        _ = await WaitAsync(Timeout.InfiniteTimeSpan, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Asynchronously waits for completion up to the supplied timeout.
    /// </summary>
    public ValueTask<bool> WaitAsync(TimeSpan timeout, CancellationToken cancellationToken = default)
    {
        ValidateTimeout(timeout);
        cancellationToken.ThrowIfCancellationRequested();
        return WaitAsyncCore(timeout, cancellationToken);
    }

    /// <summary>
    /// Maps a completed readback exactly once. Completion must first be observed through
    /// <see cref="IsCompleted"/>, <see cref="Wait()"/>, or <see cref="WaitAsync(CancellationToken)"/>.
    /// </summary>
    public ReadbackMapping Map()
    {
        using var contextOperation = context.EnterOperation();
        lock (gate)
        {
            ThrowIfUnavailableLocked();
            if (state != ReadbackOperationState.Completed)
            {
                throw new InvalidOperationException("The readback must be observed complete before it can be mapped.");
            }

            try
            {
                NativeMethods.ThrowIfFailed(NativeMethods.fe_readback_map(handle, out var mapping));
                if (mapping.Data == IntPtr.Zero || mapping.ByteSize != (ulong)expectedByteLength ||
                    mapping.RowPitch != (ulong)expectedRowPitch)
                {
                    try
                    {
                        NativeMethods.ThrowIfFailed(NativeMethods.fe_readback_unmap(handle));
                    }
                    catch
                    {
                    }
                    throw new InvalidDataException("Native readback mapping metadata does not match the submitted region.");
                }

                mappedData = mapping.Data;
                activeMappingToken = checked(++nextMappingToken);
                state = ReadbackOperationState.Mapped;
                return new ReadbackMapping(this, activeMappingToken, expectedByteLength, expectedRowPitch);
            }
            catch (Exception exception)
            {
                RecordFaultLocked(exception);
                throw;
            }
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        GpuContext.OperationLease? contextOperation = null;
        try
        {
            contextOperation = context.EnterOperation();
        }
        catch (ObjectDisposedException)
        {
            DisposeWithoutContextLease(throwOnError: false);
            GC.SuppressFinalize(this);
            return;
        }

        using (contextOperation)
        {
            DisposeWithoutContextLease(throwOnError: true);
        }
        GC.SuppressFinalize(this);
    }

    /// <inheritdoc />
    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }

    internal void CancelForContextShutdown()
    {
        DisposeWithoutContextLease(throwOnError: false);
        GC.SuppressFinalize(this);
    }

    internal unsafe void CopyMappedData(long token, Span<byte> destination)
    {
        using var contextOperation = context.EnterOperation();
        lock (gate)
        {
            ValidateActiveMappingLocked(token);
            if (destination.Length < expectedByteLength)
            {
                throw new ArgumentException("Destination span is shorter than the readback mapping.", nameof(destination));
            }

            new ReadOnlySpan<byte>((void*)mappedData, expectedByteLength).CopyTo(destination);
        }
    }

    internal void ReleaseMapping(long token)
    {
        lock (gate)
        {
            if (state != ReadbackOperationState.Mapped || activeMappingToken != token)
            {
                return;
            }
        }

        NativeHandleLease? detachedTextureLease;
        NativeHandleLease? detachedStagingLease;
        GpuContext.OperationLease contextOperation;
        try
        {
            contextOperation = context.EnterOperation();
        }
        catch (ObjectDisposedException)
        {
            DisposeWithoutContextLease(throwOnError: false);
            return;
        }

        using (contextOperation)
        {
            lock (gate)
            {
                if (state != ReadbackOperationState.Mapped || activeMappingToken != token)
                {
                    return;
                }

                try
                {
                    NativeMethods.ThrowIfFailed(NativeMethods.fe_readback_unmap(handle));
                    mappedData = IntPtr.Zero;
                    activeMappingToken = 0;
                    state = ReadbackOperationState.Consumed;
                    ClaimCleanupLocked(out detachedTextureLease, out detachedStagingLease);
                }
                catch (Exception exception)
                {
                    RecordFaultLocked(exception);
                    throw;
                }
            }
        }

        ExecuteCleanup(detachedTextureLease, detachedStagingLease, throwOnError: true);
    }

    private bool WaitCore(TimeSpan timeout, CancellationToken cancellationToken)
    {
        ValidateTimeout(timeout);
        cancellationToken.ThrowIfCancellationRequested();
        if (timeout == TimeSpan.Zero)
        {
            return IsCompleted;
        }

        var stopwatch = timeout == Timeout.InfiniteTimeSpan ? null : Stopwatch.StartNew();
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var remaining = GetRemaining(timeout, stopwatch);
            if (remaining == TimeSpan.Zero)
            {
                return false;
            }

            var waitNanoseconds = remaining == Timeout.InfiniteTimeSpan
                ? WaitSliceNanoseconds
                : System.Math.Min(ToNanoseconds(remaining), WaitSliceNanoseconds);

            using var contextOperation = context.EnterOperation();
            lock (gate)
            {
                ThrowIfUnavailableLocked();
                if (state is ReadbackOperationState.Completed or ReadbackOperationState.Mapped or ReadbackOperationState.Consumed)
                {
                    return true;
                }

                try
                {
                    NativeMethods.ThrowIfFailed(NativeMethods.fe_readback_wait(handle, waitNanoseconds, out var completed));
                    if (completed)
                    {
                        state = ReadbackOperationState.Completed;
                        return true;
                    }
                }
                catch (Exception exception)
                {
                    RecordFaultLocked(exception);
                    throw;
                }
            }
        }
    }

    private async ValueTask<bool> WaitAsyncCore(TimeSpan timeout, CancellationToken cancellationToken)
    {
        if (timeout == TimeSpan.Zero)
        {
            return IsCompleted;
        }

        var stopwatch = timeout == Timeout.InfiniteTimeSpan ? null : Stopwatch.StartNew();
        var observationDelay = TimeSpan.FromMilliseconds(1);
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (IsCompleted)
            {
                return true;
            }

            var remaining = GetRemaining(timeout, stopwatch);
            if (remaining == TimeSpan.Zero)
            {
                return false;
            }

            // The native fence is the correctness signal. This delay only bounds observation overhead.
            var delay = remaining == Timeout.InfiniteTimeSpan || observationDelay <= remaining
                ? observationDelay
                : remaining;
            await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
            if (observationDelay < TimeSpan.FromMilliseconds(8))
            {
                observationDelay += observationDelay;
            }
        }
    }

    private void DisposeWithoutContextLease(bool throwOnError)
    {
        NativeHandleLease? detachedTextureLease;
        NativeHandleLease? detachedStagingLease;
        lock (gate)
        {
            if (cleanupClaimed)
            {
                if (state == ReadbackOperationState.Consumed)
                {
                    state = ReadbackOperationState.Disposed;
                }
                return;
            }

            state = state == ReadbackOperationState.Pending
                ? ReadbackOperationState.Cancelled
                : ReadbackOperationState.Disposed;
            mappedData = IntPtr.Zero;
            activeMappingToken = 0;
            ClaimCleanupLocked(out detachedTextureLease, out detachedStagingLease);
        }

        ExecuteCleanup(detachedTextureLease, detachedStagingLease, throwOnError);
    }

    private void ClaimCleanupLocked(
        out NativeHandleLease? detachedTextureLease,
        out NativeHandleLease? detachedStagingLease)
    {
        cleanupClaimed = true;
        detachedTextureLease = textureLease;
        detachedStagingLease = stagingLease;
        textureLease = null;
        stagingLease = null;
    }

    private void ExecuteCleanup(
        NativeHandleLease? detachedTextureLease,
        NativeHandleLease? detachedStagingLease,
        bool throwOnError)
    {
        Exception? error = null;
        try
        {
            var result = handle.ReleaseReadback();
            if (!result.Succeeded())
            {
                error = new FeatherNativeException(result, NativeMethods.GetLastError());
            }
        }
        catch (Exception exception)
        {
            error = exception;
        }
        finally
        {
            handle.Dispose();
            detachedStagingLease?.Dispose();
            detachedTextureLease?.Dispose();
            context.UnregisterReadback(this);
        }

        if (throwOnError && error is not null)
        {
            throw error;
        }
    }

    private void ValidateActiveMappingLocked(long token)
    {
        if (state != ReadbackOperationState.Mapped || activeMappingToken != token || mappedData == IntPtr.Zero)
        {
            throw new ObjectDisposedException(nameof(ReadbackMapping), "The readback mapping is no longer active.");
        }
    }

    private void ThrowIfUnavailableLocked()
    {
        if (state == ReadbackOperationState.Faulted)
        {
            throw new InvalidOperationException("The readback operation faulted.", fault);
        }
        if (state is ReadbackOperationState.Cancelled or ReadbackOperationState.Disposed)
        {
            throw new ObjectDisposedException(nameof(ReadbackOperation));
        }
    }

    private void RecordFaultLocked(Exception exception)
    {
        fault = exception;
        state = ReadbackOperationState.Faulted;
    }

    private static int GetColorPixelSize(PixelFormat format)
        => format switch
        {
            PixelFormat.R8 => 1,
            PixelFormat.Rg8 or PixelFormat.R16Float => 2,
            PixelFormat.Rgba8 or PixelFormat.Bgra8 or PixelFormat.Rg16Float or PixelFormat.R32Float => 4,
            PixelFormat.Rgba16Float or PixelFormat.Rg32Float => 8,
            PixelFormat.Rgba32Float => 16,
            _ => throw new NotSupportedException($"Asynchronous readback does not support {format} textures.")
        };

    private static void ValidateTimeout(TimeSpan timeout)
    {
        if (timeout < TimeSpan.Zero && timeout != Timeout.InfiniteTimeSpan)
        {
            throw new ArgumentOutOfRangeException(nameof(timeout));
        }
    }

    private static TimeSpan GetRemaining(TimeSpan timeout, Stopwatch? stopwatch)
    {
        if (timeout == Timeout.InfiniteTimeSpan)
        {
            return Timeout.InfiniteTimeSpan;
        }

        var remaining = timeout - stopwatch!.Elapsed;
        return remaining <= TimeSpan.Zero ? TimeSpan.Zero : remaining;
    }

    private static ulong ToNanoseconds(TimeSpan timeout)
    {
        var ticks = (ulong)timeout.Ticks;
        return ticks >= ulong.MaxValue / 100UL ? ulong.MaxValue : ticks * 100UL;
    }
}

/// <summary>
/// A stack-confined lease over one readback mapping. It exposes copy operations, never a pointer or escaping span.
/// </summary>
public ref struct ReadbackMapping
{
    private ReadbackOperation? owner;
    private readonly long token;

    internal ReadbackMapping(ReadbackOperation owner, long token, int byteLength, int rowPitch)
    {
        this.owner = owner;
        this.token = token;
        ByteLength = byteLength;
        RowPitch = rowPitch;
    }

    /// <summary>
    /// Gets the number of mapped bytes.
    /// </summary>
    public int ByteLength { get; }

    /// <summary>
    /// Gets the number of tightly packed bytes in one image row.
    /// </summary>
    public int RowPitch { get; }

    /// <summary>
    /// Copies the active mapping into caller-owned memory.
    /// </summary>
    public readonly void CopyTo(Span<byte> destination)
    {
        var currentOwner = owner ?? throw new ObjectDisposedException(nameof(ReadbackMapping));
        currentOwner.CopyMappedData(token, destination);
    }

    /// <summary>
    /// Unmaps and consumes this readback. Copies of the lease share the same exactly-once token.
    /// </summary>
    public void Dispose()
    {
        var currentOwner = owner;
        owner = null;
        currentOwner?.ReleaseMapping(token);
    }
}
