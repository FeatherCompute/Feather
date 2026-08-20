using System.Diagnostics;
using System.Runtime.ExceptionServices;
using Feather.Graphics;
using Feather.Interop;
using Feather.Math;
using Feather.Native;
using Feather.Resources;

namespace Feather;

/// <summary>
/// Selects the memory dependencies made visible by a command-list barrier.
/// </summary>
[Flags]
public enum GpuMemoryBarrier : uint
{
    None = 0,
    Buffer = 1u << 0,
    Texture = 1u << 1,
    Uniform = 1u << 2,
    All = Buffer | Texture | Uniform
}

/// <summary>
/// Records a reusable, ordered sequence of GPU commands.
/// </summary>
public sealed class GpuCommandList : IDisposable
{
    private readonly object gate = new();
    private readonly GpuContext context;
    private readonly List<IRecordedGpuCommand> commands = [];
    private bool closed;
    private bool disposed;

    internal GpuCommandList(GpuContext context)
    {
        this.context = context;
    }

    public int Count
    {
        get
        {
            lock (gate)
            {
                return commands.Count;
            }
        }
    }

    public bool IsClosed
    {
        get
        {
            lock (gate)
            {
                return closed;
            }
        }
    }

    public bool IsDisposed
    {
        get
        {
            lock (gate)
            {
                return disposed;
            }
        }
    }

    public void Dispatch<TKernel>(TKernel kernel, int x)
        where TKernel : struct, IKernel1D, IGeneratedKernel<TKernel>
        => AddDispatch(kernel, new GpuDispatchSize(x, 1, 1));

    public void Dispatch<TKernel>(TKernel kernel, int2 size)
        where TKernel : struct, IKernel2D, IGeneratedKernel<TKernel>
        => AddDispatch(kernel, new GpuDispatchSize(size.X, size.Y, 1));

    public void Dispatch<TKernel>(TKernel kernel, int3 size)
        where TKernel : struct, IKernel3D, IGeneratedKernel<TKernel>
        => AddDispatch(kernel, new GpuDispatchSize(size.X, size.Y, size.Z));

    /// <summary>
    /// Records a type-safe buffer copy. Source and destination ranges are expressed in elements.
    /// </summary>
    public void CopyBuffer<T>(
        GpuBuffer<T> source,
        int sourceIndex,
        GpuBuffer<T> destination,
        int destinationIndex,
        int count)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        ArgumentOutOfRangeException.ThrowIfNegative(sourceIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(destinationIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        if (!ReferenceEquals(source.Context, context) || !ReferenceEquals(destination.Context, context))
        {
            throw new ArgumentException("Buffer copies require resources created by this command list's context.");
        }
        if (sourceIndex > source.Length || count > source.Length - sourceIndex)
        {
            throw new ArgumentOutOfRangeException(nameof(count), "Source copy range exceeds the buffer length.");
        }
        if (destinationIndex > destination.Length || count > destination.Length - destinationIndex)
        {
            throw new ArgumentOutOfRangeException(nameof(count), "Destination copy range exceeds the buffer length.");
        }
        if (count == 0)
        {
            return;
        }

        Add(new RecordedBufferCopy<T>(source, sourceIndex, destination, destinationIndex, count));
    }

    /// <summary>
    /// Records a full type-safe buffer copy.
    /// </summary>
    public void CopyBuffer<T>(GpuBuffer<T> source, GpuBuffer<T> destination)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        if (source.Length != destination.Length)
        {
            throw new ArgumentException("Full buffer copies require equal source and destination lengths.");
        }
        CopyBuffer(source, 0, destination, 0, source.Length);
    }

    /// <summary>
    /// Records an explicit memory dependency between preceding and subsequent commands.
    /// </summary>
    public void MemoryBarrier(GpuMemoryBarrier barriers = GpuMemoryBarrier.All)
    {
        if ((barriers & ~GpuMemoryBarrier.All) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(barriers));
        }
        if (barriers != GpuMemoryBarrier.None)
        {
            Add(new RecordedMemoryBarrier(barriers));
        }
    }

    /// <summary>
    /// Records a non-indexed graphics draw.
    /// </summary>
    public void Draw<TVertexShader, TFragmentShader, TVaryings>(
        GpuGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings> pipeline,
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        IGpuTexture2D target,
        uint vertexCount,
        GraphicsDrawDesc drawDesc = default)
        where TVertexShader : struct, IVertexShader<TVaryings>, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TFragmentShader : struct, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TVaryings : unmanaged
    {
        ArgumentNullException.ThrowIfNull(target);
        Draw(pipeline, vertexShader, fragmentShader, [target], vertexCount, drawDesc);
    }

    /// <summary>
    /// Records a non-indexed graphics draw with one color target and a depth target.
    /// </summary>
    public void Draw<TVertexShader, TFragmentShader, TVaryings>(
        GpuGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings> pipeline,
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        IGpuTexture2D target,
        IGpuTexture2D depthTarget,
        uint vertexCount,
        GraphicsDrawDesc drawDesc = default)
        where TVertexShader : struct, IVertexShader<TVaryings>, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TFragmentShader : struct, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TVaryings : unmanaged
    {
        ArgumentNullException.ThrowIfNull(target);
        ArgumentNullException.ThrowIfNull(depthTarget);
        Draw(pipeline, vertexShader, fragmentShader, [target], vertexCount, drawDesc, depthTarget);
    }

    /// <summary>
    /// Records a non-indexed graphics draw with one or more color targets.
    /// </summary>
    public void Draw<TVertexShader, TFragmentShader, TVaryings>(
        GpuGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings> pipeline,
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        IReadOnlyList<IGpuTexture2D> colorTargets,
        uint vertexCount,
        GraphicsDrawDesc drawDesc = default,
        IGpuTexture2D? depthTarget = null)
        where TVertexShader : struct, IVertexShader<TVaryings>, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TFragmentShader : struct, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TVaryings : unmanaged
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        var targets = ValidateDraw(pipeline, colorTargets, vertexCount);
        ValidateDepthTarget(depthTarget);
        Add(new RecordedGraphicsDraw<TVertexShader, TFragmentShader, TVaryings>(
            pipeline,
            vertexShader,
            fragmentShader,
            targets,
            depthTarget,
            vertexCount,
            drawDesc));
    }

    /// <summary>
    /// Records an indexed graphics draw using the complete index buffer.
    /// </summary>
    public void DrawIndexed<TVertexShader, TFragmentShader, TVaryings, TIndex>(
        GpuGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings> pipeline,
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        IGpuTexture2D target,
        GpuBuffer<TIndex> indices,
        GraphicsDrawDesc drawDesc = default)
        where TVertexShader : struct, IVertexShader<TVaryings>, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TFragmentShader : struct, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TVaryings : unmanaged
        where TIndex : unmanaged
    {
        ArgumentNullException.ThrowIfNull(target);
        DrawIndexed(pipeline, vertexShader, fragmentShader, [target], indices, drawDesc);
    }

    /// <summary>
    /// Records an indexed graphics draw with one color target and a depth target.
    /// </summary>
    public void DrawIndexed<TVertexShader, TFragmentShader, TVaryings, TIndex>(
        GpuGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings> pipeline,
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        IGpuTexture2D target,
        IGpuTexture2D depthTarget,
        GpuBuffer<TIndex> indices,
        GraphicsDrawDesc drawDesc = default)
        where TVertexShader : struct, IVertexShader<TVaryings>, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TFragmentShader : struct, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TVaryings : unmanaged
        where TIndex : unmanaged
    {
        ArgumentNullException.ThrowIfNull(target);
        ArgumentNullException.ThrowIfNull(depthTarget);
        DrawIndexed(pipeline, vertexShader, fragmentShader, [target], indices, drawDesc, depthTarget);
    }

    /// <summary>
    /// Records an indexed graphics draw using the complete index buffer.
    /// </summary>
    public void DrawIndexed<TVertexShader, TFragmentShader, TVaryings, TIndex>(
        GpuGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings> pipeline,
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        IReadOnlyList<IGpuTexture2D> colorTargets,
        GpuBuffer<TIndex> indices,
        GraphicsDrawDesc drawDesc = default,
        IGpuTexture2D? depthTarget = null)
        where TVertexShader : struct, IVertexShader<TVaryings>, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TFragmentShader : struct, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TVaryings : unmanaged
        where TIndex : unmanaged
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(indices);
        if (!ReferenceEquals(indices.Context, context))
        {
            throw new ArgumentException("The index buffer belongs to a different GPU context.", nameof(indices));
        }
        var targets = ValidateDraw(pipeline, colorTargets, checked((uint)indices.Length));
        ValidateDepthTarget(depthTarget);
        Add(new RecordedIndexedGraphicsDraw<TVertexShader, TFragmentShader, TVaryings, TIndex>(
            pipeline,
            vertexShader,
            fragmentShader,
            targets,
            depthTarget,
            indices,
            drawDesc));
    }

    /// <summary>
    /// Ends recording. A closed list can be submitted repeatedly until reset.
    /// </summary>
    public void Close()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            closed = true;
        }
    }

    /// <summary>
    /// Clears all commands and returns the list to the recording state.
    /// </summary>
    public void Reset()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            commands.Clear();
            closed = false;
        }
    }

    public void Dispose()
    {
        lock (gate)
        {
            if (disposed)
            {
                return;
            }
            commands.Clear();
            closed = true;
            disposed = true;
        }
    }

    internal GpuContext Context => context;

    internal IRecordedGpuCommand[] SnapshotForSubmission()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (!closed)
            {
                throw new InvalidOperationException("Close the GPU command list before submitting it.");
            }
            return [.. commands];
        }
    }

    private void AddDispatch<TKernel>(TKernel kernel, GpuDispatchSize size)
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        if (size.X <= 0 || size.Y <= 0 || size.Z <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(size), "Dispatch dimensions must be positive.");
        }
        Add(new RecordedGpuDispatch<TKernel>(kernel, size));
    }

    private void Add(IRecordedGpuCommand command)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (closed)
            {
                throw new InvalidOperationException("Reset the GPU command list before recording more commands.");
            }
            commands.Add(command);
        }
    }

    private IGpuTexture2D[] ValidateDraw<TVertexShader, TFragmentShader, TVaryings>(
        GpuGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings> pipeline,
        IReadOnlyList<IGpuTexture2D> colorTargets,
        uint count)
        where TVertexShader : struct, IVertexShader<TVaryings>, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TFragmentShader : struct, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TVaryings : unmanaged
    {
        ArgumentNullException.ThrowIfNull(colorTargets);
        if (!ReferenceEquals(pipeline.Context, context))
        {
            throw new ArgumentException("The graphics pipeline belongs to a different GPU context.", nameof(pipeline));
        }
        if (count == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(count), "Draw count must be positive.");
        }
        if (colorTargets.Count == 0 || colorTargets.Count > 8)
        {
            throw new ArgumentOutOfRangeException(nameof(colorTargets), "Graphics draws require one to eight color targets.");
        }
        if (colorTargets.Count != pipeline.Desc.ColorAttachmentCount)
        {
            throw new ArgumentException("Color target count must match the graphics pipeline descriptor.", nameof(colorTargets));
        }

        var targets = new IGpuTexture2D[colorTargets.Count];
        for (var i = 0; i < targets.Length; i++)
        {
            targets[i] = colorTargets[i] ?? throw new ArgumentException("Color targets must not contain null entries.", nameof(colorTargets));
            if (targets[i] is not IGpuTexture2DNative nativeTarget)
            {
                throw new ArgumentException("Color targets must be Feather GPU textures.", nameof(colorTargets));
            }
            if (!ReferenceEquals(nativeTarget.Context, context))
            {
                throw new ArgumentException("Color targets must belong to this command list's context.", nameof(colorTargets));
            }
            _ = nativeTarget.NativeHandle;
        }
        return targets;
    }

    private void ValidateDepthTarget(IGpuTexture2D? depthTarget)
    {
        if (depthTarget is null)
        {
            return;
        }
        if (depthTarget is not IGpuTexture2DNative nativeDepthTarget)
        {
            throw new ArgumentException("The depth target must be a Feather GPU texture.", nameof(depthTarget));
        }
        if (!ReferenceEquals(nativeDepthTarget.Context, context))
        {
            throw new ArgumentException("The depth target must belong to this command list's context.", nameof(depthTarget));
        }
        _ = nativeDepthTarget.NativeHandle;
    }

    internal interface IRecordedGpuCommand
    {
        void Execute(GpuContext context, List<IDisposable> submissionLeases);
    }

    private sealed class RecordedGpuDispatch<TKernel>(TKernel kernel, GpuDispatchSize size) : IRecordedGpuCommand
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        public void Execute(GpuContext context, List<IDisposable> submissionLeases)
        {
            var compiled = context.GetOrCreateKernel<TKernel>();
            submissionLeases.AddRange(GpuKernel.DispatchForQueue(context, compiled, kernel, size));
        }
    }

    private sealed class RecordedBufferCopy<T>(
        GpuBuffer<T> source,
        int sourceIndex,
        GpuBuffer<T> destination,
        int destinationIndex,
        int count) : IRecordedGpuCommand
        where T : unmanaged
    {
        public void Execute(GpuContext context, List<IDisposable> submissionLeases)
        {
            var sourceLease = new NativeHandleLease(source.Handle);
            var destinationLease = new NativeHandleLease(destination.Handle);
            try
            {
                NativeMethods.ThrowIfFailed(NativeMethods.fe_buffer_copy(
                    source.Handle,
                    (ulong)checked(sourceIndex * source.ElementStride),
                    destination.Handle,
                    (ulong)checked(destinationIndex * destination.ElementStride),
                    (ulong)checked(count * source.ElementStride)));
                submissionLeases.Add(sourceLease);
                submissionLeases.Add(destinationLease);
            }
            catch
            {
                destinationLease.Dispose();
                sourceLease.Dispose();
                throw;
            }
        }
    }

    private sealed class RecordedMemoryBarrier(GpuMemoryBarrier barriers) : IRecordedGpuCommand
    {
        public void Execute(GpuContext context, List<IDisposable> submissionLeases)
            => NativeMethods.ThrowIfFailed(NativeMethods.fe_queue_memory_barrier(context.Handle, (uint)barriers));
    }

    private sealed class RecordedGraphicsDraw<TVertexShader, TFragmentShader, TVaryings>(
        GpuGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings> pipeline,
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        IGpuTexture2D[] targets,
        IGpuTexture2D? depthTarget,
        uint vertexCount,
        GraphicsDrawDesc drawDesc) : IRecordedGpuCommand
        where TVertexShader : struct, IVertexShader<TVaryings>, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TFragmentShader : struct, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TVaryings : unmanaged
    {
        public void Execute(GpuContext context, List<IDisposable> submissionLeases)
            => submissionLeases.AddRange(pipeline.DrawForQueue(
                vertexShader,
                fragmentShader,
                targets,
                depthTarget,
                vertexCount,
                FeBufferHandle.Null,
                indexed: false,
                drawDesc));
    }

    private sealed class RecordedIndexedGraphicsDraw<TVertexShader, TFragmentShader, TVaryings, TIndex>(
        GpuGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings> pipeline,
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        IGpuTexture2D[] targets,
        IGpuTexture2D? depthTarget,
        GpuBuffer<TIndex> indices,
        GraphicsDrawDesc drawDesc) : IRecordedGpuCommand
        where TVertexShader : struct, IVertexShader<TVaryings>, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TFragmentShader : struct, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TVaryings : unmanaged
        where TIndex : unmanaged
    {
        public void Execute(GpuContext context, List<IDisposable> submissionLeases)
            => submissionLeases.AddRange(pipeline.DrawForQueue(
                vertexShader,
                fragmentShader,
                targets,
                depthTarget,
                checked((uint)indices.Length),
                indices.Handle,
                indexed: true,
                drawDesc));
    }
}

/// <summary>
/// Serializes command lists onto a GPU context's default queue.
/// </summary>
public sealed class GpuQueue
{
    private readonly GpuContext context;

    internal GpuQueue(GpuContext context)
    {
        this.context = context;
    }

    public GpuCommandList CreateCommandList()
    {
        context.ThrowIfDisposed();
        return new GpuCommandList(context);
    }

    /// <summary>
    /// Inserts a completion point after all work currently recorded on this queue.
    /// </summary>
    public GpuFence Signal()
        => SubmitSnapshots([]);

    /// <summary>
    /// Submits one closed command list and returns its queue-ordered completion fence.
    /// </summary>
    public GpuFence Submit(GpuCommandList commandList)
        => SubmitSnapshots([Snapshot(commandList, nameof(commandList))]);

    /// <summary>
    /// Atomically replays zero or more closed command lists in order and returns one completion fence.
    /// </summary>
    public GpuFence Submit(IReadOnlyList<GpuCommandList> commandLists)
    {
        ArgumentNullException.ThrowIfNull(commandLists);
        var snapshots = new GpuCommandList.IRecordedGpuCommand[commandLists.Count][];
        for (var i = 0; i < commandLists.Count; ++i)
        {
            var commandList = commandLists[i]
                ?? throw new ArgumentException($"Command list at index {i} is null.", nameof(commandLists));
            snapshots[i] = Snapshot(commandList, nameof(commandLists));
        }
        return SubmitSnapshots(snapshots);
    }

    /// <summary>
    /// Records one aggregate GPU timestamp interval around <paramref name="record"/> and submits
    /// the exact command stream that owns it. Unsupported backends still submit normally; the
    /// returned fence then reports no GPU timestamp rather than substituting CPU or fence time.
    /// </summary>
    public GpuTimestampedSubmission<T> SubmitTimestamped<T>(Func<T> record)
    {
        ArgumentNullException.ThrowIfNull(record);
        using var operation = context.EnterOperation();
        lock (context.QueueGate)
        {
            var leases = new List<IDisposable>();
            FeFenceHandle? nativeFence = null;
            GpuFence? managedFence = null;
            var failureSubmissionReaping = false;
            try
            {
                NativeMethods.ThrowIfFailed(NativeMethods.fe_queue_begin_submission_timestamp(
                    context.Handle,
                    out var timestampQuery));

                T result = default!;
                ExceptionDispatchInfo? recordingFailure = null;
                try
                {
                    result = record();
                }
                catch (Exception exception)
                {
                    recordingFailure = ExceptionDispatchInfo.Capture(exception);
                }

                if (timestampQuery == 0)
                {
                    NativeMethods.ThrowIfFailed(NativeMethods.fe_queue_submit(
                        context.Handle,
                        out var submittedFence));
                    nativeFence = submittedFence;
                }
                else
                {
                    NativeMethods.ThrowIfFailed(NativeMethods.fe_queue_submit_timestamped(
                        context.Handle,
                        timestampQuery,
                        out var submittedFence));
                    nativeFence = submittedFence;
                }

                context.TransferSubmittedWorkTo(leases);
                managedFence = new GpuFence(
                    nativeFence ?? throw new InvalidOperationException("Native timestamped submission returned no fence."),
                    leases);
                nativeFence = null;
                if (recordingFailure is not null)
                {
                    ReapFailedTimestampedSubmission(managedFence);
                    managedFence = null;
                    failureSubmissionReaping = true;
                    recordingFailure.Throw();
                }

                return new GpuTimestampedSubmission<T>(
                    result,
                    managedFence ?? throw new InvalidOperationException("Timestamped submission returned no fence."));
            }
            catch when (failureSubmissionReaping)
            {
                throw;
            }
            catch
            {
                try
                {
                    NativeMethods.ThrowIfFailed(NativeMethods.fe_context_wait_idle(context.Handle));
                    context.CompleteSubmittedWork();
                }
                finally
                {
                    managedFence?.Dispose();
                    nativeFence?.Dispose();
                    DisposeLeases(leases);
                }
                throw;
            }
        }
    }

    private GpuCommandList.IRecordedGpuCommand[] Snapshot(GpuCommandList commandList, string parameterName)
    {
        ArgumentNullException.ThrowIfNull(commandList, parameterName);
        if (!ReferenceEquals(commandList.Context, context))
        {
            throw new ArgumentException("The command list was created by a different GPU queue.", parameterName);
        }
        return commandList.SnapshotForSubmission();
    }

    private GpuFence SubmitSnapshots(IReadOnlyList<GpuCommandList.IRecordedGpuCommand[]> snapshots)
    {
        using var operation = context.EnterOperation();
        lock (context.QueueGate)
        {
            var leases = new List<IDisposable>();
            FeFenceHandle? nativeFence = null;
            try
            {
                foreach (var commands in snapshots)
                {
                    foreach (var command in commands)
                    {
                        command.Execute(context, leases);
                    }
                }

                NativeMethods.ThrowIfFailed(NativeMethods.fe_queue_submit(context.Handle, out var submittedFence));
                nativeFence = submittedFence;
                context.TransferSubmittedWorkTo(leases);
                return new GpuFence(submittedFence, leases);
            }
            catch
            {
                try
                {
                    NativeMethods.ThrowIfFailed(NativeMethods.fe_context_wait_idle(context.Handle));
                    context.CompleteSubmittedWork();
                }
                finally
                {
                    nativeFence?.Dispose();
                    DisposeLeases(leases);
                }
                throw;
            }
        }
    }

    public void WaitIdle() => context.WaitIdle();

    private static void ReapFailedTimestampedSubmission(GpuFence fence)
    {
        _ = Task.Run(async () =>
        {
            try
            {
                await fence.DisposeAsync().ConfigureAwait(false);
            }
            catch (Exception exception)
            {
                Console.Error.WriteLine("[feather] failed timestamped submission retirement: " + exception.Message);
            }
        });
    }

    private static void DisposeLeases(List<IDisposable> leases)
    {
        foreach (var lease in leases)
        {
            lease.Dispose();
        }
        leases.Clear();
    }
}

/// <summary>Result and queue fence for one aggregate timestamped submission.</summary>
public readonly record struct GpuTimestampedSubmission<T>(T Result, GpuFence Fence);

/// <summary>
/// Represents completion of one queue submission.
/// </summary>
public sealed class GpuFence : IDisposable, IAsyncDisposable
{
    private readonly object gate = new();
    private readonly FeFenceHandle handle;
    private List<IDisposable>? leases;
    private Task? asyncCompletionTask;
    private DisposeAttempt? disposeAttempt;
    private int completionState;
    private int disposeState;

    internal GpuFence(FeFenceHandle handle, List<IDisposable> leases)
    {
        this.handle = handle;
        this.leases = leases;
    }

    ~GpuFence()
    {
        _ = ThreadPool.QueueUserWorkItem(
            static state => ((GpuFence)state!).StartAbandonedReaper(),
            this);
    }

    /// <summary>
    /// Gets whether this fence has released its native submission marker.
    /// </summary>
    public bool IsDisposed
    {
        get
        {
            lock (gate)
            {
                return disposeState == 2;
            }
        }
    }

    public bool IsCompleted
    {
        get
        {
            lock (gate)
            {
                if (completionState == 2)
                {
                    return true;
                }
                if (completionState == 1)
                {
                    return false;
                }
                completionState = 1;
            }

            try
            {
                NativeMethods.ThrowIfFailed(NativeMethods.fe_fence_is_complete(handle, out var completed));
                if (completed)
                {
                    MarkCompleted();
                }
                else
                {
                    ResetWaiter();
                }
                return completed;
            }
            catch
            {
                ResetWaiter();
                throw;
            }
        }
    }

    /// <summary>
    /// Attempts to resolve the real GPU timestamp interval owned by this submission without
    /// waiting. Returns false when pending, unsupported, or invalid; it never substitutes a
    /// CPU duration or fence latency.
    /// </summary>
    public bool TryGetGpuElapsedNanoseconds(out ulong elapsedNanoseconds)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposeState == 2, this);
        }
        NativeMethods.ThrowIfFailed(NativeMethods.fe_fence_try_get_timestamp(
            handle,
            out var available,
            out elapsedNanoseconds));
        if (available)
        {
            MarkCompleted();
        }
        return available;
    }

    public void Wait() => _ = Wait(Timeout.InfiniteTimeSpan);

    /// <summary>
    /// Waits up to <paramref name="timeout"/> and returns whether the submission completed.
    /// </summary>
    public bool Wait(TimeSpan timeout)
    {
        ValidateTimeout(timeout);
        var stopwatch = timeout == Timeout.InfiniteTimeSpan ? null : Stopwatch.StartNew();

        while (true)
        {
            TimeSpan remaining;
            lock (gate)
            {
                if (completionState == 2)
                {
                    return true;
                }

                remaining = GetRemaining(timeout, stopwatch);
                if (completionState == 1)
                {
                    if (remaining == TimeSpan.Zero || !Monitor.Wait(gate, remaining))
                    {
                        return completionState == 2;
                    }
                    continue;
                }
                completionState = 1;
            }

            try
            {
                NativeMethods.ThrowIfFailed(NativeMethods.fe_fence_wait(
                    handle,
                    ToNanoseconds(remaining),
                    out var completed));
                if (completed)
                {
                    MarkCompleted();
                    return true;
                }
                ResetWaiter();
                return false;
            }
            catch
            {
                ResetWaiter();
                throw;
            }
        }
    }

    public async ValueTask WaitAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var completion = GetOrCreateAsyncCompletion();
        try
        {
            await completion.WaitAsync(cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            ClearFaultedAsyncCompletion(completion);
            throw;
        }
    }

    /// <summary>
    /// Asynchronously waits up to <paramref name="timeout"/> without blocking a worker thread.
    /// </summary>
    public async ValueTask<bool> WaitAsync(TimeSpan timeout, CancellationToken cancellationToken = default)
    {
        ValidateTimeout(timeout);
        cancellationToken.ThrowIfCancellationRequested();
        if (timeout == Timeout.InfiniteTimeSpan)
        {
            await WaitAsync(cancellationToken).ConfigureAwait(false);
            return true;
        }
        if (timeout == TimeSpan.Zero)
        {
            return IsCompleted;
        }

        var completion = GetOrCreateAsyncCompletion();
        try
        {
            await completion.WaitAsync(timeout, cancellationToken).ConfigureAwait(false);
            return true;
        }
        catch (TimeoutException)
        {
            return IsCompleted;
        }
        catch
        {
            ClearFaultedAsyncCompletion(completion);
            throw;
        }
    }

    private Task GetOrCreateAsyncCompletion()
    {
        lock (gate)
        {
            if (completionState == 2)
            {
                return Task.CompletedTask;
            }
            return asyncCompletionTask ??= ObserveCompletionAsync();
        }
    }

    private async Task ObserveCompletionAsync()
    {
        await Task.Yield();
        var delayMilliseconds = 1;
        while (!IsCompleted)
        {
            await Task.Delay(delayMilliseconds).ConfigureAwait(false);
            delayMilliseconds = System.Math.Min(delayMilliseconds * 2, 8);
        }
    }

    private void ClearFaultedAsyncCompletion(Task completion)
    {
        if (!completion.IsFaulted)
        {
            return;
        }
        lock (gate)
        {
            if (ReferenceEquals(asyncCompletionTask, completion))
            {
                asyncCompletionTask = null;
            }
        }
    }

    public void Dispose()
    {
        if (!TryBeginDispose(out var attempt))
        {
            attempt.Wait();
            return;
        }

        try
        {
            Wait();
            ReleaseNativeFence();
            CompleteDispose(attempt, null);
        }
        catch (Exception exception)
        {
            CompleteDispose(attempt, exception);
            throw;
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (!TryBeginDispose(out var attempt))
        {
            await attempt.WaitAsync().ConfigureAwait(false);
            return;
        }

        try
        {
            await WaitAsync().ConfigureAwait(false);
            ReleaseNativeFence();
            CompleteDispose(attempt, null);
        }
        catch (Exception exception)
        {
            CompleteDispose(attempt, exception);
            throw;
        }
    }

    private bool TryBeginDispose(out DisposeAttempt attempt)
    {
        lock (gate)
        {
            if (disposeState == 2)
            {
                attempt = DisposeAttempt.Completed;
                return false;
            }
            if (disposeState == 1)
            {
                attempt = disposeAttempt!;
                return false;
            }

            disposeState = 1;
            attempt = new DisposeAttempt();
            disposeAttempt = attempt;
            return true;
        }
    }

    private void CompleteDispose(DisposeAttempt attempt, Exception? exception)
    {
        lock (gate)
        {
            disposeState = exception is null ? 2 : 0;
            if (ReferenceEquals(disposeAttempt, attempt))
            {
                disposeAttempt = null;
            }
        }
        if (exception is null)
        {
            GC.SuppressFinalize(this);
        }
        attempt.Complete(exception);
    }

    private void ReleaseNativeFence()
    {
        NativeMethods.ThrowIfFailed(handle.ReleaseSubmission());
        handle.Dispose();
    }

    private void StartAbandonedReaper()
    {
        _ = ReapAbandonedAsync();
    }

    private async Task ReapAbandonedAsync()
    {
        var retryDelay = TimeSpan.FromMilliseconds(10);
        while (!NativeMethods.IsProcessExiting)
        {
            try
            {
                await DisposeAsync().ConfigureAwait(false);
                return;
            }
            catch when (!NativeMethods.IsProcessExiting)
            {
                await Task.Delay(retryDelay).ConfigureAwait(false);
                if (retryDelay < TimeSpan.FromSeconds(1))
                {
                    retryDelay = TimeSpan.FromMilliseconds(System.Math.Min(retryDelay.TotalMilliseconds * 2, 1000));
                }
            }
            catch
            {
                break;
            }
        }

        MarkCompleted();
        handle.Dispose();
        lock (gate)
        {
            disposeState = 2;
            disposeAttempt = null;
        }
    }

    private void MarkCompleted()
    {
        List<IDisposable>? completedLeases;
        lock (gate)
        {
            if (completionState == 2)
            {
                return;
            }
            completionState = 2;
            completedLeases = leases;
            leases = null;
            Monitor.PulseAll(gate);
        }

        if (completedLeases is not null)
        {
            foreach (var lease in completedLeases)
            {
                lease.Dispose();
            }
        }
    }

    private void ResetWaiter()
    {
        lock (gate)
        {
            if (completionState == 1)
            {
                completionState = 0;
                Monitor.PulseAll(gate);
            }
        }
    }

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
        if (timeout == Timeout.InfiniteTimeSpan)
        {
            return ulong.MaxValue;
        }
        var ticks = (ulong)timeout.Ticks;
        return ticks >= ulong.MaxValue / 100UL ? ulong.MaxValue - 1 : ticks * 100UL;
    }

    private sealed class DisposeAttempt
    {
        private readonly TaskCompletionSource<bool> completion = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private ExceptionDispatchInfo? error;

        internal static DisposeAttempt Completed { get; } = CreateCompleted();

        internal void Complete(Exception? exception)
        {
            if (exception is not null)
            {
                error = ExceptionDispatchInfo.Capture(exception);
            }
            completion.TrySetResult(true);
        }

        internal void Wait()
        {
            completion.Task.GetAwaiter().GetResult();
            error?.Throw();
        }

        internal async ValueTask WaitAsync()
        {
            await completion.Task.ConfigureAwait(false);
            error?.Throw();
        }

        private static DisposeAttempt CreateCompleted()
        {
            var attempt = new DisposeAttempt();
            attempt.Complete(null);
            return attempt;
        }
    }
}
