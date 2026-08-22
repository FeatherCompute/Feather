using System.Runtime.InteropServices;
using Feather.Resources;

namespace Feather.Integration.Tests;

[CollectionDefinition(AsyncTextureReadbackCollection.CollectionName, DisableParallelization = true)]
public sealed class AsyncTextureReadbackCollection
{
    public const string CollectionName = "Async texture readback GPU";
}

[Collection(AsyncTextureReadbackCollection.CollectionName)]
public class AsyncTextureReadbackTests
{
    [Fact]
    public async Task OddSizedReadbackPreservesPixelsAfterResourcesAreDisposed()
    {
        using var context = GpuContext.GetDefault();
        var texture = GpuTexture2D<uint, uint>.Create(context, 5, 3, PixelFormat.Rgba8, TextureAccess.ReadWrite);
        IReadbackGpuTexture2D inspectionTexture = texture;
        var staging = GpuBuffer<byte>.Create(context, 72, BufferAccess.ReadWrite);
        var pixels = Enumerable.Range(0, 15)
            .Select(index => 0xff000000u | ((uint)(index * 17) << 16) | ((uint)(index * 11) << 8) | (uint)(index * 5))
            .ToArray();
        var expected = MemoryMarshal.AsBytes(pixels.AsSpan()).ToArray();
        texture.Upload(pixels);

        Assert.Equal(1, inspectionTexture.MipLevels);
        Assert.Equal(PixelFormat.Rgba8, inspectionTexture.Format);
        using var readback = inspectionTexture.BeginReadback(
            staging,
            0,
            0,
            5,
            3,
            stagingByteOffset: 12);
        texture.Dispose();
        staging.Dispose();

        Assert.True(await readback.WaitAsync(TimeSpan.FromSeconds(30)));
        var mapping = readback.Map();
        try
        {
            Assert.Equal(60, mapping.ByteLength);
            Assert.Equal(20, mapping.RowPitch);
            var actual = new byte[mapping.ByteLength];
            mapping.CopyTo(actual);
            Assert.Equal(expected, actual);
        }
        finally
        {
            mapping.Dispose();
        }

        Assert.Equal(ReadbackOperationState.Consumed, readback.State);
    }

    [Fact]
    public void ManagedValidationRejectsInvalidRequestsBeforeNativeSubmission()
    {
        using var context = GpuContext.GetDefault();
        using var otherContext = GpuContext.GetDefault();
        using var texture = GpuTexture2D<uint, uint>.Create(context, 5, 3, PixelFormat.Rgba8, TextureAccess.ReadWrite);
        using var depth = GpuTexture2D<float, float>.Create(context, 1, 1, PixelFormat.Depth32Float, TextureAccess.DepthStencil);
        using var staging = GpuBuffer<byte>.Create(context, 72, BufferAccess.ReadWrite);
        using var otherStaging = GpuBuffer<byte>.Create(otherContext, 72, BufferAccess.ReadWrite);
        var before = context.OperationCounters;

        Assert.Throws<ArgumentOutOfRangeException>(() => texture.BeginReadback(staging, -1, 0, 1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => texture.BeginReadback(staging, 0, -1, 1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => texture.BeginReadback(staging, 0, 0, 0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => texture.BeginReadback(staging, 4, 0, 2, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => texture.BeginReadback(staging, 0, 0, 5, 3, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => texture.BeginReadback(staging, 0, 0, 5, 3, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => texture.BeginReadback(staging, 0, 0, 5, 3, 16));
        Assert.Throws<ArgumentException>(() => texture.BeginReadback(otherStaging, 0, 0, 1, 1));
        Assert.Throws<NotSupportedException>(() => depth.BeginReadback(staging, 0, 0, 1, 1));

        var after = context.OperationCounters;
        Assert.Equal(before.AsyncTextureReadbackCalls, after.AsyncTextureReadbackCalls);
    }

    [Fact]
    public async Task MappingRequiresObservedCompletionAndWaitCancellationDoesNotConsumeOperation()
    {
        using var context = GpuContext.GetDefault();
        using var texture = GpuTexture2D<uint, uint>.Create(context, 1, 1, PixelFormat.Rgba8, TextureAccess.ReadWrite);
        using var staging = GpuBuffer<byte>.Create(context, 4, BufferAccess.ReadWrite);
        texture.Upload([0x44332211u]);
        using var readback = texture.BeginReadback(staging, 0, 0, 1, 1);

        Assert.Throws<InvalidOperationException>(() => MapAndDispose(readback));
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            async () => await readback.WaitAsync(cancellation.Token));

        readback.Wait();
        var mapping = readback.Map();
        var bytes = new byte[4];
        mapping.CopyTo(bytes);
        var copiedLease = mapping;
        mapping.Dispose();
        copiedLease.Dispose();
        Assert.Equal([0x11, 0x22, 0x33, 0x44], bytes);
        try
        {
            CopyDisposedMapping(copiedLease);
            Assert.Fail("A disposed readback mapping must not permit another copy.");
        }
        catch (ObjectDisposedException)
        {
        }
    }

    [Fact]
    public void ThreeIndependentSlotsCompleteWhileAnEarlierConsumerIsSlow()
    {
        using var context = GpuContext.GetDefault();
        using var texture = GpuTexture2D<uint, uint>.Create(context, 2, 2, PixelFormat.Rgba8, TextureAccess.ReadWrite);
        var pixels = new[] { 0x04030201u, 0x08070605u, 0x0c0b0a09u, 0x100f0e0du };
        var expected = MemoryMarshal.AsBytes(pixels.AsSpan()).ToArray();
        texture.Upload(pixels);

        var staging = Enumerable.Range(0, 3)
            .Select(_ => GpuBuffer<byte>.Create(context, expected.Length, BufferAccess.ReadWrite))
            .ToArray();
        var operations = staging
            .Select(slot => texture.BeginReadback(slot, 0, 0, 2, 2))
            .ToArray();
        try
        {
            operations[^1].Wait();
            for (var index = operations.Length - 1; index >= 0; index--)
            {
                Assert.True(operations[index].IsCompleted);
                var mapping = operations[index].Map();
                var actual = new byte[expected.Length];
                mapping.CopyTo(actual);
                mapping.Dispose();
                Assert.Equal(expected, actual);
            }
        }
        finally
        {
            foreach (var operation in operations)
            {
                operation.Dispose();
            }
            foreach (var slot in staging)
            {
                slot.Dispose();
            }
        }
    }

    [Fact]
    public void StagingSlotCanBeConsumedAndReusedFiveHundredTimes()
    {
        using var context = GpuContext.GetDefault();
        using var texture = GpuTexture2D<uint, uint>.Create(context, 1, 1, PixelFormat.Rgba8, TextureAccess.ReadWrite);
        using var staging = GpuBuffer<byte>.Create(context, 4, BufferAccess.ReadWrite);
        texture.Upload([0x78563412u]);
        var actual = new byte[4];

        for (var iteration = 0; iteration < 500; iteration++)
        {
            using var readback = texture.BeginReadback(staging, 0, 0, 1, 1);
            readback.Wait();
            var mapping = readback.Map();
            mapping.CopyTo(actual);
            mapping.Dispose();
            Assert.Equal([0x12, 0x34, 0x56, 0x78], actual);
        }
    }

    [Fact]
    public void ColdReadbackAndPendingDisposeDoNotEnterBlockingBackendPaths()
    {
        using var context = GpuContext.GetDefault();
        using var texture = GpuTexture2D<uint, uint>.Create(context, 4, 4, PixelFormat.Rgba8, TextureAccess.RenderTarget);
        using var staging = GpuBuffer<byte>.Create(context, 64, BufferAccess.ReadWrite);
        var before = context.OperationCounters;

        var readback = texture.BeginReadback(staging, 0, 0, 4, 4);
        var afterBegin = context.OperationCounters;
        Assert.Equal(before.AsyncTextureReadbackCalls + 1, afterBegin.AsyncTextureReadbackCalls);
        Assert.Equal(before.FinishCalls, afterBegin.FinishCalls);
        Assert.Equal(before.DeviceWaitIdleCalls, afterBegin.DeviceWaitIdleCalls);
        Assert.Equal(before.GlobalDrainCalls, afterBegin.GlobalDrainCalls);
        Assert.Equal(before.BlockingSubmissionWaitCalls, afterBegin.BlockingSubmissionWaitCalls);
        Assert.Equal(before.BlockingTextureDownloadCalls, afterBegin.BlockingTextureDownloadCalls);

        readback.Dispose();
        Assert.Equal(ReadbackOperationState.Cancelled, readback.State);
        var afterDispose = context.OperationCounters;
        Assert.Equal(afterBegin.GlobalDrainCalls, afterDispose.GlobalDrainCalls);
        Assert.Equal(afterBegin.BlockingSubmissionWaitCalls, afterDispose.BlockingSubmissionWaitCalls);

        context.WaitIdle();
        using var reused = texture.BeginReadback(staging, 0, 0, 4, 4);
        reused.Wait();
        var mapping = reused.Map();
        mapping.Dispose();
    }

    [Fact]
    public void ContextShutdownCancelsPendingReadbackAndReleasesItsLeases()
    {
        var context = GpuContext.GetDefault();
        var texture = GpuTexture2D<uint, uint>.Create(context, 4, 4, PixelFormat.Rgba8, TextureAccess.RenderTarget);
        var staging = GpuBuffer<byte>.Create(context, 64, BufferAccess.ReadWrite);
        var readback = texture.BeginReadback(staging, 0, 0, 4, 4);

        context.Dispose();

        Assert.Equal(ReadbackOperationState.Cancelled, readback.State);
        readback.Dispose();
        staging.Dispose();
        texture.Dispose();
    }

    private static void MapAndDispose(ReadbackOperation operation)
    {
        var mapping = operation.Map();
        mapping.Dispose();
    }

    private static void CopyDisposedMapping(ReadbackMapping mapping)
    {
        Span<byte> destination = stackalloc byte[mapping.ByteLength];
        mapping.CopyTo(destination);
    }
}
