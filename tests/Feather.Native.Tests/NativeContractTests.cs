using System.Runtime.InteropServices;
using Feather.Native;

namespace Feather.Native.Tests;

public class NativeContractTests
{
    [Fact]
    public void NativeRuntimeCanLoadContractExport()
    {
        Assert.Equal(1u, NativeMethods.fe_ir_bridge_contract_version());
        Assert.Equal(1u, NativeMethods.fe_runtime_abi_version());
    }

    [Fact]
    public void NativeRuntimeCanFlushPersistentCaches()
    {
        Assert.True(NativeMethods.fe_runtime_flush_caches().Succeeded());
    }

    [Fact]
    public void NativeBufferRangesRejectUnsignedWraparound()
    {
        NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_default(out var context));
        using (context)
        {
            var desc = new FeBufferDesc(16, mode: 2, elementStride: 4);
            NativeMethods.ThrowIfFailed(NativeMethods.fe_buffer_create(context, in desc, IntPtr.Zero, out var buffer));
            using (buffer)
            {
                var data = Marshal.AllocHGlobal(4);
                try
                {
                    Assert.Equal(
                        FeResult.ErrorInvalidArgument,
                        NativeMethods.fe_buffer_upload(buffer, ulong.MaxValue, 2, data));
                    Assert.Equal(
                        FeResult.ErrorInvalidArgument,
                        NativeMethods.fe_buffer_download(buffer, ulong.MaxValue, 2, data));
                }
                finally
                {
                    Marshal.FreeHGlobal(data);
                }
            }
        }
    }

    [Fact]
    public void NativeSubmissionFenceTracksQueuedBufferCopy()
    {
        NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_default(out var context));
        using (context)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_context_initialize(context));
            var desc = new FeBufferDesc(16, mode: 2, elementStride: 4);
            var sourceData = new[] { 3, 5, 8, 13 };
            var sourcePointer = Marshal.AllocHGlobal(16);
            var resultPointer = Marshal.AllocHGlobal(16);
            try
            {
                Marshal.Copy(sourceData, 0, sourcePointer, sourceData.Length);
                NativeMethods.ThrowIfFailed(NativeMethods.fe_buffer_create(context, in desc, sourcePointer, out var source));
                NativeMethods.ThrowIfFailed(NativeMethods.fe_buffer_create(context, in desc, IntPtr.Zero, out var destination));
                using (source)
                using (destination)
                {
                    NativeMethods.ThrowIfFailed(NativeMethods.fe_buffer_copy(source, 0, destination, 0, 16));
                    NativeMethods.ThrowIfFailed(NativeMethods.fe_queue_memory_barrier(context, 1u));
                    NativeMethods.ThrowIfFailed(NativeMethods.fe_queue_submit(context, out var fence));
                    using (fence)
                    {
                        NativeMethods.ThrowIfFailed(NativeMethods.fe_fence_wait(fence, 0, out _));
                        NativeMethods.ThrowIfFailed(NativeMethods.fe_fence_wait(fence, ulong.MaxValue, out var completed));
                        Assert.True(completed);
                        NativeMethods.ThrowIfFailed(NativeMethods.fe_fence_is_complete(fence, out completed));
                        Assert.True(completed);
                    }

                    NativeMethods.ThrowIfFailed(NativeMethods.fe_buffer_download(destination, 0, 16, resultPointer));
                    var result = new int[4];
                    Marshal.Copy(resultPointer, result, 0, result.Length);
                    Assert.Equal(sourceData, result);
                }
            }
            finally
            {
                Marshal.FreeHGlobal(resultPointer);
                Marshal.FreeHGlobal(sourcePointer);
            }
        }
    }

    [Fact]
    public async Task NativeFenceWaitAndDestroyCanRaceSafely()
    {
        NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_default(out var context));
        using (context)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_context_initialize(context));
            for (var iteration = 0; iteration < 32; ++iteration)
            {
                NativeMethods.ThrowIfFailed(NativeMethods.fe_queue_submit(context, out var fence));
                using (fence)
                {
                    var rawFence = fence.DangerousGetHandle();
                    var waitTask = Task.Run(() =>
                    {
                        var result = NativeMethods.fe_fence_wait(fence, ulong.MaxValue, out var completed);
                        return (result, completed);
                    });
                    var destroyTask = Task.Run(() => NativeMethods.fe_fence_destroy_raw(rawFence));

                    var wait = await waitTask.WaitAsync(TimeSpan.FromSeconds(30));
                    var destroy = await destroyTask.WaitAsync(TimeSpan.FromSeconds(30));
                    Assert.Equal(FeResult.Ok, destroy);
                    fence.SetHandleAsInvalid();
                    Assert.True(
                        wait.result is FeResult.Ok or FeResult.ErrorInvalidHandle,
                        $"Unexpected wait result while racing fence destruction: {wait.result}");
                    if (wait.result == FeResult.Ok)
                    {
                        Assert.True(wait.completed);
                    }
                }
            }
        }
    }

    [Fact]
    public void NativeQueueRejectsUnknownBarrierFlags()
    {
        NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_default(out var context));
        using (context)
        {
            Assert.Equal(
                FeResult.ErrorInvalidArgument,
                NativeMethods.fe_queue_memory_barrier(context, uint.MaxValue));
        }
    }

    [Fact]
    public void NativeTextureRangesRejectUnsignedWraparound()
    {
        NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_default(out var context));
        using (context)
        {
            var texture2DDesc = new FeTexture2DDesc(1, 1, 1, pixelFormat: 3, access: 2);
            NativeMethods.ThrowIfFailed(NativeMethods.fe_texture2d_create(
                context,
                in texture2DDesc,
                IntPtr.Zero,
                out var texture2D));
            var texture3DDesc = new FeTexture3DDesc(1, 1, 1, 1, pixelFormat: 3, access: 2);
            NativeMethods.ThrowIfFailed(NativeMethods.fe_texture3d_create(
                context,
                in texture3DDesc,
                IntPtr.Zero,
                out var texture3D));

            using (texture2D)
            using (texture3D)
            {
                var data = Marshal.AllocHGlobal(8);
                try
                {
                    Assert.Equal(
                        FeResult.ErrorInvalidArgument,
                        NativeMethods.fe_texture2d_upload(texture2D, uint.MaxValue, 0, 2, 1, data));
                    Assert.Equal(
                        FeResult.ErrorInvalidArgument,
                        NativeMethods.fe_texture2d_download(texture2D, 0, uint.MaxValue, 1, 2, data));
                    Assert.Equal(
                        FeResult.ErrorInvalidArgument,
                        NativeMethods.fe_texture3d_upload(texture3D, 0, 0, uint.MaxValue, 1, 1, 2, data));
                    Assert.Equal(
                        FeResult.ErrorInvalidArgument,
                        NativeMethods.fe_texture3d_download(texture3D, uint.MaxValue, 0, 0, 2, 1, 1, data));
                }
                finally
                {
                    Marshal.FreeHGlobal(data);
                }
            }
        }
    }

    [Fact]
    public void NativeAsyncTextureReadbackMapsExactlyOnceWithStableMetadata()
    {
        NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_default(out var context));
        using (context)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_context_initialize(context));
            const int width = 5;
            const int height = 3;
            const int offset = 12;
            var source = Enumerable.Range(0, width * height * 4)
                .Select(index => (byte)((index * 37 + 11) & 0xff))
                .ToArray();
            var sourcePointer = Marshal.AllocHGlobal(source.Length);
            try
            {
                Marshal.Copy(source, 0, sourcePointer, source.Length);
                var textureDesc = new FeTexture2DDesc(width, height, 1, pixelFormat: 3, access: 3);
                var stagingDesc = new FeBufferDesc((ulong)(offset + source.Length), mode: 3, elementStride: 1);
                NativeMethods.ThrowIfFailed(NativeMethods.fe_texture2d_create(
                    context,
                    in textureDesc,
                    sourcePointer,
                    out var texture));
                NativeMethods.ThrowIfFailed(NativeMethods.fe_buffer_create(
                    context,
                    in stagingDesc,
                    IntPtr.Zero,
                    out var staging));
                using (texture)
                using (staging)
                {
                    Assert.Equal(
                        FeResult.ErrorInvalidArgument,
                        NativeMethods.fe_texture2d_begin_readback(
                            context,
                            texture,
                            staging,
                            0,
                            0,
                            0,
                            height,
                            0,
                            out _));
                    Assert.Equal(
                        FeResult.ErrorInvalidArgument,
                        NativeMethods.fe_texture2d_begin_readback(
                            context,
                            texture,
                            staging,
                            width - 1,
                            0,
                            2,
                            height,
                            0,
                            out _));
                    Assert.Equal(
                        FeResult.ErrorInvalidArgument,
                        NativeMethods.fe_texture2d_begin_readback(
                            context,
                            texture,
                            staging,
                            0,
                            0,
                            width,
                            height,
                            2,
                            out _));
                    Assert.Equal(
                        FeResult.ErrorInvalidArgument,
                        NativeMethods.fe_texture2d_begin_readback(
                            context,
                            texture,
                            staging,
                            0,
                            0,
                            width,
                            height,
                            ulong.MaxValue - 3,
                            out _));

                    var depthDesc = new FeTexture2DDesc(1, 1, 1, pixelFormat: 101, access: 6);
                    NativeMethods.ThrowIfFailed(NativeMethods.fe_texture2d_create(
                        context,
                        in depthDesc,
                        IntPtr.Zero,
                        out var depth));
                    using (depth)
                    {
                        Assert.Equal(
                            FeResult.ErrorUnsupported,
                            NativeMethods.fe_texture2d_begin_readback(
                                context,
                                depth,
                                staging,
                                0,
                                0,
                                1,
                                1,
                                0,
                                out _));
                    }

                    NativeMethods.ThrowIfFailed(NativeMethods.fe_texture2d_begin_readback(
                        context,
                        texture,
                        staging,
                        0,
                        0,
                        width,
                        height,
                        offset,
                        out var readback));
                    using (readback)
                    {
                        NativeMethods.ThrowIfFailed(NativeMethods.fe_readback_wait(
                            readback,
                            ulong.MaxValue,
                            out var completed));
                        Assert.True(completed);
                        NativeMethods.ThrowIfFailed(NativeMethods.fe_readback_map(readback, out var mapping));
                        Assert.Equal((ulong)source.Length, mapping.ByteSize);
                        Assert.Equal((ulong)(width * 4), mapping.RowPitch);
                        Assert.NotEqual(IntPtr.Zero, mapping.Data);

                        var actual = new byte[source.Length];
                        Marshal.Copy(mapping.Data, actual, 0, actual.Length);
                        Assert.Equal(source, actual);
                        Assert.Equal(
                            FeResult.ErrorInvalidArgument,
                            NativeMethods.fe_readback_map(readback, out _));

                        NativeMethods.ThrowIfFailed(NativeMethods.fe_readback_unmap(readback));
                        Assert.Equal(FeResult.ErrorInvalidArgument, NativeMethods.fe_readback_unmap(readback));
                        Assert.Equal(
                            FeResult.ErrorInvalidArgument,
                            NativeMethods.fe_readback_map(readback, out _));
                    }
                }
            }
            finally
            {
                Marshal.FreeHGlobal(sourcePointer);
            }
        }
    }

    [Fact]
    public void ColdNativeReadbackAndCancellationAvoidBlockingOperations()
    {
        NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_default(out var context));
        using (context)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_context_initialize(context));
            var textureDesc = new FeTexture2DDesc(4, 4, 1, pixelFormat: 3, access: 5);
            var stagingDesc = new FeBufferDesc(64, mode: 3, elementStride: 1);
            NativeMethods.ThrowIfFailed(NativeMethods.fe_texture2d_create(
                context,
                in textureDesc,
                IntPtr.Zero,
                out var texture));
            NativeMethods.ThrowIfFailed(NativeMethods.fe_buffer_create(
                context,
                in stagingDesc,
                IntPtr.Zero,
                out var staging));
            using (texture)
            using (staging)
            {
                NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_operation_counters(context, out var before));
                NativeMethods.ThrowIfFailed(NativeMethods.fe_texture2d_begin_readback(
                    context,
                    texture,
                    staging,
                    0,
                    0,
                    4,
                    4,
                    0,
                    out var readback));
                NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_operation_counters(context, out var afterBegin));
                Assert.Equal(before.AsyncTextureReadbackCalls + 1, afterBegin.AsyncTextureReadbackCalls);
                Assert.Equal(before.FinishCalls, afterBegin.FinishCalls);
                Assert.Equal(before.DeviceWaitIdleCalls, afterBegin.DeviceWaitIdleCalls);
                Assert.Equal(before.GlobalDrainCalls, afterBegin.GlobalDrainCalls);
                Assert.Equal(before.BlockingSubmissionWaitCalls, afterBegin.BlockingSubmissionWaitCalls);
                Assert.Equal(before.BlockingTextureDownloadCalls, afterBegin.BlockingTextureDownloadCalls);

                var rawReadback = readback.DangerousGetHandle();
                NativeMethods.ThrowIfFailed(NativeMethods.fe_readback_destroy_raw(rawReadback));
                readback.SetHandleAsInvalid();
                readback.Dispose();
                NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_operation_counters(context, out var afterDestroy));
                Assert.Equal(afterBegin.GlobalDrainCalls, afterDestroy.GlobalDrainCalls);
                Assert.Equal(afterBegin.BlockingSubmissionWaitCalls, afterDestroy.BlockingSubmissionWaitCalls);
            }
        }
    }

    [Fact]
    public void NativeRuntimeBilinearlyUpscalesRgba8()
    {
        var source = new byte[]
        {
            0, 0, 0, 255,
            255, 0, 0, 255,
            0, 255, 0, 255,
            255, 255, 255, 255
        };
        var destination = new byte[4 * 4 * 4];
        var sourceBuffer = Marshal.AllocHGlobal(source.Length);
        var destinationBuffer = Marshal.AllocHGlobal(destination.Length);
        try
        {
            Marshal.Copy(source, 0, sourceBuffer, source.Length);
            NativeMethods.ThrowIfFailed(NativeMethods.fe_bilinear_upscale_rgba8(
                sourceBuffer,
                2,
                2,
                destinationBuffer,
                4,
                4));
            Marshal.Copy(destinationBuffer, destination, 0, destination.Length);
        }
        finally
        {
            Marshal.FreeHGlobal(destinationBuffer);
            Marshal.FreeHGlobal(sourceBuffer);
        }

        // Half-texel mapping puts (1,1) at (0.25,0.25), so the white tap contributes 1/16 of 255: B=16.
        Assert.Equal(new byte[] { 64, 64, 16, 255 }, destination.Skip(20).Take(4));
    }

    [Fact]
    public void ResultValuesMatchNativeAbiSpecification()
    {
        Assert.Equal(0u, (uint)FeResult.Ok);
        Assert.Equal(3u, (uint)FeResult.ErrorInvalidHandle);
        Assert.Equal(7u, (uint)FeResult.ErrorUnsupported);
    }

    [Fact]
    public void BufferDescriptorHasStableSequentialLayout()
    {
        Assert.Equal(16, Marshal.SizeOf<FeBufferDesc>());
        Assert.Equal(0, Marshal.OffsetOf<FeBufferDesc>(nameof(FeBufferDesc.SizeInBytes)).ToInt32());
        Assert.Equal(8, Marshal.OffsetOf<FeBufferDesc>(nameof(FeBufferDesc.Mode)).ToInt32());
        Assert.Equal(12, Marshal.OffsetOf<FeBufferDesc>(nameof(FeBufferDesc.ElementStride)).ToInt32());
    }

    [Fact]
    public void TextureDescriptorHasStableSequentialLayout()
    {
        Assert.Equal(20, Marshal.SizeOf<FeTexture2DDesc>());
        Assert.Equal(0, Marshal.OffsetOf<FeTexture2DDesc>(nameof(FeTexture2DDesc.Width)).ToInt32());
        Assert.Equal(16, Marshal.OffsetOf<FeTexture2DDesc>(nameof(FeTexture2DDesc.Access)).ToInt32());
    }

    [Fact]
    public void ReadbackDescriptorsHaveStableSequentialLayout()
    {
        Assert.Equal(24, Marshal.SizeOf<FeReadbackMapping>());
        Assert.Equal(0, Marshal.OffsetOf<FeReadbackMapping>(nameof(FeReadbackMapping.Data)).ToInt32());
        Assert.Equal(8, Marshal.OffsetOf<FeReadbackMapping>(nameof(FeReadbackMapping.ByteSize)).ToInt32());
        Assert.Equal(16, Marshal.OffsetOf<FeReadbackMapping>(nameof(FeReadbackMapping.RowPitch)).ToInt32());
        Assert.Equal(48, Marshal.SizeOf<FeBackendOperationCounters>());
    }

    [Fact]
    public void Texture3DDescriptorHasStableSequentialLayout()
    {
        Assert.Equal(24, Marshal.SizeOf<FeTexture3DDesc>());
        Assert.Equal(0, Marshal.OffsetOf<FeTexture3DDesc>(nameof(FeTexture3DDesc.Width)).ToInt32());
        Assert.Equal(8, Marshal.OffsetOf<FeTexture3DDesc>(nameof(FeTexture3DDesc.Depth)).ToInt32());
        Assert.Equal(20, Marshal.OffsetOf<FeTexture3DDesc>(nameof(FeTexture3DDesc.Access)).ToInt32());
    }

    [Fact]
    public void SamplerDescriptorHasStableSequentialLayout()
    {
        Assert.Equal(56, Marshal.SizeOf<FeSamplerDesc>());
        Assert.Equal(0, Marshal.OffsetOf<FeSamplerDesc>(nameof(FeSamplerDesc.MinFilter)).ToInt32());
        Assert.Equal(4, Marshal.OffsetOf<FeSamplerDesc>(nameof(FeSamplerDesc.MagFilter)).ToInt32());
        Assert.Equal(8, Marshal.OffsetOf<FeSamplerDesc>(nameof(FeSamplerDesc.MipmapMode)).ToInt32());
        Assert.Equal(12, Marshal.OffsetOf<FeSamplerDesc>(nameof(FeSamplerDesc.AddressU)).ToInt32());
        Assert.Equal(16, Marshal.OffsetOf<FeSamplerDesc>(nameof(FeSamplerDesc.AddressV)).ToInt32());
        Assert.Equal(20, Marshal.OffsetOf<FeSamplerDesc>(nameof(FeSamplerDesc.AddressW)).ToInt32());
        Assert.Equal(24, Marshal.OffsetOf<FeSamplerDesc>(nameof(FeSamplerDesc.MipLodBias)).ToInt32());
        Assert.Equal(28, Marshal.OffsetOf<FeSamplerDesc>(nameof(FeSamplerDesc.MinLod)).ToInt32());
        Assert.Equal(32, Marshal.OffsetOf<FeSamplerDesc>(nameof(FeSamplerDesc.MaxLod)).ToInt32());
        Assert.Equal(36, Marshal.OffsetOf<FeSamplerDesc>(nameof(FeSamplerDesc.AnisotropyEnable)).ToInt32());
        Assert.Equal(40, Marshal.OffsetOf<FeSamplerDesc>(nameof(FeSamplerDesc.MaxAnisotropy)).ToInt32());
        Assert.Equal(44, Marshal.OffsetOf<FeSamplerDesc>(nameof(FeSamplerDesc.CompareEnable)).ToInt32());
        Assert.Equal(48, Marshal.OffsetOf<FeSamplerDesc>(nameof(FeSamplerDesc.CompareOp)).ToInt32());
        Assert.Equal(52, Marshal.OffsetOf<FeSamplerDesc>(nameof(FeSamplerDesc.BorderColor)).ToInt32());
    }

    [Fact]
    public void GraphicsStencilFaceDescriptorHasStableSequentialLayout()
    {
        Assert.Equal(16, Marshal.SizeOf<FeGraphicsStencilFaceDesc>());
        Assert.Equal(0, Marshal.OffsetOf<FeGraphicsStencilFaceDesc>(nameof(FeGraphicsStencilFaceDesc.FailOp)).ToInt32());
        Assert.Equal(4, Marshal.OffsetOf<FeGraphicsStencilFaceDesc>(nameof(FeGraphicsStencilFaceDesc.PassOp)).ToInt32());
        Assert.Equal(8, Marshal.OffsetOf<FeGraphicsStencilFaceDesc>(nameof(FeGraphicsStencilFaceDesc.DepthFailOp)).ToInt32());
        Assert.Equal(12, Marshal.OffsetOf<FeGraphicsStencilFaceDesc>(nameof(FeGraphicsStencilFaceDesc.CompareOp)).ToInt32());
    }

    [Fact]
    public void GraphicsColorBlendAttachmentDescriptorHasStableSequentialLayout()
    {
        Assert.Equal(32, Marshal.SizeOf<FeGraphicsColorBlendAttachmentDesc>());
        Assert.Equal(0, Marshal.OffsetOf<FeGraphicsColorBlendAttachmentDesc>(nameof(FeGraphicsColorBlendAttachmentDesc.BlendEnable)).ToInt32());
        Assert.Equal(4, Marshal.OffsetOf<FeGraphicsColorBlendAttachmentDesc>(nameof(FeGraphicsColorBlendAttachmentDesc.SrcColor)).ToInt32());
        Assert.Equal(8, Marshal.OffsetOf<FeGraphicsColorBlendAttachmentDesc>(nameof(FeGraphicsColorBlendAttachmentDesc.DstColor)).ToInt32());
        Assert.Equal(12, Marshal.OffsetOf<FeGraphicsColorBlendAttachmentDesc>(nameof(FeGraphicsColorBlendAttachmentDesc.ColorOp)).ToInt32());
        Assert.Equal(16, Marshal.OffsetOf<FeGraphicsColorBlendAttachmentDesc>(nameof(FeGraphicsColorBlendAttachmentDesc.SrcAlpha)).ToInt32());
        Assert.Equal(20, Marshal.OffsetOf<FeGraphicsColorBlendAttachmentDesc>(nameof(FeGraphicsColorBlendAttachmentDesc.DstAlpha)).ToInt32());
        Assert.Equal(24, Marshal.OffsetOf<FeGraphicsColorBlendAttachmentDesc>(nameof(FeGraphicsColorBlendAttachmentDesc.AlphaOp)).ToInt32());
        Assert.Equal(28, Marshal.OffsetOf<FeGraphicsColorBlendAttachmentDesc>(nameof(FeGraphicsColorBlendAttachmentDesc.WriteMask)).ToInt32());
    }

    [Fact]
    public void GraphicsPipelineCreateDescriptorHasStableSequentialLayout()
    {
        Assert.Equal(440, Marshal.SizeOf<FeGraphicsPipelineCreateDesc>());
        Assert.Equal(0, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.IrData)).ToInt32());
        Assert.Equal(8, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.IrSize)).ToInt32());
        Assert.Equal(16, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.VertexIrData)).ToInt32());
        Assert.Equal(24, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.VertexIrSize)).ToInt32());
        Assert.Equal(32, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.FragmentIrData)).ToInt32());
        Assert.Equal(40, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.FragmentIrSize)).ToInt32());
        Assert.Equal(48, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.Topology)).ToInt32());
        Assert.Equal(52, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.SampleCount)).ToInt32());
        Assert.Equal(56, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.ColorAttachmentCount)).ToInt32());
        Assert.Equal(60, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.DepthTest)).ToInt32());
        Assert.Equal(64, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.DepthWrite)).ToInt32());
        Assert.Equal(68, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.DepthCompare)).ToInt32());
        Assert.Equal(72, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.StencilTest)).ToInt32());
        Assert.Equal(76, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.StencilFront)).ToInt32());
        Assert.Equal(92, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.StencilBack)).ToInt32());
        Assert.Equal(108, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.StencilReadMask)).ToInt32());
        Assert.Equal(112, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.StencilWriteMask)).ToInt32());
        Assert.Equal(116, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.StencilReference)).ToInt32());
        Assert.Equal(120, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.BlendEnable)).ToInt32());
        Assert.Equal(124, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.BlendSrcColor)).ToInt32());
        Assert.Equal(128, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.BlendDstColor)).ToInt32());
        Assert.Equal(132, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.BlendColorOp)).ToInt32());
        Assert.Equal(136, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.BlendSrcAlpha)).ToInt32());
        Assert.Equal(140, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.BlendDstAlpha)).ToInt32());
        Assert.Equal(144, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.BlendAlphaOp)).ToInt32());
        Assert.Equal(148, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.BlendWriteMask)).ToInt32());
        Assert.Equal(152, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.ColorBlendAttachmentCount)).ToInt32());
        Assert.Equal(156, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.ColorBlendAttachment0)).ToInt32());
        Assert.Equal(188, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.ColorBlendAttachment1)).ToInt32());
        Assert.Equal(220, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.ColorBlendAttachment2)).ToInt32());
        Assert.Equal(252, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.ColorBlendAttachment3)).ToInt32());
        Assert.Equal(284, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.ColorBlendAttachment4)).ToInt32());
        Assert.Equal(316, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.ColorBlendAttachment5)).ToInt32());
        Assert.Equal(348, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.ColorBlendAttachment6)).ToInt32());
        Assert.Equal(380, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.ColorBlendAttachment7)).ToInt32());
        Assert.Equal(412, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.CullMode)).ToInt32());
        Assert.Equal(416, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.FrontFace)).ToInt32());
        Assert.Equal(420, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.PolygonMode)).ToInt32());
        Assert.Equal(424, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.DepthClamp)).ToInt32());
        Assert.Equal(432, Marshal.OffsetOf<FeGraphicsPipelineCreateDesc>(nameof(FeGraphicsPipelineCreateDesc.DebugName)).ToInt32());
    }

    [Fact]
    public void GraphicsDrawDescriptorHasStableSequentialLayout()
    {
        Assert.Equal(144, Marshal.SizeOf<FeGraphicsDrawDesc>());
        Assert.Equal(0, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ColorTargets)).ToInt32());
        Assert.Equal(8, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ColorTargetCount)).ToInt32());
        Assert.Equal(16, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.DepthTarget)).ToInt32());
        Assert.Equal(24, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.Count)).ToInt32());
        Assert.Equal(32, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.IndexBuffer)).ToInt32());
        Assert.Equal(40, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.Indexed)).ToInt32());
        Assert.Equal(44, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.Wait)).ToInt32());
        Assert.Equal(48, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ViewportEnabled)).ToInt32());
        Assert.Equal(52, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ViewportX)).ToInt32());
        Assert.Equal(56, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ViewportY)).ToInt32());
        Assert.Equal(60, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ViewportWidth)).ToInt32());
        Assert.Equal(64, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ViewportHeight)).ToInt32());
        Assert.Equal(68, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ScissorEnabled)).ToInt32());
        Assert.Equal(72, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ScissorX)).ToInt32());
        Assert.Equal(76, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ScissorY)).ToInt32());
        Assert.Equal(80, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ScissorWidth)).ToInt32());
        Assert.Equal(84, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ScissorHeight)).ToInt32());
        Assert.Equal(88, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ClearDepth)).ToInt32());
        Assert.Equal(92, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ClearDepthValue)).ToInt32());
        Assert.Equal(96, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.DepthLoadOp)).ToInt32());
        Assert.Equal(100, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ClearColor)).ToInt32());
        Assert.Equal(104, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ClearColorR)).ToInt32());
        Assert.Equal(108, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ClearColorG)).ToInt32());
        Assert.Equal(112, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ClearColorB)).ToInt32());
        Assert.Equal(116, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ClearColorA)).ToInt32());
        Assert.Equal(120, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.ColorLoadOp)).ToInt32());
        Assert.Equal(124, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.InstanceCount)).ToInt32());
        Assert.Equal(128, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.FirstVertex)).ToInt32());
        Assert.Equal(132, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.FirstIndex)).ToInt32());
        Assert.Equal(136, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.VertexOffset)).ToInt32());
        Assert.Equal(140, Marshal.OffsetOf<FeGraphicsDrawDesc>(nameof(FeGraphicsDrawDesc.FirstInstance)).ToInt32());
    }

    [Fact]
    public void ProfilerQueryResultHasStableSequentialLayout()
    {
        Assert.Equal(40, Marshal.SizeOf<FeProfilerQueryResult>());
        Assert.Equal(0, Marshal.OffsetOf<FeProfilerQueryResult>(nameof(FeProfilerQueryResult.Count)).ToInt32());
        Assert.Equal(8, Marshal.OffsetOf<FeProfilerQueryResult>(nameof(FeProfilerQueryResult.MinTimeMs)).ToInt32());
        Assert.Equal(16, Marshal.OffsetOf<FeProfilerQueryResult>(nameof(FeProfilerQueryResult.MaxTimeMs)).ToInt32());
        Assert.Equal(24, Marshal.OffsetOf<FeProfilerQueryResult>(nameof(FeProfilerQueryResult.AverageTimeMs)).ToInt32());
        Assert.Equal(32, Marshal.OffsetOf<FeProfilerQueryResult>(nameof(FeProfilerQueryResult.TotalTimeMs)).ToInt32());
    }

    [Fact]
    public void WindowDescriptorHasStableSequentialLayout()
    {
        Assert.Equal(40, Marshal.SizeOf<FeWindowDesc>());
        Assert.Equal(0, Marshal.OffsetOf<FeWindowDesc>(nameof(FeWindowDesc.Width)).ToInt32());
        Assert.Equal(4, Marshal.OffsetOf<FeWindowDesc>(nameof(FeWindowDesc.Height)).ToInt32());
        Assert.Equal(8, Marshal.OffsetOf<FeWindowDesc>(nameof(FeWindowDesc.Title)).ToInt32());
        Assert.Equal(16, Marshal.OffsetOf<FeWindowDesc>(nameof(FeWindowDesc.Resizable)).ToInt32());
        Assert.Equal(20, Marshal.OffsetOf<FeWindowDesc>(nameof(FeWindowDesc.Visible)).ToInt32());
        Assert.Equal(24, Marshal.OffsetOf<FeWindowDesc>(nameof(FeWindowDesc.VSync)).ToInt32());
        Assert.Equal(28, Marshal.OffsetOf<FeWindowDesc>(nameof(FeWindowDesc.HighDpi)).ToInt32());
        Assert.Equal(32, Marshal.OffsetOf<FeWindowDesc>(nameof(FeWindowDesc.CenterOnCreate)).ToInt32());
    }

    [Fact]
    public void WindowEventHasStableSequentialLayout()
    {
        Assert.Equal(56, Marshal.SizeOf<FeWindowEvent>());
        Assert.Equal(0, Marshal.OffsetOf<FeWindowEvent>(nameof(FeWindowEvent.Kind)).ToInt32());
        Assert.Equal(4, Marshal.OffsetOf<FeWindowEvent>(nameof(FeWindowEvent.Key)).ToInt32());
        Assert.Equal(16, Marshal.OffsetOf<FeWindowEvent>(nameof(FeWindowEvent.Pressed)).ToInt32());
        Assert.Equal(20, Marshal.OffsetOf<FeWindowEvent>(nameof(FeWindowEvent.X)).ToInt32());
        Assert.Equal(44, Marshal.OffsetOf<FeWindowEvent>(nameof(FeWindowEvent.Width)).ToInt32());
        Assert.Equal(52, Marshal.OffsetOf<FeWindowEvent>(nameof(FeWindowEvent.Codepoint)).ToInt32());
    }

    [Fact]
    public void BackendCapsHasStableSequentialLayout()
    {
        Assert.Equal(40, Marshal.SizeOf<FeBackendCaps>());
        Assert.Equal(0, Marshal.OffsetOf<FeBackendCaps>(nameof(FeBackendCaps.BackendType)).ToInt32());
        Assert.Equal(4, Marshal.OffsetOf<FeBackendCaps>(nameof(FeBackendCaps.MaxWorkGroupSizeX)).ToInt32());
        Assert.Equal(8, Marshal.OffsetOf<FeBackendCaps>(nameof(FeBackendCaps.MaxWorkGroupSizeY)).ToInt32());
        Assert.Equal(12, Marshal.OffsetOf<FeBackendCaps>(nameof(FeBackendCaps.MaxWorkGroupSizeZ)).ToInt32());
        Assert.Equal(16, Marshal.OffsetOf<FeBackendCaps>(nameof(FeBackendCaps.SupportsGraphics)).ToInt32());
        Assert.Equal(20, Marshal.OffsetOf<FeBackendCaps>(nameof(FeBackendCaps.SupportsAD)).ToInt32());
        Assert.Equal(24, Marshal.OffsetOf<FeBackendCaps>(nameof(FeBackendCaps.SupportsNN)).ToInt32());
        Assert.Equal(28, Marshal.OffsetOf<FeBackendCaps>(nameof(FeBackendCaps.SupportsWindow)).ToInt32());
        Assert.Equal(32, Marshal.OffsetOf<FeBackendCaps>(nameof(FeBackendCaps.SupportsDepthClamp)).ToInt32());
        Assert.Equal(36, Marshal.OffsetOf<FeBackendCaps>(nameof(FeBackendCaps.SupportsNonFillPolygonMode)).ToInt32());
    }

    [Fact]
    public void BackendDeviceInfoHasStableSequentialLayout()
    {
        Assert.Equal(464, Marshal.SizeOf<FeBackendDeviceInfo>());
        Assert.Equal(0, Marshal.OffsetOf<FeBackendDeviceInfo>(nameof(FeBackendDeviceInfo.NativeAbiVersion)).ToInt32());
        Assert.Equal(4, Marshal.OffsetOf<FeBackendDeviceInfo>(nameof(FeBackendDeviceInfo.MaxTextureDimension2D)).ToInt32());
        Assert.Equal(8, Marshal.OffsetOf<FeBackendDeviceInfo>(nameof(FeBackendDeviceInfo.SupportsTimestampQueries)).ToInt32());
        Assert.Equal(16, Marshal.OffsetOf<FeBackendDeviceInfo>(nameof(FeBackendDeviceInfo.AdapterName)).ToInt32());
        Assert.Equal(272, Marshal.OffsetOf<FeBackendDeviceInfo>(nameof(FeBackendDeviceInfo.DriverVersion)).ToInt32());
        Assert.Equal(400, Marshal.OffsetOf<FeBackendDeviceInfo>(nameof(FeBackendDeviceInfo.BackendVersion)).ToInt32());
    }
}
