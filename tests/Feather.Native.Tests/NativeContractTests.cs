using System.Runtime.InteropServices;
using Feather.Native;

namespace Feather.Native.Tests;

public class NativeContractTests
{
    [Fact]
    public void NativeRuntimeCanLoadContractExport()
    {
        Assert.Equal(1u, NativeMethods.fe_ir_bridge_contract_version());
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
        Assert.Equal(128, Marshal.SizeOf<FeGraphicsDrawDesc>());
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
}
