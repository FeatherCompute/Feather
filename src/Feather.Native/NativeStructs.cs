using System.Runtime.InteropServices;

namespace Feather.Native;

public enum FeResult : uint
{
    Ok = 0,
    ErrorUnknown = 1,
    ErrorInvalidArgument = 2,
    ErrorInvalidHandle = 3,
    ErrorBackendUnavailable = 4,
    ErrorShaderCompileFailed = 5,
    ErrorOutOfMemory = 6,
    ErrorUnsupported = 7,
    ErrorDeviceLost = 8
}

public enum FeDispatchPath : uint
{
    None = 0,
    TypedEasyGpu = 1,
    CpuReferenceFallback = 2,
    GraphicsFallback = 3,
    Rejected = 4
}

public enum FeShaderBinaryFormat : uint
{
    Unavailable = 0,
    SpirV = 1
}

public enum FeKernelDiagnosticMode : uint
{
    None = 0,
    ExecutionHeat = 1,
    LineValue = 2,
    Ubsan = 3,
    PrintAssert = 4,
    BranchDivergence = 5,
    ComputeTrace = 6
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeKernelDiagnosticLayout
{
    public readonly uint AbiVersion;
    public readonly uint Mode;
    public readonly uint BufferBinding;
    public readonly uint SiteCount;
    public readonly uint CounterStrideBytes;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeKernelDiagnosticConfigV2
{
    public FeKernelDiagnosticConfigV2(
        uint abiVersion,
        FeKernelDiagnosticMode mode,
        uint sourceSiteIndex,
        uint selectedX,
        uint selectedY,
        uint selectedZ,
        uint recordCapacity,
        uint flags)
    {
        AbiVersion = abiVersion;
        Mode = (uint)mode;
        SourceSiteIndex = sourceSiteIndex;
        SelectedX = selectedX;
        SelectedY = selectedY;
        SelectedZ = selectedZ;
        RecordCapacity = recordCapacity;
        Flags = flags;
    }

    public readonly uint AbiVersion;
    public readonly uint Mode;
    public readonly uint SourceSiteIndex;
    public readonly uint SelectedX;
    public readonly uint SelectedY;
    public readonly uint SelectedZ;
    public readonly uint RecordCapacity;
    public readonly uint Flags;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeKernelDiagnosticLayoutV2
{
    public readonly uint AbiVersion;
    public readonly uint Mode;
    public readonly uint BufferBinding;
    public readonly uint SourceSiteIndex;
    public readonly uint RecordStrideBytes;
    public readonly uint RecordCapacity;
    public readonly uint ValueTypeTag;
    public readonly uint ComponentCount;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeKernelDiagnosticConfigV3
{
    public FeKernelDiagnosticConfigV3(
        uint abiVersion,
        FeKernelDiagnosticMode mode,
        uint recordCapacity,
        uint flags,
        uint sourceSiteIndex,
        uint selectedX,
        uint selectedY,
        uint selectedZ)
    {
        AbiVersion = abiVersion;
        Mode = (uint)mode;
        RecordCapacity = recordCapacity;
        Flags = flags;
        SourceSiteIndex = sourceSiteIndex;
        SelectedX = selectedX;
        SelectedY = selectedY;
        SelectedZ = selectedZ;
    }

    public readonly uint AbiVersion;
    public readonly uint Mode;
    public readonly uint RecordCapacity;
    public readonly uint Flags;
    public readonly uint SourceSiteIndex;
    public readonly uint SelectedX;
    public readonly uint SelectedY;
    public readonly uint SelectedZ;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeKernelDiagnosticLayoutV3
{
    public readonly uint AbiVersion;
    public readonly uint Mode;
    public readonly uint BufferBinding;
    public readonly uint SiteCount;
    public readonly uint HeaderStrideBytes;
    public readonly uint RecordStrideBytes;
    public readonly uint RecordCapacity;
    public readonly uint Flags;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeKernelDiagnosticConfigV4
{
    public FeKernelDiagnosticConfigV4(
        uint abiVersion,
        FeKernelDiagnosticMode mode,
        uint recordCapacity,
        uint filterMode,
        uint selectedX,
        uint selectedY,
        uint selectedZ,
        uint flags,
        uint logicalX,
        uint logicalY,
        uint logicalZ,
        uint reserved)
    {
        AbiVersion = abiVersion;
        Mode = (uint)mode;
        RecordCapacity = recordCapacity;
        FilterMode = filterMode;
        SelectedX = selectedX;
        SelectedY = selectedY;
        SelectedZ = selectedZ;
        Flags = flags;
        LogicalX = logicalX;
        LogicalY = logicalY;
        LogicalZ = logicalZ;
        Reserved = reserved;
    }

    public readonly uint AbiVersion;
    public readonly uint Mode;
    public readonly uint RecordCapacity;
    public readonly uint FilterMode;
    public readonly uint SelectedX;
    public readonly uint SelectedY;
    public readonly uint SelectedZ;
    public readonly uint Flags;
    public readonly uint LogicalX;
    public readonly uint LogicalY;
    public readonly uint LogicalZ;
    public readonly uint Reserved;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeKernelDiagnosticLayoutV4
{
    public readonly uint AbiVersion;
    public readonly uint Mode;
    public readonly uint BufferBinding;
    public readonly uint SiteCount;
    public readonly uint HeaderStrideBytes;
    public readonly uint RecordStrideBytes;
    public readonly uint RecordCapacity;
    public readonly uint FilterMode;
    public readonly uint MaskHeaderStrideBytes;
    public readonly uint MaskCellStrideBytes;
    public readonly uint LogicalX;
    public readonly uint LogicalY;
    public readonly uint LogicalZ;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeKernelDiagnosticConfigV5
{
    public FeKernelDiagnosticConfigV5(
        uint abiVersion,
        FeKernelDiagnosticMode mode,
        uint sourceSiteIndex,
        uint recordCapacity,
        uint flags,
        uint reserved)
    {
        AbiVersion = abiVersion;
        Mode = (uint)mode;
        SourceSiteIndex = sourceSiteIndex;
        RecordCapacity = recordCapacity;
        Flags = flags;
        Reserved = reserved;
    }

    public readonly uint AbiVersion;
    public readonly uint Mode;
    public readonly uint SourceSiteIndex;
    public readonly uint RecordCapacity;
    public readonly uint Flags;
    public readonly uint Reserved;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeKernelDiagnosticLayoutV5
{
    public readonly uint AbiVersion;
    public readonly uint Mode;
    public readonly uint BufferBinding;
    public readonly uint SiteCount;
    public readonly uint SourceSiteIndex;
    public readonly uint HeaderStrideBytes;
    public readonly uint RecordStrideBytes;
    public readonly uint RecordCapacity;
    public readonly uint RequiredSubgroupFeatures;
    public readonly uint Reserved;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeKernelDiagnosticConfigV6
{
    public FeKernelDiagnosticConfigV6(
        uint abiVersion,
        FeKernelDiagnosticMode mode,
        uint recordCapacity,
        uint selectedX,
        uint selectedY,
        uint selectedZ,
        uint flags,
        uint reserved)
    {
        AbiVersion = abiVersion;
        Mode = (uint)mode;
        RecordCapacity = recordCapacity;
        SelectedX = selectedX;
        SelectedY = selectedY;
        SelectedZ = selectedZ;
        Flags = flags;
        Reserved = reserved;
    }

    public readonly uint AbiVersion;
    public readonly uint Mode;
    public readonly uint RecordCapacity;
    public readonly uint SelectedX;
    public readonly uint SelectedY;
    public readonly uint SelectedZ;
    public readonly uint Flags;
    public readonly uint Reserved;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeKernelDiagnosticLayoutV6
{
    public readonly uint AbiVersion;
    public readonly uint Mode;
    public readonly uint BufferBinding;
    public readonly uint SiteCount;
    public readonly uint HeaderStrideBytes;
    public readonly uint RecordStrideBytes;
    public readonly uint RecordCapacity;
    public readonly uint EventSchemaVersion;
    public readonly uint Flags;
    public readonly uint Reserved;
}

[StructLayout(LayoutKind.Sequential)]
public struct FeBackendCaps
{
    public uint BackendType;
    public uint MaxWorkGroupSizeX;
    public uint MaxWorkGroupSizeY;
    public uint MaxWorkGroupSizeZ;
    public uint SupportsGraphics;
    public uint SupportsAD;
    public uint SupportsNN;
    public uint SupportsWindow;
    public uint SupportsDepthClamp;
    public uint SupportsNonFillPolygonMode;
}

[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
public unsafe struct FeBackendDeviceInfo
{
    public uint NativeAbiVersion;
    public uint MaxTextureDimension2D;
    public uint SupportsTimestampQueries;
    public uint Reserved;
    public fixed byte AdapterName[256];
    public fixed byte DriverVersion[128];
    public fixed byte BackendVersion[64];
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeBackendOperationCounters
{
    public readonly ulong FinishCalls;
    public readonly ulong DeviceWaitIdleCalls;
    public readonly ulong GlobalDrainCalls;
    public readonly ulong BlockingSubmissionWaitCalls;
    public readonly ulong BlockingTextureDownloadCalls;
    public readonly ulong AsyncTextureReadbackCalls;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeBackendResourceCounters
{
    public readonly ulong TrackingSupported;
    public readonly ulong LiveBufferHandles;
    public readonly ulong LiveTextureHandles;
    public readonly ulong LivePipelineHandles;
    public readonly ulong LiveShaderHandles;
    public readonly ulong LiveSubmissionHandles;
    public readonly ulong CachedDescriptorSets;
    public readonly ulong DescriptorPools;
    public readonly ulong CachedSamplers;
    public readonly ulong CachedSubmissionResources;
    public readonly ulong LiveMsaaAttachments;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeResourceAllocationInfo
{
    public readonly ulong Available;
    public readonly ulong PhysicalBytes;
    public readonly ulong AllocationGroup;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeBackendShaderCacheCounters
{
    public readonly ulong TrackingSupported;
    public readonly ulong MemoryCacheHits;
    public readonly ulong DiskCacheHits;
    public readonly ulong DiskCacheMisses;
    public readonly ulong FrontendCompilations;
    public readonly ulong DiskCacheWriteFailures;
    public readonly double LastFrontendMilliseconds;
    public readonly double LastOptimizationMilliseconds;
    public readonly ulong LastMemoryCacheHit;
    public readonly ulong LastDiskCacheHit;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeReadbackMapping
{
    public readonly IntPtr Data;
    public readonly ulong ByteSize;
    public readonly ulong RowPitch;
}

[StructLayout(LayoutKind.Sequential)]
public struct FeWindowDesc
{
    public uint Width;
    public uint Height;

    [MarshalAs(UnmanagedType.LPUTF8Str)]
    public string? Title;

    public uint Resizable;
    public uint Visible;
    public uint VSync;
    public uint HighDpi;
    public uint CenterOnCreate;
}

[StructLayout(LayoutKind.Sequential)]
public struct FeWindowEvent
{
    public uint Kind;
    public uint Key;
    public uint MouseButton;
    public uint Modifiers;
    public uint Pressed;
    public int X;
    public int Y;
    public int DeltaX;
    public int DeltaY;
    public float ScrollX;
    public float ScrollY;
    public uint Width;
    public uint Height;
    public uint Codepoint;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeBufferDesc
{
    public FeBufferDesc(ulong sizeInBytes, uint mode, uint elementStride)
    {
        SizeInBytes = sizeInBytes;
        Mode = mode;
        ElementStride = elementStride;
    }

    public readonly ulong SizeInBytes;
    public readonly uint Mode;
    public readonly uint ElementStride;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeTexture2DDesc
{
    public FeTexture2DDesc(uint width, uint height, uint mipLevels, uint pixelFormat, uint access)
    {
        Width = width;
        Height = height;
        MipLevels = mipLevels;
        PixelFormat = pixelFormat;
        Access = access;
    }

    public readonly uint Width;
    public readonly uint Height;
    public readonly uint MipLevels;
    public readonly uint PixelFormat;
    public readonly uint Access;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeTexture3DDesc
{
    public FeTexture3DDesc(uint width, uint height, uint depth, uint mipLevels, uint pixelFormat, uint access)
    {
        Width = width;
        Height = height;
        Depth = depth;
        MipLevels = mipLevels;
        PixelFormat = pixelFormat;
        Access = access;
    }

    public readonly uint Width;
    public readonly uint Height;
    public readonly uint Depth;
    public readonly uint MipLevels;
    public readonly uint PixelFormat;
    public readonly uint Access;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeSamplerDesc
{
    public FeSamplerDesc(
        uint minFilter,
        uint magFilter,
        uint mipmapMode,
        uint addressU,
        uint addressV,
        uint addressW,
        float mipLodBias,
        float minLod,
        float maxLod,
        uint anisotropyEnable,
        float maxAnisotropy,
        uint compareEnable,
        uint compareOp,
        uint borderColor)
    {
        MinFilter = minFilter;
        MagFilter = magFilter;
        MipmapMode = mipmapMode;
        AddressU = addressU;
        AddressV = addressV;
        AddressW = addressW;
        MipLodBias = mipLodBias;
        MinLod = minLod;
        MaxLod = maxLod;
        AnisotropyEnable = anisotropyEnable;
        MaxAnisotropy = maxAnisotropy;
        CompareEnable = compareEnable;
        CompareOp = compareOp;
        BorderColor = borderColor;
    }

    public readonly uint MinFilter;
    public readonly uint MagFilter;
    public readonly uint MipmapMode;
    public readonly uint AddressU;
    public readonly uint AddressV;
    public readonly uint AddressW;
    public readonly float MipLodBias;
    public readonly float MinLod;
    public readonly float MaxLod;
    public readonly uint AnisotropyEnable;
    public readonly float MaxAnisotropy;
    public readonly uint CompareEnable;
    public readonly uint CompareOp;
    public readonly uint BorderColor;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeGraphicsStencilFaceDesc
{
    public FeGraphicsStencilFaceDesc(uint failOp, uint passOp, uint depthFailOp, uint compareOp)
    {
        FailOp = failOp;
        PassOp = passOp;
        DepthFailOp = depthFailOp;
        CompareOp = compareOp;
    }

    public readonly uint FailOp;
    public readonly uint PassOp;
    public readonly uint DepthFailOp;
    public readonly uint CompareOp;
}

[StructLayout(LayoutKind.Sequential)]
public struct FeKernelCreateDesc
{
    public FeKernelCreateDesc(IntPtr irData, ulong irSize, string? debugName, bool autoDiff = false, bool boundsCheck = false)
    {
        IrData = irData;
        IrSize = irSize;
        DebugName = debugName;
        AutoDiff = autoDiff;
        BoundsCheck = boundsCheck;
    }

    public IntPtr IrData;
    public ulong IrSize;

    [MarshalAs(UnmanagedType.LPUTF8Str)]
    public string? DebugName;

    [MarshalAs(UnmanagedType.I1)]
    public bool AutoDiff;

    [MarshalAs(UnmanagedType.I1)]
    public bool BoundsCheck;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeGraphicsColorBlendAttachmentDesc
{
    public FeGraphicsColorBlendAttachmentDesc(
        uint blendEnable,
        uint srcColor,
        uint dstColor,
        uint colorOp,
        uint srcAlpha,
        uint dstAlpha,
        uint alphaOp,
        uint writeMask)
    {
        BlendEnable = blendEnable;
        SrcColor = srcColor;
        DstColor = dstColor;
        ColorOp = colorOp;
        SrcAlpha = srcAlpha;
        DstAlpha = dstAlpha;
        AlphaOp = alphaOp;
        WriteMask = writeMask;
    }

    public readonly uint BlendEnable;
    public readonly uint SrcColor;
    public readonly uint DstColor;
    public readonly uint ColorOp;
    public readonly uint SrcAlpha;
    public readonly uint DstAlpha;
    public readonly uint AlphaOp;
    public readonly uint WriteMask;
}

[StructLayout(LayoutKind.Sequential)]
public struct FeGraphicsPipelineCreateDesc
{
    public FeGraphicsPipelineCreateDesc(
        IntPtr irData,
        ulong irSize,
        IntPtr vertexIrData,
        ulong vertexIrSize,
        IntPtr fragmentIrData,
        ulong fragmentIrSize,
        uint topology,
        uint sampleCount,
        uint colorAttachmentCount,
        uint depthTest,
        uint depthWrite,
        uint depthCompare,
        uint stencilTest,
        FeGraphicsStencilFaceDesc stencilFront,
        FeGraphicsStencilFaceDesc stencilBack,
        uint stencilReadMask,
        uint stencilWriteMask,
        uint stencilReference,
        uint blendEnable,
        uint blendSrcColor,
        uint blendDstColor,
        uint blendColorOp,
        uint blendSrcAlpha,
        uint blendDstAlpha,
        uint blendAlphaOp,
        uint blendWriteMask,
        uint colorBlendAttachmentCount,
        FeGraphicsColorBlendAttachmentDesc colorBlendAttachment0,
        FeGraphicsColorBlendAttachmentDesc colorBlendAttachment1,
        FeGraphicsColorBlendAttachmentDesc colorBlendAttachment2,
        FeGraphicsColorBlendAttachmentDesc colorBlendAttachment3,
        FeGraphicsColorBlendAttachmentDesc colorBlendAttachment4,
        FeGraphicsColorBlendAttachmentDesc colorBlendAttachment5,
        FeGraphicsColorBlendAttachmentDesc colorBlendAttachment6,
        FeGraphicsColorBlendAttachmentDesc colorBlendAttachment7,
        uint cullMode,
        uint frontFace,
        uint polygonMode,
        uint depthClamp,
        string? debugName)
    {
        IrData = irData;
        IrSize = irSize;
        VertexIrData = vertexIrData;
        VertexIrSize = vertexIrSize;
        FragmentIrData = fragmentIrData;
        FragmentIrSize = fragmentIrSize;
        Topology = topology;
        SampleCount = sampleCount;
        ColorAttachmentCount = colorAttachmentCount;
        DepthTest = depthTest;
        DepthWrite = depthWrite;
        DepthCompare = depthCompare;
        StencilTest = stencilTest;
        StencilFront = stencilFront;
        StencilBack = stencilBack;
        StencilReadMask = stencilReadMask;
        StencilWriteMask = stencilWriteMask;
        StencilReference = stencilReference;
        BlendEnable = blendEnable;
        BlendSrcColor = blendSrcColor;
        BlendDstColor = blendDstColor;
        BlendColorOp = blendColorOp;
        BlendSrcAlpha = blendSrcAlpha;
        BlendDstAlpha = blendDstAlpha;
        BlendAlphaOp = blendAlphaOp;
        BlendWriteMask = blendWriteMask;
        ColorBlendAttachmentCount = colorBlendAttachmentCount;
        ColorBlendAttachment0 = colorBlendAttachment0;
        ColorBlendAttachment1 = colorBlendAttachment1;
        ColorBlendAttachment2 = colorBlendAttachment2;
        ColorBlendAttachment3 = colorBlendAttachment3;
        ColorBlendAttachment4 = colorBlendAttachment4;
        ColorBlendAttachment5 = colorBlendAttachment5;
        ColorBlendAttachment6 = colorBlendAttachment6;
        ColorBlendAttachment7 = colorBlendAttachment7;
        CullMode = cullMode;
        FrontFace = frontFace;
        PolygonMode = polygonMode;
        DepthClamp = depthClamp;
        DebugName = debugName;
    }

    public IntPtr IrData;
    public ulong IrSize;
    public IntPtr VertexIrData;
    public ulong VertexIrSize;
    public IntPtr FragmentIrData;
    public ulong FragmentIrSize;
    public uint Topology;
    public uint SampleCount;

    public uint ColorAttachmentCount;
    public uint DepthTest;
    public uint DepthWrite;
    public uint DepthCompare;
    public uint StencilTest;
    public FeGraphicsStencilFaceDesc StencilFront;
    public FeGraphicsStencilFaceDesc StencilBack;
    public uint StencilReadMask;
    public uint StencilWriteMask;
    public uint StencilReference;
    public uint BlendEnable;
    public uint BlendSrcColor;
    public uint BlendDstColor;
    public uint BlendColorOp;
    public uint BlendSrcAlpha;
    public uint BlendDstAlpha;
    public uint BlendAlphaOp;
    public uint BlendWriteMask;
    public uint ColorBlendAttachmentCount;
    public FeGraphicsColorBlendAttachmentDesc ColorBlendAttachment0;
    public FeGraphicsColorBlendAttachmentDesc ColorBlendAttachment1;
    public FeGraphicsColorBlendAttachmentDesc ColorBlendAttachment2;
    public FeGraphicsColorBlendAttachmentDesc ColorBlendAttachment3;
    public FeGraphicsColorBlendAttachmentDesc ColorBlendAttachment4;
    public FeGraphicsColorBlendAttachmentDesc ColorBlendAttachment5;
    public FeGraphicsColorBlendAttachmentDesc ColorBlendAttachment6;
    public FeGraphicsColorBlendAttachmentDesc ColorBlendAttachment7;
    public uint CullMode;
    public uint FrontFace;
    public uint PolygonMode;
    public uint DepthClamp;

    [MarshalAs(UnmanagedType.LPUTF8Str)]
    public string? DebugName;
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct FeGraphicsDrawDesc
{
    public FeGraphicsDrawDesc(
        IntPtr colorTargets,
        uint colorTargetCount,
        ulong depthTarget,
        uint count,
        ulong indexBuffer,
        uint indexed,
        uint wait,
        uint viewportEnabled,
        uint viewportX,
        uint viewportY,
        uint viewportWidth,
        uint viewportHeight,
        uint scissorEnabled,
        uint scissorX,
        uint scissorY,
        uint scissorWidth,
        uint scissorHeight,
        uint clearDepth,
        float clearDepthValue,
        uint depthLoadOp,
        uint clearColor,
        float clearColorR,
        float clearColorG,
        float clearColorB,
        float clearColorA,
        uint colorLoadOp,
        uint instanceCount,
        uint firstVertex,
        uint firstIndex,
        int vertexOffset,
        uint firstInstance)
    {
        ColorTargets = colorTargets;
        ColorTargetCount = colorTargetCount;
        DepthTarget = depthTarget;
        Count = count;
        IndexBuffer = indexBuffer;
        Indexed = indexed;
        Wait = wait;
        ViewportEnabled = viewportEnabled;
        ViewportX = viewportX;
        ViewportY = viewportY;
        ViewportWidth = viewportWidth;
        ViewportHeight = viewportHeight;
        ScissorEnabled = scissorEnabled;
        ScissorX = scissorX;
        ScissorY = scissorY;
        ScissorWidth = scissorWidth;
        ScissorHeight = scissorHeight;
        ClearDepth = clearDepth;
        ClearDepthValue = clearDepthValue;
        DepthLoadOp = depthLoadOp;
        ClearColor = clearColor;
        ClearColorR = clearColorR;
        ClearColorG = clearColorG;
        ClearColorB = clearColorB;
        ClearColorA = clearColorA;
        ColorLoadOp = colorLoadOp;
        InstanceCount = instanceCount;
        FirstVertex = firstVertex;
        FirstIndex = firstIndex;
        VertexOffset = vertexOffset;
        FirstInstance = firstInstance;
    }

    public readonly IntPtr ColorTargets;
    public readonly uint ColorTargetCount;
    public readonly ulong DepthTarget;
    public readonly uint Count;
    public readonly ulong IndexBuffer;
    public readonly uint Indexed;
    public readonly uint Wait;
    public readonly uint ViewportEnabled;
    public readonly uint ViewportX;
    public readonly uint ViewportY;
    public readonly uint ViewportWidth;
    public readonly uint ViewportHeight;
    public readonly uint ScissorEnabled;
    public readonly uint ScissorX;
    public readonly uint ScissorY;
    public readonly uint ScissorWidth;
    public readonly uint ScissorHeight;
    public readonly uint ClearDepth;
    public readonly float ClearDepthValue;
    public readonly uint DepthLoadOp;
    public readonly uint ClearColor;
    public readonly float ClearColorR;
    public readonly float ClearColorG;
    public readonly float ClearColorB;
    public readonly float ClearColorA;
    public readonly uint ColorLoadOp;
    public readonly uint InstanceCount;
    public readonly uint FirstVertex;
    public readonly uint FirstIndex;
    public readonly int VertexOffset;
    public readonly uint FirstInstance;
}

/// <summary>
/// Native aggregate timing data returned for a profiled GPU command name.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct FeProfilerQueryResult
{
    public readonly ulong Count;
    public readonly double MinTimeMs;
    public readonly double MaxTimeMs;
    public readonly double AverageTimeMs;
    public readonly double TotalTimeMs;
}

/// <summary>
/// Native hardware interval placed on its submission-relative GPU timeline.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct FeTimestampIntervalResult
{
    public readonly ulong StartOffsetNanoseconds;
    public readonly ulong DurationNanoseconds;
}

[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
public unsafe struct FeADGradientInfo
{
    public fixed byte Name[128];
    public fixed byte ResourceName[128];
    public fixed byte ElementType[64];
    public fixed byte EasyGpuName[64];
    public uint SourceBinding;
    public uint GradientBinding;
    public uint ElementCount;
    public uint ElementStride;
    public ulong ByteSize;
    public uint ComponentCount;
    public uint Reserved;
}
