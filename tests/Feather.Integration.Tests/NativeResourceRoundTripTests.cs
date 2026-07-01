using System.Runtime.InteropServices;
using Feather.Native;
using Feather.Resources;

namespace Feather.Integration.Tests;

public class NativeResourceRoundTripTests
{
    [Fact]
    public void ContextReportsRealBackendCapsWhenNativeBackendIsAvailable()
    {
        NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_default(out var context));
        try
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_context_initialize(context));
        }
        catch (FeatherNativeException ex) when (ex.Result == FeResult.ErrorBackendUnavailable)
        {
            Assert.Contains("EasyGPU", ex.Message, StringComparison.OrdinalIgnoreCase);
            return;
        }

        NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_backend_type(context, out var backend));
        NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_caps(context, out var caps));

        Assert.True(backend is 1u or 2u);
        Assert.Equal(backend, caps.BackendType);
        Assert.True(caps.MaxWorkGroupSizeX > 0);
        Assert.True(caps.MaxWorkGroupSizeY > 0);
        Assert.True(caps.MaxWorkGroupSizeZ > 0);
    }

    [Fact]
    public void BufferUploadAndDownloadRoundTripThroughNativeAbi()
    {
        using var buffer = GPU.CreateBuffer<int>([1, 2, 3, 4]);

        buffer.Upload(1, [20, 30]);

        Assert.Equal([1, 20, 30, 4], buffer.ToArray());
    }

    [Fact]
    public void TextureUploadAndDownloadRoundTripThroughNativeAbi()
    {
        using var texture = GPU.CreateTexture2D<Rgba32, Rgba32>(2, 2, PixelFormat.Rgba8);
        var pixels = new[]
        {
            new Rgba32(1, 2, 3, 4),
            new Rgba32(5, 6, 7, 8),
            new Rgba32(9, 10, 11, 12),
            new Rgba32(13, 14, 15, 16)
        };

        texture.Upload(pixels);

        var readback = new Rgba32[4];
        texture.Read(readback);
        Assert.Equal(pixels, readback);
    }

    [Fact]
    public void TextureGenerateMipmapsCallsNativeAbiAndPreservesBaseLevel()
    {
        using var texture = GPU.CreateTexture2D<Rgba32, Rgba32>(4, 4, 3, PixelFormat.Rgba8);
        var pixels = Enumerable.Range(0, 16)
            .Select(value => new Rgba32((byte)value, (byte)(value + 1), (byte)(value + 2), byte.MaxValue))
            .ToArray();

        texture.Upload(pixels);
        texture.GenerateMipmaps();

        var readback = new Rgba32[16];
        texture.Read(readback);
        Assert.Equal(3, texture.MipLevels);
        Assert.Equal(pixels, readback);
    }

    [Fact]
    public void Texture3DUploadAndDownloadRoundTripThroughNativeAbi()
    {
        using var texture = GPU.CreateTexture3D<Rgba32, Rgba32>(2, 2, 2, 2, PixelFormat.Rgba8);
        var voxels = new[]
        {
            new Rgba32(1, 2, 3, 4),
            new Rgba32(5, 6, 7, 8),
            new Rgba32(9, 10, 11, 12),
            new Rgba32(13, 14, 15, 16),
            new Rgba32(17, 18, 19, 20),
            new Rgba32(21, 22, 23, 24),
            new Rgba32(25, 26, 27, 28),
            new Rgba32(29, 30, 31, 32)
        };

        texture.Upload(voxels);
        var exception = Assert.Throws<FeatherNativeException>(texture.GenerateMipmaps);
        Assert.Equal(FeResult.ErrorUnsupported, exception.Result);
        Assert.Contains("2D textures", exception.Message, StringComparison.Ordinal);

        var readback = new Rgba32[8];
        texture.Read(readback);
        Assert.Equal(new Feather.Math.int3(2, 2, 2), texture.Size);
        Assert.Equal(voxels, readback);
    }

    [Fact]
    public void TextureImageSaveAndLoadRoundTripsRgba8()
    {
        var path = Path.Combine(Path.GetTempPath(), $"feather-texture-{Guid.NewGuid():N}.tga");
        var pixels = new[]
        {
            new Rgba32(1, 2, 3, 255),
            new Rgba32(5, 6, 7, 128),
            new Rgba32(9, 10, 11, 64),
            new Rgba32(13, 14, 15, 0)
        };

        try
        {
            using (var texture = GPU.CreateTexture2D<Rgba32, Rgba32>(2, 2, PixelFormat.Rgba8))
            {
                texture.Upload(pixels);
                texture.Save(path);
            }

            using var loaded = GPU.LoadReadWriteTexture2D<Rgba32, Rgba32>(path);
            var readback = new Rgba32[4];
            loaded.Read(readback);

            Assert.Equal(2, loaded.Width);
            Assert.Equal(2, loaded.Height);
            Assert.Equal(PixelFormat.Rgba8, loaded.Format);
            Assert.Equal(pixels, readback);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TextureImageLoadsAsSampledTexture()
    {
        var path = Path.Combine(Path.GetTempPath(), $"feather-sampled-texture-{Guid.NewGuid():N}.tga");
        var pixels = new[]
        {
            new Rgba32(20, 30, 40, 255),
            new Rgba32(50, 60, 70, 255),
            new Rgba32(80, 90, 100, 255),
            new Rgba32(110, 120, 130, 255)
        };

        try
        {
            using (var texture = GPU.CreateTexture2D<Rgba32, Rgba32>(2, 2, PixelFormat.Rgba8))
            {
                texture.Upload(pixels);
                texture.Save(path);
            }

            using var sampled = GPU.LoadSampledTexture2D<Rgba32, Rgba32>(path);
            var view = sampled.AsSampled();
            var readback = new Rgba32[4];
            sampled.Read(readback);

            Assert.Equal(TextureAccess.Sampled, sampled.Access);
            Assert.Equal(new Feather.Math.int2(2, 2), view.Size);
            Assert.Equal(pixels, readback);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void SamplerCreationRoundTripsThroughNativeAbi()
    {
        using var sampler = GPU.CreateSampler(SamplerDesc.LinearRepeat);

        Assert.Equal(SamplerDesc.LinearRepeat, sampler.Desc);
    }

    [Fact]
    public void NativeIrValidationAcceptsStructuredFeatherIr()
    {
        var ir = BuildMinimalComputeIr();
        Assert.Equal(FeResult.Ok, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationAcceptsElementwiseAssignmentSection()
    {
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(1, BuildElementwiseAssignmentSection(new IrElementwiseAssignment(0, 1, 0, uint.MaxValue, 1, 0, 1, uint.MaxValue)))
            ],
            new IrInstruction(2, 2, 0));

        Assert.Equal(FeResult.Ok, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationAcceptsElementwiseExpressionAssignmentSection()
    {
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(2, BuildElementwiseExpressionAssignmentSection())
            ],
            new IrInstruction(2, 2, 0));

        Assert.Equal(FeResult.Ok, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationAcceptsVersionedTypedIrSection()
    {
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, BuildTypedIrSection())
            ]);

        Assert.Equal(FeResult.Ok, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationRejectsOldUnversionedTypedIrSection()
    {
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            writer.Write((uint)0);
            writer.Write((uint)0);
            writer.Write((uint)0);
            writer.Write((uint)1);
            writer.Write((uint)1);
            writer.Write((uint)0);
            writer.Write((uint)0);
        }

        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, stream.ToArray())
            ]);

        Assert.Equal(FeResult.ErrorInvalidArgument, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationRejectsTypedIrWithNonContiguousTableOffset()
    {
        var section = BuildTypedIrSection();
        WriteUInt32(section, offset: 24, value: 999); // Type table offset must immediately follow functions.
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, section)
            ]);

        Assert.Equal(FeResult.ErrorInvalidArgument, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationRejectsTypedIrWithOutOfRangeArgumentSpan()
    {
        var section = BuildTypedIrSection(includeCallableExpression: true);
        var expressionOffset = ReadUInt32(section, 56);
        WriteUInt32(section, checked((int)expressionOffset + 29), 2); // Only one argument exists.
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, section)
            ]);

        Assert.Equal(FeResult.ErrorInvalidArgument, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationRejectsTypedIrCallableExpressionWithoutFunctionRecord()
    {
        var section = BuildTypedIrSection(includeCallableExpression: true);
        var expressionOffset = ReadUInt32(section, 56);
        WriteUInt32(section, checked((int)expressionOffset + 17), 1); // Literal string, not a callable name.
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, section)
            ]);

        Assert.Equal(FeResult.ErrorInvalidArgument, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationRejectsTypedIrNonSquareMatrixType()
    {
        var section = BuildTypedIrSection();
        var typeOffset = ReadUInt32(section, 24);
        var matrixTypeOffset = checked((int)typeOffset + (17 * 2));
        section[matrixTypeOffset] = 3; // matrix
        WriteUInt32(section, matrixTypeOffset + 1, 1); // float type id
        WriteUInt32(section, matrixTypeOffset + 5, 2); // rows
        WriteUInt32(section, matrixTypeOffset + 9, 3); // columns
        WriteUInt32(section, matrixTypeOffset + 13, 0);
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, section)
            ]);

        Assert.Equal(FeResult.ErrorInvalidArgument, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationRejectsTypedIrStructFieldWithInvalidLayout()
    {
        var section = BuildTypedIrStructLayoutSection();
        var fieldOffset = ReadUInt32(section, 40);
        WriteUInt32(section, checked((int)fieldOffset + 8), 4); // float3 field must be aligned to 16.
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, section)
            ]);

        Assert.Equal(FeResult.ErrorInvalidArgument, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationAcceptsTypedIrZeroArgumentCallExpression()
    {
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, BuildTypedIrSection(includeZeroArgumentCallExpression: true))
            ]);

        Assert.Equal(FeResult.Ok, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationRejectsTypedIrBarrierWithUnknownKind()
    {
        var section = BuildTypedIrSection();
        var statementOffset = ReadUInt32(section, 48);
        section[statementOffset] = 13; // barrier statement
        WriteUInt32(section, checked((int)statementOffset + 13), 99); // barrier kind
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, section)
            ]);

        Assert.Equal(FeResult.ErrorInvalidArgument, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationAcceptsTypedIrSharedMemoryRecords()
    {
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, BuildTypedIrSharedMemorySection())
            ]);

        Assert.Equal(FeResult.Ok, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationRejectsTypedIrSharedMemoryDeclarationWithZeroLength()
    {
        var section = BuildTypedIrSharedMemorySection();
        var statementOffset = ReadUInt32(section, 48);
        WriteUInt32(section, checked((int)statementOffset + 29 + 1), 0); // shared declaration A = element count.
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, section)
            ]);

        Assert.Equal(FeResult.ErrorInvalidArgument, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationRejectsTypedIrSharedMemoryExpressionWithoutName()
    {
        var section = BuildTypedIrSharedMemorySection();
        var expressionOffset = ReadUInt32(section, 56);
        WriteUInt32(section, checked((int)expressionOffset + 33 + 17), uint.MaxValue); // shared element expression name.
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, section)
            ]);

        Assert.Equal(FeResult.ErrorInvalidArgument, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationRejectsTypedIrSharedMemoryLValueWithOutOfRangeIndex()
    {
        var section = BuildTypedIrSharedMemorySection();
        var lvalueOffset = ReadUInt32(section, 64);
        WriteUInt32(section, checked((int)lvalueOffset + 5), 999); // shared l-value A = index expression id.
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, section)
            ]);

        Assert.Equal(FeResult.ErrorInvalidArgument, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationAcceptsTypedIrAtomicRecords()
    {
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, BuildTypedIrAtomicSection())
            ]);

        Assert.Equal(FeResult.Ok, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationRejectsTypedIrAtomicWithOutOfRangeLValue()
    {
        var section = BuildTypedIrAtomicSection();
        var expressionOffset = ReadUInt32(section, 56);
        WriteUInt32(section, checked((int)expressionOffset + (2 * 33) + 5), 999); // atomic A = l-value id.
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, section)
            ]);

        Assert.Equal(FeResult.ErrorInvalidArgument, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationRejectsTypedIrAtomicWithUnknownOperation()
    {
        var section = BuildTypedIrAtomicSection();
        var expressionOffset = ReadUInt32(section, 56);
        WriteUInt32(section, checked((int)expressionOffset + (2 * 33) + 21), 99); // atomic op.
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, section)
            ]);

        Assert.Equal(FeResult.ErrorInvalidArgument, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationRejectsTypedIrAtomicWithWrongArgumentCount()
    {
        var section = BuildTypedIrAtomicSection();
        var expressionOffset = ReadUInt32(section, 56);
        WriteUInt32(section, checked((int)expressionOffset + (2 * 33) + 29), 2); // add requires one value argument.
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, section)
            ]);

        Assert.Equal(FeResult.ErrorInvalidArgument, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationAcceptsTypedIrTextureSampleRecords()
    {
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, BuildTypedIrTextureSampleSection())
            ]);

        Assert.Equal(FeResult.Ok, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationAcceptsTypedIrTextureSampleGradRecords()
    {
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, BuildTypedIrTextureSampleSection(sampleOp: 2, argumentCount: 5))
            ]);

        Assert.Equal(FeResult.Ok, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationRejectsTypedIrTextureSampleWithUnknownOperation()
    {
        var section = BuildTypedIrTextureSampleSection();
        var expressionOffset = ReadUInt32(section, 56);
        WriteUInt32(section, checked((int)expressionOffset + (5 * 33) + 21), 99); // texture sample op.
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, section)
            ]);

        Assert.Equal(FeResult.ErrorInvalidArgument, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationRejectsTypedIrTextureSampleWithWrongArgumentCount()
    {
        var section = BuildTypedIrTextureSampleSection();
        var expressionOffset = ReadUInt32(section, 56);
        WriteUInt32(section, checked((int)expressionOffset + (5 * 33) + 29), 4); // Sample requires three operands.
        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, section)
            ]);

        Assert.Equal(FeResult.ErrorInvalidArgument, ValidateIr(ir));
    }

    [Theory]
    [InlineData(TypedIrMutation.BadMagic)]
    [InlineData(TypedIrMutation.UnsupportedMajorVersion)]
    [InlineData(TypedIrMutation.UnsupportedMinorVersion)]
    [InlineData(TypedIrMutation.WrongEndian)]
    [InlineData(TypedIrMutation.WrongHeaderSize)]
    [InlineData(TypedIrMutation.OutOfRangeEntryFunction)]
    [InlineData(TypedIrMutation.ZeroFunctionCount)]
    [InlineData(TypedIrMutation.ZeroTypeCount)]
    [InlineData(TypedIrMutation.ZeroStatementCount)]
    [InlineData(TypedIrMutation.OverlappingTypeTable)]
    [InlineData(TypedIrMutation.StringTableGap)]
    [InlineData(TypedIrMutation.StringTableTooLong)]
    [InlineData(TypedIrMutation.TruncatedStringBytes)]
    [InlineData(TypedIrMutation.FunctionBadName)]
    [InlineData(TypedIrMutation.FunctionBadReturnType)]
    [InlineData(TypedIrMutation.FunctionBadParameterSpan)]
    [InlineData(TypedIrMutation.FunctionBadBody)]
    [InlineData(TypedIrMutation.TypeBadKind)]
    [InlineData(TypedIrMutation.StatementBadKind)]
    [InlineData(TypedIrMutation.StatementBadChildSpan)]
    [InlineData(TypedIrMutation.ExpressionBadKind)]
    [InlineData(TypedIrMutation.ExpressionBadArgumentSpan)]
    [InlineData(TypedIrMutation.ExpressionUnexpectedArgumentSpan)]
    [InlineData(TypedIrMutation.ArgumentBadExpressionIndex)]
    [InlineData(TypedIrMutation.ParameterBadDirection)]
    public void NativeIrValidationRejectsFuzzedTypedIrPayloads(TypedIrMutation mutation)
    {
        var section = BuildTypedIrSection(includeCallableExpression: true);
        MutateTypedIrSection(section, mutation);

        var ir = BuildMinimalComputeIr(
            minor: 1,
            sections:
            [
                new IrSection(7, section)
            ]);

        Assert.Equal(FeResult.ErrorInvalidArgument, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationRejectsMalformedIr()
    {
        var ir = new byte[] { 1, 2, 3, 4 };
        Assert.Equal(FeResult.ErrorInvalidArgument, ValidateIr(ir));
    }

    [Fact]
    public void NativeIrValidationRejectsUnbalancedStructuredControlFlow()
    {
        var ir = BuildMinimalComputeIr(new IrInstruction(15, 0, 0));

        Assert.Equal(FeResult.ErrorInvalidArgument, ValidateIr(ir));
    }

    private static FeResult ValidateIr(byte[] ir)
    {
        var ptr = Marshal.AllocHGlobal(ir.Length);
        try
        {
            Marshal.Copy(ir, 0, ptr, ir.Length);
            return NativeMethods.fe_ir_validate(ptr, (ulong)ir.Length);
        }
        finally
        {
            Marshal.FreeHGlobal(ptr);
        }
    }

    private readonly record struct Rgba32(byte R, byte G, byte B, byte A);

    private static byte[] BuildMinimalComputeIr(params IrInstruction[] instructions)
        => BuildMinimalComputeIr(minor: 0, sections: [], instructions);

    private static byte[] BuildMinimalComputeIr(ushort minor, IrSection[] sections, params IrInstruction[] instructions)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        writer.Write("FEIR"u8);
        writer.Write((ushort)1);
        writer.Write(minor);
        writer.Write((byte)1);
        writer.Write((byte)1);
        writer.Write((ushort)sections.Length);
        writer.Write(256);
        writer.Write(1);
        writer.Write(1);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)instructions.Length);

        using var strings = new MemoryStream();
        using (var stringWriter = new BinaryWriter(strings, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            stringWriter.Write((uint)2);
            stringWriter.Write((uint)5);
            stringWriter.Write("Smoke"u8);
            stringWriter.Write((uint)1);
            stringWriter.Write((uint)1);
            stringWriter.Write("i"u8);
        }

        var stringBytes = strings.ToArray();
        writer.Write((uint)stringBytes.Length);

        foreach (var instruction in instructions)
        {
            writer.Write(instruction.Opcode);
            writer.Write(instruction.OperandKind);
            writer.Write((ushort)0);
            writer.Write(instruction.OperandStringId);
        }

        foreach (var section in sections)
        {
            writer.Write(section.Kind);
            writer.Write((uint)section.Payload.Length);
        }

        foreach (var section in sections)
        {
            writer.Write(section.Payload);
        }

        writer.Write(stringBytes);
        return stream.ToArray();
    }

    private static byte[] BuildElementwiseAssignmentSection(params IrElementwiseAssignment[] assignments)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        writer.Write((uint)assignments.Length);
        foreach (var assignment in assignments)
        {
            writer.Write(assignment.InstructionIndex);
            writer.Write(assignment.DestinationBinding);
            writer.Write(assignment.LeftBinding);
            writer.Write(assignment.RightBinding);
            writer.Write(assignment.Operation);
            writer.Write(assignment.RightOperandKind);
            writer.Write((ushort)0);
            writer.Write(assignment.IndexStringId);
            writer.Write(assignment.RightLiteralStringId);
        }

        return stream.ToArray();
    }

    private static byte[] BuildElementwiseExpressionAssignmentSection()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        writer.Write((uint)1);
        writer.Write((uint)3);

        writer.Write((uint)0);
        writer.Write((uint)1);
        writer.Write((uint)1);
        writer.Write((uint)2);

        WriteExpressionNode(writer, kind: 1, operation: 0, resourceBinding: 0, indexStringId: 1, literalStringId: uint.MaxValue, typeStringId: uint.MaxValue);
        WriteExpressionNode(writer, kind: 2, operation: 0, resourceBinding: uint.MaxValue, indexStringId: uint.MaxValue, literalStringId: 1, typeStringId: uint.MaxValue);
        WriteExpressionNode(writer, kind: 3, operation: 3, resourceBinding: uint.MaxValue, indexStringId: uint.MaxValue, literalStringId: uint.MaxValue, typeStringId: uint.MaxValue, leftNodeIndex: 0, rightNodeIndex: 1);
        return stream.ToArray();
    }

    private static byte[] BuildTypedIrSection(bool includeCallableExpression = false, bool includeZeroArgumentCallExpression = false)
    {
        const int headerSize = 104;

        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        writer.Write(new byte[headerSize]);

        var functionOffset = (uint)stream.Position;
        writer.Write((byte)0); // Compute1D
        writer.Write((uint)0); // name
        writer.Write((uint)0); // mangled name
        writer.Write((uint)0); // void type
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);
        writer.Write((uint)0); // body statement

        var hasCallable = includeCallableExpression || includeZeroArgumentCallExpression;
        if (hasCallable)
        {
            writer.Write((byte)5); // Callable
            writer.Write((uint)2); // name "Helper"
            writer.Write((uint)2); // mangled name "Helper"
            writer.Write((uint)0); // void type
            writer.Write(includeCallableExpression ? 0u : uint.MaxValue);
            writer.Write(includeCallableExpression ? 1u : 0u);
            writer.Write((uint)1); // callable body statement
        }

        var typeOffset = (uint)stream.Position;
        writer.Write((byte)7); // void
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)0);

        var structOffset = (uint)stream.Position;
        var structFieldOffset = (uint)stream.Position;

        var statementOffset = (uint)stream.Position;
        writer.Write((byte)1); // block
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);

        if (hasCallable)
        {
            writer.Write((byte)1); // callable block
            writer.Write(uint.MaxValue);
            writer.Write(uint.MaxValue);
            writer.Write(uint.MaxValue);
            writer.Write((uint)0);
            writer.Write(uint.MaxValue);
            writer.Write(uint.MaxValue);
            writer.Write((uint)0);
        }

        var expressionOffset = (uint)stream.Position;
        var expressionCount = includeCallableExpression ? 2u : includeZeroArgumentCallExpression ? 1u : 0u;
        if (includeCallableExpression || includeZeroArgumentCallExpression)
        {
            writer.Write((byte)14); // callable call
            writer.Write((uint)0); // void type is enough for validation
            writer.Write(uint.MaxValue);
            writer.Write(uint.MaxValue);
            writer.Write(uint.MaxValue);
            writer.Write((uint)2); // callable name id
            writer.Write((uint)0);
            writer.Write(includeCallableExpression ? 0u : uint.MaxValue); // first argument
            writer.Write(includeCallableExpression ? 1u : 0u); // argument count
        }

        if (includeCallableExpression)
        {
            writer.Write((byte)1); // literal argument
            writer.Write((uint)0);
            writer.Write(uint.MaxValue);
            writer.Write(uint.MaxValue);
            writer.Write(uint.MaxValue);
            writer.Write((uint)1); // literal string id
            writer.Write((uint)0);
            writer.Write(uint.MaxValue);
            writer.Write((uint)0);
        }

        var lvalueOffset = (uint)stream.Position;
        var childOffset = (uint)stream.Position;
        var argumentOffset = (uint)stream.Position;
        var argumentCount = includeCallableExpression ? 1u : 0u;
        if (includeCallableExpression)
        {
            writer.Write((uint)1);
        }

        var parameterOffset = (uint)stream.Position;
        var parameterCount = includeCallableExpression ? 1u : 0u;
        if (includeCallableExpression)
        {
            writer.Write((byte)0); // in
            writer.Write((uint)3); // parameter name "value"
            writer.Write((uint)0); // void type is enough for parser validation
        }

        var stringOffset = (uint)stream.Position;
        using (var stringStream = new MemoryStream())
        {
            using var stringWriter = new BinaryWriter(stringStream, System.Text.Encoding.UTF8, leaveOpen: true);
            stringWriter.Write(includeCallableExpression ? 4u : hasCallable ? 3u : 2u);
            stringWriter.Write((uint)5);
            stringWriter.Write("Entry"u8);
            stringWriter.Write((uint)1);
            stringWriter.Write("1"u8);
            if (hasCallable)
            {
                stringWriter.Write((uint)6);
                stringWriter.Write("Helper"u8);
            }
            if (includeCallableExpression)
            {
                stringWriter.Write((uint)5);
                stringWriter.Write("value"u8);
            }
            var strings = stringStream.ToArray();
            writer.Write(strings);
        }

        var payload = stream.ToArray();
        using var headerStream = new MemoryStream(payload);
        using var header = new BinaryWriter(headerStream, System.Text.Encoding.UTF8, leaveOpen: true);
        header.Write("FTIR"u8);
        header.Write((ushort)1);
        header.Write((ushort)0);
        header.Write((byte)1);
        header.Write((byte)0);
        header.Write((ushort)headerSize);
        header.Write((uint)0);
        WriteRange(header, functionOffset, hasCallable ? 2u : 1u);
        WriteRange(header, typeOffset, 1);
        WriteRange(header, structOffset, 0);
        WriteRange(header, structFieldOffset, 0);
        WriteRange(header, statementOffset, hasCallable ? 2u : 1u);
        WriteRange(header, expressionOffset, expressionCount);
        WriteRange(header, lvalueOffset, 0);
        WriteRange(header, childOffset, 0);
        WriteRange(header, argumentOffset, argumentCount);
        WriteRange(header, parameterOffset, parameterCount);
        header.Write(stringOffset);
        header.Write(checked((uint)(payload.Length - stringOffset)));
        return payload;

        static void WriteRange(BinaryWriter writer, uint offset, uint count)
        {
            writer.Write(offset);
            writer.Write(count);
        }
    }

    private static byte[] BuildTypedIrStructLayoutSection()
    {
        const int headerSize = 104;
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        writer.Write(new byte[headerSize]);

        var functionOffset = checked((uint)stream.Position);
        writer.Write((byte)0); // Compute1D
        writer.Write((uint)0); // name
        writer.Write((uint)0); // mangled name
        writer.Write((uint)0); // void return
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);
        writer.Write((uint)0); // body statement

        var typeOffset = checked((uint)stream.Position);
        writer.Write((byte)7); // void
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((byte)1); // primitive float
        writer.Write((uint)3);
        writer.Write((uint)32);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((byte)2); // float3
        writer.Write((uint)1);
        writer.Write((uint)3);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((byte)4); // struct Packed
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)0);

        var structOffset = checked((uint)stream.Position);
        writer.Write((uint)1); // Packed
        writer.Write((uint)2); // global::Packed
        writer.Write((uint)0);
        writer.Write((uint)1);
        writer.Write((uint)16);
        writer.Write((uint)16);

        var structFieldOffset = checked((uint)stream.Position);
        writer.Write((uint)3); // Position
        writer.Write((uint)2); // float3 type
        writer.Write((uint)0);
        writer.Write((uint)12);

        var statementOffset = checked((uint)stream.Position);
        writer.Write((byte)1); // block
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);

        var expressionOffset = checked((uint)stream.Position);
        var lvalueOffset = checked((uint)stream.Position);
        var childOffset = checked((uint)stream.Position);
        var argumentOffset = checked((uint)stream.Position);
        var parameterOffset = checked((uint)stream.Position);
        var stringOffset = checked((uint)stream.Position);
        writer.Write((uint)4);
        WriteString(writer, "Entry");
        WriteString(writer, "Packed");
        WriteString(writer, "global::Packed");
        WriteString(writer, "Position");
        var stringLength = checked((uint)(stream.Position - stringOffset));

        stream.Position = 0;
        writer.Write("FTIR"u8);
        writer.Write((ushort)1);
        writer.Write((ushort)0);
        writer.Write((byte)1);
        writer.Write((byte)0);
        writer.Write((ushort)headerSize);
        writer.Write((uint)0);
        WriteRange(writer, functionOffset, 1);
        WriteRange(writer, typeOffset, 4);
        WriteRange(writer, structOffset, 1);
        WriteRange(writer, structFieldOffset, 1);
        WriteRange(writer, statementOffset, 1);
        WriteRange(writer, expressionOffset, 0);
        WriteRange(writer, lvalueOffset, 0);
        WriteRange(writer, childOffset, 0);
        WriteRange(writer, argumentOffset, 0);
        WriteRange(writer, parameterOffset, 0);
        writer.Write(stringOffset);
        writer.Write(stringLength);
        return stream.ToArray();

        static void WriteString(BinaryWriter writer, string value)
        {
            var bytes = System.Text.Encoding.UTF8.GetBytes(value);
            writer.Write((uint)bytes.Length);
            writer.Write(bytes);
        }

        static void WriteRange(BinaryWriter writer, uint offset, uint count)
        {
            writer.Write(offset);
            writer.Write(count);
        }
    }

    private static byte[] BuildTypedIrSharedMemorySection()
    {
        const int headerSize = 104;

        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        writer.Write(new byte[headerSize]);

        var functionOffset = (uint)stream.Position;
        writer.Write((byte)0); // Compute1D
        writer.Write((uint)0); // name
        writer.Write((uint)0); // mangled name
        writer.Write((uint)0); // void type
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);
        writer.Write((uint)0); // body statement

        var typeOffset = (uint)stream.Position;
        writer.Write((byte)7); // void
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((byte)1); // primitive int
        writer.Write((uint)1);
        writer.Write((uint)32);
        writer.Write((uint)0);
        writer.Write((uint)0);

        var structOffset = (uint)stream.Position;
        var structFieldOffset = (uint)stream.Position;

        var statementOffset = (uint)stream.Position;
        writer.Write((byte)1); // block
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0); // first child
        writer.Write((uint)2); // child count

        writer.Write((byte)15); // shared memory declaration
        writer.Write((uint)8); // element count
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)1); // int type id
        writer.Write((uint)2); // "shared_values"
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);

        writer.Write((byte)3); // assignment
        writer.Write((uint)0); // shared l-value
        writer.Write((uint)1); // shared read expression
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);

        var expressionOffset = (uint)stream.Position;
        writer.Write((byte)1); // literal index
        writer.Write((uint)1); // int type id
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)1); // "0"
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);

        writer.Write((byte)21); // shared memory element expression
        writer.Write((uint)1); // int type id
        writer.Write((uint)0); // index expression
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)2); // "shared_values"
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);

        var lvalueOffset = (uint)stream.Position;
        writer.Write((byte)9); // shared memory element l-value
        writer.Write((uint)1); // int type id
        writer.Write((uint)0); // index expression
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)2); // "shared_values"

        var childOffset = (uint)stream.Position;
        writer.Write((uint)1);
        writer.Write((uint)2);

        var argumentOffset = (uint)stream.Position;
        var parameterOffset = (uint)stream.Position;
        var stringOffset = (uint)stream.Position;
        using (var stringStream = new MemoryStream())
        {
            using var stringWriter = new BinaryWriter(stringStream, System.Text.Encoding.UTF8, leaveOpen: true);
            stringWriter.Write((uint)3);
            stringWriter.Write((uint)5);
            stringWriter.Write("Entry"u8);
            stringWriter.Write((uint)1);
            stringWriter.Write("0"u8);
            stringWriter.Write((uint)13);
            stringWriter.Write("shared_values"u8);
            writer.Write(stringStream.ToArray());
        }

        var payload = stream.ToArray();
        using var headerStream = new MemoryStream(payload);
        using var header = new BinaryWriter(headerStream, System.Text.Encoding.UTF8, leaveOpen: true);
        header.Write("FTIR"u8);
        header.Write((ushort)1);
        header.Write((ushort)0);
        header.Write((byte)1);
        header.Write((byte)0);
        header.Write((ushort)headerSize);
        header.Write((uint)0);
        WriteRange(header, functionOffset, 1);
        WriteRange(header, typeOffset, 2);
        WriteRange(header, structOffset, 0);
        WriteRange(header, structFieldOffset, 0);
        WriteRange(header, statementOffset, 3);
        WriteRange(header, expressionOffset, 2);
        WriteRange(header, lvalueOffset, 1);
        WriteRange(header, childOffset, 2);
        WriteRange(header, argumentOffset, 0);
        WriteRange(header, parameterOffset, 0);
        header.Write(stringOffset);
        header.Write(checked((uint)(payload.Length - stringOffset)));
        return payload;

        static void WriteRange(BinaryWriter writer, uint offset, uint count)
        {
            writer.Write(offset);
            writer.Write(count);
        }
    }

    private static byte[] BuildTypedIrAtomicSection()
    {
        const int headerSize = 104;

        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        writer.Write(new byte[headerSize]);

        var functionOffset = (uint)stream.Position;
        writer.Write((byte)0); // Compute1D
        writer.Write((uint)0); // name
        writer.Write((uint)0); // mangled name
        writer.Write((uint)0); // void type
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);
        writer.Write((uint)0); // body statement

        var typeOffset = (uint)stream.Position;
        writer.Write((byte)7); // void
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((byte)1); // primitive int
        writer.Write((uint)1);
        writer.Write((uint)32);
        writer.Write((uint)0);
        writer.Write((uint)0);

        var structOffset = (uint)stream.Position;
        var structFieldOffset = (uint)stream.Position;

        var statementOffset = (uint)stream.Position;
        writer.Write((byte)1); // block
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0); // first child
        writer.Write((uint)1); // child count

        writer.Write((byte)12); // expression statement
        writer.Write((uint)2); // atomic expression
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);

        var expressionOffset = (uint)stream.Position;
        writer.Write((byte)1); // literal index
        writer.Write((uint)1); // int type id
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)1); // "0"
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);

        writer.Write((byte)1); // literal atomic value
        writer.Write((uint)1); // int type id
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)2); // "1"
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);

        writer.Write((byte)22); // atomic add
        writer.Write((uint)1); // int type id
        writer.Write((uint)0); // l-value id
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0); // add operation
        writer.Write((uint)0); // first argument
        writer.Write((uint)1); // one value operand

        var lvalueOffset = (uint)stream.Position;
        writer.Write((byte)4); // resource element l-value
        writer.Write((uint)1); // int type id
        writer.Write((uint)0); // index expression
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)3); // "output"

        var childOffset = (uint)stream.Position;
        writer.Write((uint)1);

        var argumentOffset = (uint)stream.Position;
        writer.Write((uint)1);

        var parameterOffset = (uint)stream.Position;
        var stringOffset = (uint)stream.Position;
        using (var stringStream = new MemoryStream())
        {
            using var stringWriter = new BinaryWriter(stringStream, System.Text.Encoding.UTF8, leaveOpen: true);
            stringWriter.Write((uint)4);
            stringWriter.Write((uint)5);
            stringWriter.Write("Entry"u8);
            stringWriter.Write((uint)1);
            stringWriter.Write("0"u8);
            stringWriter.Write((uint)1);
            stringWriter.Write("1"u8);
            stringWriter.Write((uint)6);
            stringWriter.Write("output"u8);
            writer.Write(stringStream.ToArray());
        }

        var payload = stream.ToArray();
        using var headerStream = new MemoryStream(payload);
        using var header = new BinaryWriter(headerStream, System.Text.Encoding.UTF8, leaveOpen: true);
        header.Write("FTIR"u8);
        header.Write((ushort)1);
        header.Write((ushort)0);
        header.Write((byte)1);
        header.Write((byte)0);
        header.Write((ushort)headerSize);
        header.Write((uint)0);
        WriteRange(header, functionOffset, 1);
        WriteRange(header, typeOffset, 2);
        WriteRange(header, structOffset, 0);
        WriteRange(header, structFieldOffset, 0);
        WriteRange(header, statementOffset, 2);
        WriteRange(header, expressionOffset, 3);
        WriteRange(header, lvalueOffset, 1);
        WriteRange(header, childOffset, 1);
        WriteRange(header, argumentOffset, 1);
        WriteRange(header, parameterOffset, 0);
        header.Write(stringOffset);
        header.Write(checked((uint)(payload.Length - stringOffset)));
        return payload;

        static void WriteRange(BinaryWriter writer, uint offset, uint count)
        {
            writer.Write(offset);
            writer.Write(count);
        }
    }

    private static byte[] BuildTypedIrTextureSampleSection(uint sampleOp = 0, uint argumentCount = 3)
    {
        const int headerSize = 104;
        if (argumentCount is < 3 or > 5)
        {
            throw new ArgumentOutOfRangeException(nameof(argumentCount));
        }

        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        writer.Write(new byte[headerSize]);

        var functionOffset = (uint)stream.Position;
        writer.Write((byte)0); // Compute1D
        writer.Write((uint)0); // name
        writer.Write((uint)0); // mangled name
        writer.Write((uint)0); // void type
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);
        writer.Write((uint)0); // body statement

        var typeOffset = (uint)stream.Position;
        writer.Write((byte)7); // void
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)0);

        var structOffset = (uint)stream.Position;
        var structFieldOffset = (uint)stream.Position;

        var statementOffset = (uint)stream.Position;
        writer.Write((byte)1); // block
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0); // first child
        writer.Write((uint)1); // child count

        writer.Write((byte)12); // expression statement
        writer.Write((uint)5); // texture sample expression
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);

        var expressionOffset = (uint)stream.Position;
        writer.Write((byte)3); // texture resource parameter
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)1); // "input"
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);

        writer.Write((byte)3); // sampler resource parameter
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)2); // "sampler"
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);

        writer.Write((byte)2); // uv local
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)3); // "uv"
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);

        writer.Write((byte)2); // ddx local
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)4); // "ddx"
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);

        writer.Write((byte)2); // ddy local
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)5); // "ddy"
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);

        writer.Write((byte)23); // texture sample
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(sampleOp);
        writer.Write((uint)0); // first argument
        writer.Write(argumentCount); // texture, sampler, uv, optional lod/ddx, optional ddy

        var lvalueOffset = (uint)stream.Position;
        var childOffset = (uint)stream.Position;
        writer.Write((uint)1);

        var argumentOffset = (uint)stream.Position;
        writer.Write((uint)0);
        writer.Write((uint)1);
        writer.Write((uint)2);
        if (argumentCount >= 4)
        {
            writer.Write((uint)3);
        }

        if (argumentCount >= 5)
        {
            writer.Write((uint)4);
        }

        var parameterOffset = (uint)stream.Position;
        var stringOffset = (uint)stream.Position;
        using (var stringStream = new MemoryStream())
        {
            using var stringWriter = new BinaryWriter(stringStream, System.Text.Encoding.UTF8, leaveOpen: true);
            stringWriter.Write((uint)6);
            stringWriter.Write((uint)5);
            stringWriter.Write("Entry"u8);
            stringWriter.Write((uint)5);
            stringWriter.Write("input"u8);
            stringWriter.Write((uint)7);
            stringWriter.Write("sampler"u8);
            stringWriter.Write((uint)2);
            stringWriter.Write("uv"u8);
            stringWriter.Write((uint)3);
            stringWriter.Write("ddx"u8);
            stringWriter.Write((uint)3);
            stringWriter.Write("ddy"u8);
            writer.Write(stringStream.ToArray());
        }

        var payload = stream.ToArray();
        using var headerStream = new MemoryStream(payload);
        using var header = new BinaryWriter(headerStream, System.Text.Encoding.UTF8, leaveOpen: true);
        header.Write("FTIR"u8);
        header.Write((ushort)1);
        header.Write((ushort)0);
        header.Write((byte)1);
        header.Write((byte)0);
        header.Write((ushort)headerSize);
        header.Write((uint)0);
        WriteRange(header, functionOffset, 1);
        WriteRange(header, typeOffset, 1);
        WriteRange(header, structOffset, 0);
        WriteRange(header, structFieldOffset, 0);
        WriteRange(header, statementOffset, 2);
        WriteRange(header, expressionOffset, 6);
        WriteRange(header, lvalueOffset, 0);
        WriteRange(header, childOffset, 1);
        WriteRange(header, argumentOffset, argumentCount);
        WriteRange(header, parameterOffset, 0);
        header.Write(stringOffset);
        header.Write(checked((uint)(payload.Length - stringOffset)));
        return payload;

        static void WriteRange(BinaryWriter writer, uint offset, uint count)
        {
            writer.Write(offset);
            writer.Write(count);
        }
    }

    private static uint ReadUInt32(byte[] bytes, int offset)
        => BitConverter.ToUInt32(bytes, offset);

    private static void WriteUInt32(byte[] bytes, int offset, uint value)
        => BitConverter.GetBytes(value).CopyTo(bytes, offset);

    private static void WriteUInt16(byte[] bytes, int offset, ushort value)
        => BitConverter.GetBytes(value).CopyTo(bytes, offset);

    private static void MutateTypedIrSection(byte[] section, TypedIrMutation mutation)
    {
        switch (mutation)
        {
            case TypedIrMutation.BadMagic:
                section[0] = (byte)'B';
                break;
            case TypedIrMutation.UnsupportedMajorVersion:
                WriteUInt16(section, 4, 2);
                break;
            case TypedIrMutation.UnsupportedMinorVersion:
                WriteUInt16(section, 6, 2);
                break;
            case TypedIrMutation.WrongEndian:
                section[8] = 2;
                break;
            case TypedIrMutation.WrongHeaderSize:
                WriteUInt16(section, 10, 96);
                break;
            case TypedIrMutation.OutOfRangeEntryFunction:
                WriteUInt32(section, 12, 99);
                break;
            case TypedIrMutation.ZeroFunctionCount:
                WriteUInt32(section, 20, 0);
                break;
            case TypedIrMutation.ZeroTypeCount:
                WriteUInt32(section, 28, 0);
                break;
            case TypedIrMutation.ZeroStatementCount:
                WriteUInt32(section, 52, 0);
                break;
            case TypedIrMutation.OverlappingTypeTable:
                WriteUInt32(section, 24, ReadUInt32(section, 16));
                break;
            case TypedIrMutation.StringTableGap:
                WriteUInt32(section, 96, ReadUInt32(section, 96) + 4);
                break;
            case TypedIrMutation.StringTableTooLong:
                WriteUInt32(section, 100, ReadUInt32(section, 100) + 4);
                break;
            case TypedIrMutation.TruncatedStringBytes:
            {
                var stringOffset = ReadUInt32(section, 96);
                WriteUInt32(section, checked((int)stringOffset + 4), 999);
                break;
            }
            case TypedIrMutation.FunctionBadName:
            {
                var functionOffset = ReadUInt32(section, 16);
                WriteUInt32(section, checked((int)functionOffset + 1), 999);
                break;
            }
            case TypedIrMutation.FunctionBadReturnType:
            {
                var functionOffset = ReadUInt32(section, 16);
                WriteUInt32(section, checked((int)functionOffset + 9), 999);
                break;
            }
            case TypedIrMutation.FunctionBadParameterSpan:
            {
                var functionOffset = ReadUInt32(section, 16);
                WriteUInt32(section, checked((int)functionOffset + 13), 999);
                WriteUInt32(section, checked((int)functionOffset + 17), 1);
                break;
            }
            case TypedIrMutation.FunctionBadBody:
            {
                var functionOffset = ReadUInt32(section, 16);
                WriteUInt32(section, checked((int)functionOffset + 21), 999);
                break;
            }
            case TypedIrMutation.TypeBadKind:
            {
                var typeOffset = ReadUInt32(section, 24);
                section[typeOffset] = 99;
                break;
            }
            case TypedIrMutation.StatementBadKind:
            {
                var statementOffset = ReadUInt32(section, 48);
                section[statementOffset] = 99;
                break;
            }
            case TypedIrMutation.StatementBadChildSpan:
            {
                var statementOffset = ReadUInt32(section, 48);
                WriteUInt32(section, checked((int)statementOffset + 21), 99);
                WriteUInt32(section, checked((int)statementOffset + 25), 1);
                break;
            }
            case TypedIrMutation.ExpressionBadKind:
            {
                var expressionOffset = ReadUInt32(section, 56);
                section[expressionOffset] = 99;
                break;
            }
            case TypedIrMutation.ExpressionBadArgumentSpan:
            {
                var expressionOffset = ReadUInt32(section, 56);
                WriteUInt32(section, checked((int)expressionOffset + 25), 99);
                WriteUInt32(section, checked((int)expressionOffset + 29), 1);
                break;
            }
            case TypedIrMutation.ExpressionUnexpectedArgumentSpan:
            {
                var expressionOffset = ReadUInt32(section, 56);
                var literalExpressionOffset = expressionOffset + 33;
                WriteUInt32(section, checked((int)literalExpressionOffset + 25), 0);
                WriteUInt32(section, checked((int)literalExpressionOffset + 29), 1);
                break;
            }
            case TypedIrMutation.ArgumentBadExpressionIndex:
            {
                var argumentOffset = ReadUInt32(section, 80);
                WriteUInt32(section, checked((int)argumentOffset), 99);
                break;
            }
            case TypedIrMutation.ParameterBadDirection:
            {
                var parameterOffset = ReadUInt32(section, 88);
                section[parameterOffset] = 99;
                break;
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(mutation), mutation, null);
        }
    }

    private static void WriteExpressionNode(
        BinaryWriter writer,
        byte kind,
        byte operation,
        uint resourceBinding,
        uint indexStringId,
        uint literalStringId,
        uint typeStringId,
        uint leftNodeIndex = uint.MaxValue,
        uint rightNodeIndex = uint.MaxValue)
    {
        writer.Write(kind);
        writer.Write(operation);
        writer.Write((ushort)0);
        writer.Write(resourceBinding);
        writer.Write(indexStringId);
        writer.Write(literalStringId);
        writer.Write(typeStringId);
        writer.Write(leftNodeIndex);
        writer.Write(rightNodeIndex);
    }

    private readonly record struct IrInstruction(byte Opcode, byte OperandKind, uint OperandStringId);

    private readonly record struct IrSection(uint Kind, byte[] Payload);

    private readonly record struct IrElementwiseAssignment(
        uint InstructionIndex,
        uint DestinationBinding,
        uint LeftBinding,
        uint RightBinding,
        byte Operation,
        byte RightOperandKind,
        uint IndexStringId,
        uint RightLiteralStringId);

    public enum TypedIrMutation
    {
        BadMagic,
        UnsupportedMajorVersion,
        UnsupportedMinorVersion,
        WrongEndian,
        WrongHeaderSize,
        OutOfRangeEntryFunction,
        ZeroFunctionCount,
        ZeroTypeCount,
        ZeroStatementCount,
        OverlappingTypeTable,
        StringTableGap,
        StringTableTooLong,
        TruncatedStringBytes,
        FunctionBadName,
        FunctionBadReturnType,
        FunctionBadParameterSpan,
        FunctionBadBody,
        TypeBadKind,
        StatementBadKind,
        StatementBadChildSpan,
        ExpressionBadKind,
        ExpressionBadArgumentSpan,
        ExpressionUnexpectedArgumentSpan,
        ArgumentBadExpressionIndex,
        ParameterBadDirection
    }
}
