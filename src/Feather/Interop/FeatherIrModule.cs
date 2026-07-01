using System.Buffers.Binary;
using System.Text;
using Feather.Math;

namespace Feather.Interop;

/// <summary>
/// Describes a deserialized Feather IR module.
/// </summary>
/// <param name="MajorVersion">The major binary format version.</param>
/// <param name="MinorVersion">The minor binary format version.</param>
/// <param name="ShaderKind">The shader stage represented by the module.</param>
/// <param name="ThreadGroupSize">The compute thread group size, or the default stage value for graphics shaders.</param>
/// <param name="EntryPoint">The generated entry point name.</param>
/// <param name="PushConstantCount">The number of push-constant resources in the resource table.</param>
/// <param name="Resources">The resources declared by the module.</param>
/// <param name="Instructions">The serialized instruction stream.</param>
/// <param name="ElementwiseAssignments">The structured elementwise assignment records emitted by semantic lowering.</param>
/// <param name="ElementwiseExpressionAssignments">The typed elementwise expression assignments emitted by semantic lowering.</param>
/// <param name="AdAnnotations">The AD parameter/loss metadata records emitted by semantic lowering.</param>
/// <param name="Strings">The decoded string table.</param>
public sealed record FeatherIrModule(
    ushort MajorVersion,
    ushort MinorVersion,
    FeatherIrShaderKind ShaderKind,
    int3 ThreadGroupSize,
    string EntryPoint,
    uint PushConstantCount,
    IReadOnlyList<FeatherIrResource> Resources,
    IReadOnlyList<FeatherIrInstruction> Instructions,
    IReadOnlyList<FeatherIrElementwiseAssignment> ElementwiseAssignments,
    IReadOnlyList<FeatherIrElementwiseExpressionAssignment> ElementwiseExpressionAssignments,
    IReadOnlyList<FeatherIrAdAnnotation> AdAnnotations,
    IReadOnlyList<string> Strings);

/// <summary>
/// Describes one resource declared by a Feather IR module.
/// </summary>
/// <param name="Binding">The zero-based binding slot.</param>
/// <param name="Kind">The resource kind.</param>
/// <param name="Access">The resource access mode.</param>
/// <param name="Name">The source-level resource name.</param>
/// <param name="ElementType">The normalized C# element type name.</param>
public sealed record FeatherIrResource(uint Binding, FeatherIrResourceKind Kind, FeatherIrResourceAccess Access, string Name, string ElementType);

/// <summary>
/// Describes one serialized Feather IR instruction.
/// </summary>
/// <param name="Opcode">The instruction opcode.</param>
/// <param name="OperandKind">The operand encoding kind.</param>
/// <param name="Operand">The decoded operand text, if present.</param>
public sealed record FeatherIrInstruction(FeatherIrInstructionOpcode Opcode, FeatherIrOperandKind OperandKind, string Operand);

/// <summary>
/// Describes a structured elementwise resource assignment.
/// </summary>
/// <param name="InstructionIndex">The instruction stream index for the corresponding assignment instruction.</param>
/// <param name="DestinationBinding">The destination resource binding.</param>
/// <param name="LeftBinding">The left source resource binding.</param>
/// <param name="RightBinding">The right source resource binding when <paramref name="RightOperandKind"/> is <see cref="FeatherIrAssignmentOperandKind.Resource"/>.</param>
/// <param name="Operation">The elementwise operation.</param>
/// <param name="RightOperandKind">The right operand kind.</param>
/// <param name="Index">The source-level index symbol.</param>
/// <param name="RightLiteral">The canonical literal value when <paramref name="RightOperandKind"/> is <see cref="FeatherIrAssignmentOperandKind.Literal"/>.</param>
public sealed record FeatherIrElementwiseAssignment(
    uint InstructionIndex,
    uint DestinationBinding,
    uint LeftBinding,
    uint RightBinding,
    FeatherIrAssignmentOperation Operation,
    FeatherIrAssignmentOperandKind RightOperandKind,
    string Index,
    string RightLiteral);

/// <summary>
/// Describes an elementwise assignment whose right-hand side is a typed expression tree.
/// </summary>
/// <param name="InstructionIndex">The instruction stream index for the corresponding assignment instruction.</param>
/// <param name="DestinationBinding">The destination resource binding.</param>
/// <param name="Index">The source-level index symbol shared by the elementwise expression.</param>
/// <param name="Expression">The root expression node.</param>
public sealed record FeatherIrElementwiseExpressionAssignment(
    uint InstructionIndex,
    uint DestinationBinding,
    string Index,
    FeatherIrExpressionNode Expression);

/// <summary>
/// Describes an automatic differentiation marker record.
/// </summary>
public sealed record FeatherIrAdAnnotation(
    FeatherIrAdAnnotationRole Role,
    uint Binding,
    string Name,
    string ResourceName,
    string TypeName,
    string Index,
    FeatherIrAdSourceKind SourceKind,
    uint ElementCount);

/// <summary>
/// Identifies the AD marker role.
/// </summary>
public enum FeatherIrAdAnnotationRole : uint
{
    Parameter = 0,
    Loss = 1
}

/// <summary>
/// Identifies where the AD marker source value came from.
/// </summary>
public enum FeatherIrAdSourceKind : uint
{
    Unknown = 0,
    BufferElement = 1,
    Local = 2
}

/// <summary>
/// Describes one node in a typed elementwise expression tree.
/// </summary>
/// <param name="Kind">The expression node kind.</param>
/// <param name="Operation">The operation for binary nodes.</param>
/// <param name="ResourceBinding">The resource binding for resource nodes.</param>
/// <param name="Index">The source-level index symbol for resource nodes.</param>
/// <param name="Literal">The canonical literal for literal nodes.</param>
/// <param name="TypeName">The Roslyn result type name for the node.</param>
/// <param name="Left">The left child for binary nodes.</param>
/// <param name="Right">The right child for binary nodes.</param>
/// <param name="Symbol">The fully qualified Roslyn method symbol for invocation nodes.</param>
/// <param name="Arguments">The argument child expressions for invocation nodes.</param>
public sealed record FeatherIrExpressionNode(
    FeatherIrExpressionNodeKind Kind,
    FeatherIrExpressionOperation Operation,
    uint ResourceBinding,
    string Index,
    string Literal,
    string TypeName,
    FeatherIrExpressionNode? Left,
    FeatherIrExpressionNode? Right,
    string Symbol,
    IReadOnlyList<FeatherIrExpressionNode> Arguments);

/// <summary>
/// Identifies a typed elementwise expression node kind.
/// </summary>
public enum FeatherIrExpressionNodeKind : byte
{
    /// <summary>
    /// The node kind is invalid or unspecified.
    /// </summary>
    Unknown = 0,

    /// <summary>
    /// The node reads an element from a bound resource.
    /// </summary>
    Resource = 1,

    /// <summary>
    /// The node stores a canonical numeric literal.
    /// </summary>
    Literal = 2,

    /// <summary>
    /// The node combines two child expressions.
    /// </summary>
    Binary = 3,

    /// <summary>
    /// The node calls a supported shader intrinsic by Roslyn method symbol.
    /// </summary>
    Invocation = 4,

    /// <summary>
    /// The node compares two child expressions.
    /// </summary>
    Comparison = 6,

    /// <summary>
    /// The node reads a push-constant resource by binding.
    /// </summary>
    PushConstant = 5
}

/// <summary>
/// Identifies a typed elementwise expression operation.
/// </summary>
public enum FeatherIrExpressionOperation : byte
{
    /// <summary>
    /// No operation is used.
    /// </summary>
    None = 0,

    /// <summary>
    /// Adds two child values.
    /// </summary>
    Add = 1,

    /// <summary>
    /// Subtracts the right child from the left child.
    /// </summary>
    Subtract = 2,

    /// <summary>
    /// Multiplies two child values.
    /// </summary>
    Multiply = 3,

    /// <summary>
    /// Divides the left child by the right child.
    /// </summary>
    Divide = 4,

    /// <summary>
    /// Compares two child values for equality.
    /// </summary>
    Equal = 5,

    /// <summary>
    /// Compares two child values for inequality.
    /// </summary>
    NotEqual = 6,

    /// <summary>
    /// Checks if the left child is greater than the right child.
    /// </summary>
    Greater = 7,

    /// <summary>
    /// Checks if the left child is less than the right child.
    /// </summary>
    Less = 8,

    /// <summary>
    /// Checks if the left child is greater than or equal to the right child.
    /// </summary>
    GreaterEqual = 9,

    /// <summary>
    /// Checks if the left child is less than or equal to the right child.
    /// </summary>
    LessEqual = 10
}

/// <summary>
/// Identifies the operation used by a structured elementwise assignment.
/// </summary>
public enum FeatherIrAssignmentOperation : byte
{
    /// <summary>
    /// Copies the left resource element.
    /// </summary>
    Copy = 1,

    /// <summary>
    /// Adds the left and right operands.
    /// </summary>
    Add = 2,

    /// <summary>
    /// Subtracts the right operand from the left operand.
    /// </summary>
    Subtract = 3,

    /// <summary>
    /// Multiplies the left and right operands.
    /// </summary>
    Multiply = 4,

    /// <summary>
    /// Divides the left operand by the right operand.
    /// </summary>
    Divide = 5
}

/// <summary>
/// Identifies how a structured elementwise assignment stores its right operand.
/// </summary>
public enum FeatherIrAssignmentOperandKind : byte
{
    /// <summary>
    /// No right operand is used.
    /// </summary>
    None = 0,

    /// <summary>
    /// The right operand is another resource binding.
    /// </summary>
    Resource = 1,

    /// <summary>
    /// The right operand is a canonical literal string.
    /// </summary>
    Literal = 2
}

/// <summary>
/// Identifies the shader stage encoded in a Feather IR module.
/// </summary>
public enum FeatherIrShaderKind : byte
{
    /// <summary>
    /// Unknown or invalid shader kind.
    /// </summary>
    Unknown = 0,

    /// <summary>
    /// One-dimensional compute kernel.
    /// </summary>
    Compute1D = 1,

    /// <summary>
    /// Two-dimensional compute kernel.
    /// </summary>
    Compute2D = 2,

    /// <summary>
    /// Three-dimensional compute kernel.
    /// </summary>
    Compute3D = 3,

    /// <summary>
    /// Vertex shader.
    /// </summary>
    Vertex = 4,

    /// <summary>
    /// Fragment shader.
    /// </summary>
    Fragment = 5
}

/// <summary>
/// Identifies a resource table entry kind.
/// </summary>
public enum FeatherIrResourceKind : byte
{
    /// <summary>
    /// Unknown or invalid resource kind.
    /// </summary>
    Unknown = 0,

    /// <summary>
    /// Buffer resource.
    /// </summary>
    Buffer = 1,

    /// <summary>
    /// Two-dimensional texture resource.
    /// </summary>
    Texture2D = 2,

    /// <summary>
    /// Sampler resource.
    /// </summary>
    Sampler = 3,

    /// <summary>
    /// Uniform constructor resource.
    /// </summary>
    Uniform = 4,

    /// <summary>
    /// Generated push-constant resource.
    /// </summary>
    PushConstant = 5,

    /// <summary>
    /// Three-dimensional texture resource.
    /// </summary>
    Texture3D = 6
}

/// <summary>
/// Identifies the access mode for a resource table entry.
/// </summary>
public enum FeatherIrResourceAccess : byte
{
    /// <summary>
    /// Unknown or invalid access mode.
    /// </summary>
    Unknown = 0,

    /// <summary>
    /// Read-only access.
    /// </summary>
    Read = 1,

    /// <summary>
    /// Write-only access.
    /// </summary>
    Write = 2,

    /// <summary>
    /// Read-write access.
    /// </summary>
    ReadWrite = 3,

    /// <summary>
    /// Sampled texture access.
    /// </summary>
    Sample = 4
}

/// <summary>
/// Identifies a serialized Feather IR instruction.
/// </summary>
public enum FeatherIrInstructionOpcode : byte
{
    /// <summary>
    /// Unknown or invalid instruction.
    /// </summary>
    Unknown = 0,

    /// <summary>
    /// Local variable declaration.
    /// </summary>
    LocalDeclaration = 1,

    /// <summary>
    /// Assignment expression.
    /// </summary>
    Assignment = 2,

    /// <summary>
    /// Return statement.
    /// </summary>
    Return = 3,

    /// <summary>
    /// If branch condition.
    /// </summary>
    If = 4,

    /// <summary>
    /// For loop header.
    /// </summary>
    For = 5,

    /// <summary>
    /// While loop condition.
    /// </summary>
    While = 6,

    /// <summary>
    /// Do-while loop condition.
    /// </summary>
    Do = 7,

    /// <summary>
    /// Break statement.
    /// </summary>
    Break = 8,

    /// <summary>
    /// Continue statement.
    /// </summary>
    Continue = 9,

    /// <summary>
    /// Function or intrinsic invocation.
    /// </summary>
    Invocation = 10,

    /// <summary>
    /// Resource element access.
    /// </summary>
    ResourceAccess = 11,

    /// <summary>
    /// General expression or statement placeholder.
    /// </summary>
    Expression = 12,

    /// <summary>
    /// Begins a structured control-flow block.
    /// </summary>
    BeginBlock = 13,

    /// <summary>
    /// Separates an if true block from its else block.
    /// </summary>
    Else = 14,

    /// <summary>
    /// Ends a structured control-flow block.
    /// </summary>
    EndBlock = 15,

    /// <summary>
    /// Workgroup synchronization barrier.
    /// </summary>
    WorkgroupBarrier = 16,

    /// <summary>
    /// Memory visibility barrier.
    /// </summary>
    MemoryBarrier = 17,

    /// <summary>
    /// Combined memory and workgroup synchronization barrier.
    /// </summary>
    FullBarrier = 18,

    /// <summary>
    /// Atomic add operation.
    /// </summary>
    AtomicAdd = 19,

    /// <summary>
    /// Atomic subtract operation.
    /// </summary>
    AtomicSub = 20,

    /// <summary>
    /// Atomic minimum operation.
    /// </summary>
    AtomicMin = 21,

    /// <summary>
    /// Atomic maximum operation.
    /// </summary>
    AtomicMax = 22,

    /// <summary>
    /// Atomic bitwise AND operation.
    /// </summary>
    AtomicAnd = 23,

    /// <summary>
    /// Atomic bitwise OR operation.
    /// </summary>
    AtomicOr = 24,

    /// <summary>
    /// Atomic bitwise XOR operation.
    /// </summary>
    AtomicXor = 25,

    /// <summary>
    /// Atomic exchange operation.
    /// </summary>
    AtomicExchange = 26,

    /// <summary>
    /// Atomic compare-exchange operation.
    /// </summary>
    AtomicCompareExchange = 27,

    /// <summary>
    /// Workgroup shared-memory declaration.
    /// </summary>
    SharedMemoryDeclaration = 28
}

/// <summary>
/// Identifies how an instruction operand is encoded.
/// </summary>
public enum FeatherIrOperandKind : byte
{
    /// <summary>
    /// The instruction has no operand.
    /// </summary>
    None = 0,

    /// <summary>
    /// The operand is normalized source text stored in the string table.
    /// </summary>
    Source = 1,

    /// <summary>
    /// The operand is a canonical elementwise assignment payload emitted from Roslyn syntax.
    /// </summary>
    ElementwiseAssignment = 2,

    /// <summary>
    /// The operand is a fully qualified Roslyn symbol identity.
    /// </summary>
    Symbol = 3
}

/// <summary>
/// Reads the baseline Feather binary IR format.
/// </summary>
public static class FeatherIr
{
    private const uint ElementwiseAssignmentSectionKind = 1;
    private const uint ElementwiseExpressionAssignmentSectionKind = 2;
    private const uint AdAnnotationSectionKind = 4;
    private const uint NoString = uint.MaxValue;

    /// <summary>
    /// Reads a Feather IR module from binary data.
    /// </summary>
    /// <param name="data">The binary IR payload.</param>
    /// <returns>The decoded IR module.</returns>
    public static FeatherIrModule Read(ReadOnlySpan<byte> data)
    {
        var reader = new Reader(data);
        if (reader.ReadAscii(4) != "FEIR")
        {
            throw new InvalidDataException("Feather IR magic mismatch.");
        }

        var major = reader.ReadUInt16();
        var minor = reader.ReadUInt16();
        var endian = reader.ReadByte();
        if (endian != 1)
        {
            throw new InvalidDataException("Only little-endian Feather IR is supported.");
        }

        var shaderKind = (FeatherIrShaderKind)reader.ReadByte();
        var sectionCount = reader.ReadUInt16();
        var group = new int3(reader.ReadInt32(), reader.ReadInt32(), reader.ReadInt32());
        var entryPointId = reader.ReadUInt32();
        var resourceCount = reader.ReadUInt32();
        var pushConstantCount = reader.ReadUInt32();
        var instructionCount = reader.ReadUInt32();
        var stringByteLength = reader.ReadUInt32();

        var rawResources = new List<RawResource>((int)resourceCount);
        for (var i = 0; i < resourceCount; i++)
        {
            rawResources.Add(new RawResource(
                reader.ReadUInt32(),
                (FeatherIrResourceKind)reader.ReadByte(),
                (FeatherIrResourceAccess)reader.ReadByte(),
                reader.ReadByte(),
                reader.ReadUInt32(),
                reader.ReadUInt32()));
        }

        var rawInstructions = new List<RawInstruction>((int)instructionCount);
        for (var i = 0; i < instructionCount; i++)
        {
            rawInstructions.Add(new RawInstruction(
                (FeatherIrInstructionOpcode)reader.ReadByte(),
                (FeatherIrOperandKind)reader.ReadByte(),
                reader.ReadUInt16(),
                reader.ReadUInt32()));
        }

        var sections = new List<RawSection>((int)sectionCount);
        for (var i = 0; i < sectionCount; i++)
        {
            sections.Add(new RawSection(
                reader.ReadUInt32(),
                reader.ReadUInt32()));
        }

        var rawAssignments = Array.Empty<RawElementwiseAssignment>();
        var rawExpressionAssignments = new RawElementwiseExpressionSection([], [], []);
        var rawAdAnnotations = Array.Empty<RawAdAnnotation>();
        foreach (var section in sections)
        {
            if (section.Kind == ElementwiseAssignmentSectionKind)
            {
                rawAssignments = ReadElementwiseAssignments(reader.ReadBytes(checked((int)section.ByteLength)));
                continue;
            }

            if (section.Kind == ElementwiseExpressionAssignmentSectionKind)
            {
                rawExpressionAssignments = ReadElementwiseExpressionAssignments(reader.ReadBytes(checked((int)section.ByteLength)));
                continue;
            }

            if (section.Kind == AdAnnotationSectionKind)
            {
                rawAdAnnotations = ReadAdAnnotations(reader.ReadBytes(checked((int)section.ByteLength)));
                continue;
            }

            // Unknown sections are skipped so older inspection tools can still read
            // the stable header, resources, and instruction stream of newer minor versions.
            reader.ReadBytes(checked((int)section.ByteLength));
        }

        var strings = ReadStrings(reader.ReadBytes(checked((int)stringByteLength)));
        var resources = rawResources
            .Select(resource => new FeatherIrResource(
                resource.Binding,
                resource.Kind,
                resource.Access,
                GetString(strings, resource.NameStringId),
                GetString(strings, resource.ElementTypeStringId)))
            .ToArray();
        var instructions = rawInstructions
            .Select(instruction => new FeatherIrInstruction(
                instruction.Opcode,
                instruction.OperandKind,
                instruction.OperandKind == FeatherIrOperandKind.None ? string.Empty : GetString(strings, instruction.OperandStringId)))
            .ToArray();
        var assignments = rawAssignments
            .Select(assignment => new FeatherIrElementwiseAssignment(
                assignment.InstructionIndex,
                assignment.DestinationBinding,
                assignment.LeftBinding,
                assignment.RightBinding,
                assignment.Operation,
                assignment.RightOperandKind,
                GetString(strings, assignment.IndexStringId),
                assignment.RightLiteralStringId == uint.MaxValue ? string.Empty : GetString(strings, assignment.RightLiteralStringId)))
            .ToArray();
        var expressionAssignments = rawExpressionAssignments.Assignments
            .Select(assignment => new FeatherIrElementwiseExpressionAssignment(
                assignment.InstructionIndex,
                assignment.DestinationBinding,
                GetString(strings, assignment.IndexStringId),
                BuildExpressionNode(rawExpressionAssignments.Nodes, rawExpressionAssignments.ArgumentIndices, strings, assignment.RootNodeIndex)))
            .ToArray();
        var adAnnotations = rawAdAnnotations
            .Select(annotation => new FeatherIrAdAnnotation(
                annotation.Role,
                annotation.Binding,
                annotation.NameStringId == NoString ? string.Empty : GetString(strings, annotation.NameStringId),
                annotation.ResourceNameStringId == NoString ? string.Empty : GetString(strings, annotation.ResourceNameStringId),
                annotation.TypeNameStringId == NoString ? string.Empty : GetString(strings, annotation.TypeNameStringId),
                annotation.IndexStringId == NoString ? string.Empty : GetString(strings, annotation.IndexStringId),
                annotation.SourceKind,
                annotation.ElementCount))
            .ToArray();

        return new FeatherIrModule(major, minor, shaderKind, group, GetString(strings, entryPointId), pushConstantCount, resources, instructions, assignments, expressionAssignments, adAnnotations, strings);
    }

    private static RawElementwiseAssignment[] ReadElementwiseAssignments(ReadOnlySpan<byte> data)
    {
        var reader = new Reader(data);
        var count = reader.ReadUInt32();
        var assignments = new RawElementwiseAssignment[count];
        for (var i = 0; i < assignments.Length; i++)
        {
            assignments[i] = new RawElementwiseAssignment(
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                (FeatherIrAssignmentOperation)reader.ReadByte(),
                (FeatherIrAssignmentOperandKind)reader.ReadByte(),
                reader.ReadUInt16(),
                reader.ReadUInt32(),
                reader.ReadUInt32());
        }

        return assignments;
    }

    private static RawElementwiseExpressionSection ReadElementwiseExpressionAssignments(ReadOnlySpan<byte> data)
    {
        var reader = new Reader(data);
        var assignmentCount = reader.ReadUInt32();
        var nodeCount = reader.ReadUInt32();
        var legacyPayloadLength = checked(8 + ((int)assignmentCount * 16) + ((int)nodeCount * 28));
        var hasArgumentIndexTable = data.Length != legacyPayloadLength;
        var argumentIndexCount = hasArgumentIndexTable ? reader.ReadUInt32() : 0u;
        var assignments = new RawElementwiseExpressionAssignment[assignmentCount];
        for (var i = 0; i < assignments.Length; i++)
        {
            assignments[i] = new RawElementwiseExpressionAssignment(
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                reader.ReadUInt32());
        }

        var nodes = new RawExpressionNode[nodeCount];
        for (var i = 0; i < nodes.Length; i++)
        {
            nodes[i] = new RawExpressionNode(
                (FeatherIrExpressionNodeKind)reader.ReadByte(),
                (FeatherIrExpressionOperation)reader.ReadByte(),
                reader.ReadUInt16(),
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                hasArgumentIndexTable ? reader.ReadUInt32() : uint.MaxValue,
                hasArgumentIndexTable ? reader.ReadUInt32() : uint.MaxValue,
                hasArgumentIndexTable ? reader.ReadUInt32() : 0u);
        }

        var argumentIndices = new uint[argumentIndexCount];
        for (var i = 0; i < argumentIndices.Length; i++)
        {
            argumentIndices[i] = reader.ReadUInt32();
        }

        return new RawElementwiseExpressionSection(assignments, nodes, argumentIndices);
    }

    private static RawAdAnnotation[] ReadAdAnnotations(ReadOnlySpan<byte> data)
    {
        var reader = new Reader(data);
        if (data.Length < 8)
        {
            throw new InvalidDataException("AD annotation section is too small.");
        }

        var first = reader.ReadUInt32();
        if ((first & 0xFFFFu) == 2u)
        {
            var parameterCount = reader.ReadUInt32();
            var lossCount = reader.ReadUInt32();
            var records = new RawAdAnnotation[checked((int)(parameterCount + lossCount))];
            for (var i = 0; i < parameterCount; i++)
            {
                records[i] = ReadAdRecord(ref reader, FeatherIrAdAnnotationRole.Parameter);
            }

            for (var i = 0; i < lossCount; i++)
            {
                records[parameterCount + i] = ReadAdRecord(ref reader, FeatherIrAdAnnotationRole.Loss);
            }

            return records;
        }

        var legacyParameterCount = first;
        var legacyLossCount = reader.ReadUInt32();
        var legacyRecords = new RawAdAnnotation[checked((int)(legacyParameterCount + legacyLossCount))];
        for (var i = 0; i < legacyParameterCount; i++)
        {
            var binding = reader.ReadUInt32();
            legacyRecords[i] = new RawAdAnnotation(
                FeatherIrAdAnnotationRole.Parameter,
                binding,
                NoString,
                NoString,
                NoString,
                NoString,
                FeatherIrAdSourceKind.Unknown,
                0);
        }

        for (var i = 0; i < legacyLossCount; i++)
        {
            var binding = reader.ReadUInt32();
            legacyRecords[legacyParameterCount + i] = new RawAdAnnotation(
                FeatherIrAdAnnotationRole.Loss,
                binding,
                NoString,
                NoString,
                NoString,
                NoString,
                FeatherIrAdSourceKind.Unknown,
                0);
        }

        return legacyRecords;
    }

    private static RawAdAnnotation ReadAdRecord(ref Reader reader, FeatherIrAdAnnotationRole expectedRole)
    {
        var role = (FeatherIrAdAnnotationRole)reader.ReadUInt32();
        if (role != expectedRole)
        {
            throw new InvalidDataException("AD annotation records are not grouped by role.");
        }

        return new RawAdAnnotation(
            role,
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            (FeatherIrAdSourceKind)reader.ReadUInt32(),
            reader.ReadUInt32());
    }

    private static FeatherIrExpressionNode BuildExpressionNode(
        IReadOnlyList<RawExpressionNode> nodes,
        IReadOnlyList<uint> argumentIndices,
        IReadOnlyList<string> strings,
        uint nodeIndex)
    {
        if (nodeIndex >= nodes.Count)
        {
            throw new InvalidDataException($"Expression node id {nodeIndex} is out of range.");
        }

        var node = nodes[(int)nodeIndex];
        return new FeatherIrExpressionNode(
            node.Kind,
            node.Operation,
            node.ResourceBinding,
            node.IndexStringId == uint.MaxValue ? string.Empty : GetString(strings, node.IndexStringId),
            node.LiteralStringId == uint.MaxValue ? string.Empty : GetString(strings, node.LiteralStringId),
            node.TypeStringId == uint.MaxValue ? string.Empty : GetString(strings, node.TypeStringId),
            node.LeftNodeIndex == uint.MaxValue ? null : BuildExpressionNode(nodes, argumentIndices, strings, node.LeftNodeIndex),
            node.RightNodeIndex == uint.MaxValue ? null : BuildExpressionNode(nodes, argumentIndices, strings, node.RightNodeIndex),
            node.SymbolStringId == uint.MaxValue ? string.Empty : GetString(strings, node.SymbolStringId),
            BuildExpressionArguments(nodes, argumentIndices, strings, node.FirstArgumentIndex, node.ArgumentCount));
    }

    private static IReadOnlyList<FeatherIrExpressionNode> BuildExpressionArguments(
        IReadOnlyList<RawExpressionNode> nodes,
        IReadOnlyList<uint> argumentIndices,
        IReadOnlyList<string> strings,
        uint firstArgumentIndex,
        uint argumentCount)
    {
        if (argumentCount == 0)
        {
            return [];
        }

        if (firstArgumentIndex == uint.MaxValue || firstArgumentIndex + argumentCount > argumentIndices.Count)
        {
            throw new InvalidDataException("Expression invocation argument list is out of range.");
        }

        var arguments = new FeatherIrExpressionNode[argumentCount];
        for (var i = 0; i < arguments.Length; i++)
        {
            arguments[i] = BuildExpressionNode(nodes, argumentIndices, strings, argumentIndices[(int)firstArgumentIndex + i]);
        }

        return arguments;
    }

    private static IReadOnlyList<string> ReadStrings(ReadOnlySpan<byte> data)
    {
        var reader = new Reader(data);
        var count = reader.ReadUInt32();
        var strings = new string[count];
        for (var i = 0; i < strings.Length; i++)
        {
            var length = reader.ReadUInt32();
            strings[i] = Encoding.UTF8.GetString(reader.ReadBytes(checked((int)length)));
        }

        return strings;
    }

    private static string GetString(IReadOnlyList<string> strings, uint id)
        => id < strings.Count ? strings[(int)id] : throw new InvalidDataException($"String id {id} is out of range.");

    private readonly record struct RawResource(uint Binding, FeatherIrResourceKind Kind, FeatherIrResourceAccess Access, byte Reserved, uint NameStringId, uint ElementTypeStringId);

    private readonly record struct RawInstruction(FeatherIrInstructionOpcode Opcode, FeatherIrOperandKind OperandKind, ushort Reserved, uint OperandStringId);

    private readonly record struct RawSection(uint Kind, uint ByteLength);

    private readonly record struct RawElementwiseAssignment(
        uint InstructionIndex,
        uint DestinationBinding,
        uint LeftBinding,
        uint RightBinding,
        FeatherIrAssignmentOperation Operation,
        FeatherIrAssignmentOperandKind RightOperandKind,
        ushort Reserved,
        uint IndexStringId,
        uint RightLiteralStringId);

    private readonly record struct RawElementwiseExpressionSection(
        IReadOnlyList<RawElementwiseExpressionAssignment> Assignments,
        IReadOnlyList<RawExpressionNode> Nodes,
        IReadOnlyList<uint> ArgumentIndices);

    private readonly record struct RawElementwiseExpressionAssignment(
        uint InstructionIndex,
        uint DestinationBinding,
        uint IndexStringId,
        uint RootNodeIndex);

    private readonly record struct RawExpressionNode(
        FeatherIrExpressionNodeKind Kind,
        FeatherIrExpressionOperation Operation,
        ushort Reserved,
        uint ResourceBinding,
        uint IndexStringId,
        uint LiteralStringId,
        uint TypeStringId,
        uint LeftNodeIndex,
        uint RightNodeIndex,
        uint SymbolStringId,
        uint FirstArgumentIndex,
        uint ArgumentCount);

    private readonly record struct RawAdAnnotation(
        FeatherIrAdAnnotationRole Role,
        uint Binding,
        uint NameStringId,
        uint ResourceNameStringId,
        uint TypeNameStringId,
        uint IndexStringId,
        FeatherIrAdSourceKind SourceKind,
        uint ElementCount);

    private ref struct Reader
    {
        private ReadOnlySpan<byte> remaining;

        public Reader(ReadOnlySpan<byte> data)
        {
            remaining = data;
        }

        public byte ReadByte()
        {
            Ensure(1);
            var value = remaining[0];
            remaining = remaining[1..];
            return value;
        }

        public ushort ReadUInt16()
        {
            Ensure(2);
            var value = BinaryPrimitives.ReadUInt16LittleEndian(remaining);
            remaining = remaining[2..];
            return value;
        }

        public uint ReadUInt32()
        {
            Ensure(4);
            var value = BinaryPrimitives.ReadUInt32LittleEndian(remaining);
            remaining = remaining[4..];
            return value;
        }

        public int ReadInt32()
        {
            Ensure(4);
            var value = BinaryPrimitives.ReadInt32LittleEndian(remaining);
            remaining = remaining[4..];
            return value;
        }

        public string ReadAscii(int length)
            => Encoding.ASCII.GetString(ReadBytes(length));

        public ReadOnlySpan<byte> ReadBytes(int length)
        {
            Ensure(length);
            var value = remaining[..length];
            remaining = remaining[length..];
            return value;
        }

        private void Ensure(int length)
        {
            if (remaining.Length < length)
            {
                throw new InvalidDataException("Unexpected end of Feather IR data.");
            }
        }
    }
}
