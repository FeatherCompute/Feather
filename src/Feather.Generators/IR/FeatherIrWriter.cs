using System.Text;
using Feather.Generators.Model;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Feather.Generators.IR;

internal static class FeatherIrWriter
{
    private static readonly byte[] Magic = Encoding.ASCII.GetBytes("FEIR");
    private const uint ElementwiseAssignmentSectionKind = 1;
    private const uint ElementwiseExpressionAssignmentSectionKind = 2;
        private const uint ControlFlowExpressionSectionKind = 3;
        private const uint AdAnnotationSectionKind = 4;
        private const uint LocalVariableSectionKind = 5;
        private const uint CompoundAssignmentSectionKind = 6;
    private const uint NoString = uint.MaxValue;
    private const uint NoBinding = uint.MaxValue;
    private const uint NoNode = uint.MaxValue;
    private static readonly LoweredShaderInstructionKind[] SemanticIntrinsicKinds =
    [
        LoweredShaderInstructionKind.WorkgroupBarrier,
        LoweredShaderInstructionKind.MemoryBarrier,
        LoweredShaderInstructionKind.FullBarrier,
        LoweredShaderInstructionKind.AtomicAdd,
        LoweredShaderInstructionKind.AtomicSub,
        LoweredShaderInstructionKind.AtomicMin,
        LoweredShaderInstructionKind.AtomicMax,
        LoweredShaderInstructionKind.AtomicAnd,
        LoweredShaderInstructionKind.AtomicOr,
        LoweredShaderInstructionKind.AtomicXor,
        LoweredShaderInstructionKind.AtomicExchange,
        LoweredShaderInstructionKind.AtomicCompareExchange
    ];

    public static byte[] WriteModule(ShaderModel model)
    {
        var strings = new StringTable();
        var kernelName = strings.Add(model.Name);
        var resources = model.Resources.Items
            .Select(resource => new SerializedResource(
                resource.Binding,
                ToIrResourceKind(resource.Kind),
                ToIrResourceAccess(resource.Access),
                strings.Add(resource.Name),
                strings.Add(NormalizeTypeName(resource.ElementTypeName))))
            .ToArray();
        var pushConstantCount = model.Resources.Items.Count(static resource => resource.Kind == ResourceKindModel.PushConstant);
        var instructions = BuildInstructions(model, strings).ToArray();
        var elementwiseAssignments = BuildElementwiseAssignments(model, instructions, strings).ToArray();
        var expressionAssignments = BuildElementwiseExpressionAssignments(model, instructions, strings);
        var controlFlowExpressions = BuildControlFlowExpressionSection(model, instructions, strings);
        var adAnnotations = BuildAdAnnotationSection(model, strings);
        var localVars = BuildLocalVariableSection(model, instructions, strings);
        var compoundAssigns = BuildCompoundAssignmentSection(model, instructions, strings);
        var sections = BuildSections(elementwiseAssignments, expressionAssignments, controlFlowExpressions, adAnnotations, localVars, compoundAssigns).ToList();
        if (model.TypedIrSection is { Length: > 0 } typedIr)
            sections.Add(new SerializedSection(ShaderIrModuleWriter.SectionKind, typedIr));
        var sectionsArray = sections.ToArray();

        var stringBytes = strings.ToBytes();
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8);

        writer.Write(Magic);
        writer.Write((ushort)1); // major
        writer.Write((ushort)1); // minor
        writer.Write((byte)1); // little endian
        writer.Write((byte)ToIrShaderKind(model.Kind));
        // Minor version 1 uses the former reserved header slot as a section count.
        // Sections carry typed bridge data while legacy operands remain for fallback compatibility.
        writer.Write((ushort)sectionsArray.Length);
        writer.Write(model.ThreadGroup.X);
        writer.Write(model.ThreadGroup.Y);
        writer.Write(model.ThreadGroup.Z);
        writer.Write((uint)kernelName);
        writer.Write((uint)resources.Length);
        writer.Write((uint)pushConstantCount);
        writer.Write((uint)instructions.Length);
        writer.Write((uint)stringBytes.Length);

        foreach (var resource in resources)
        {
            writer.Write(resource.Binding);
            writer.Write(resource.Kind);
            writer.Write(resource.Access);
            writer.Write((byte)0);
            writer.Write(resource.NameStringId);
            writer.Write(resource.ElementTypeStringId);
        }

        foreach (var instruction in instructions)
        {
            writer.Write(instruction.Opcode);
            writer.Write(instruction.OperandKind);
            writer.Write(instruction.Reserved);
            writer.Write(instruction.OperandStringId);
        }

        foreach (var section in sectionsArray)
        {
            writer.Write(section.Kind);
            writer.Write((uint)section.Payload.Length);
        }

        foreach (var section in sectionsArray)
        {
            writer.Write(section.Payload);
        }

        writer.Write(stringBytes);
        return stream.ToArray();
    }

    public static string ToCSharpByteArray(byte[] bytes)
        => string.Join(", ", bytes.Select(value => "0x" + value.ToString("X2")));

    private static byte ToIrShaderKind(ShaderKind kind)
        => kind switch
        {
            ShaderKind.Compute1D => 1,
            ShaderKind.Compute2D => 2,
            ShaderKind.Compute3D => 3,
            ShaderKind.Vertex => 4,
            ShaderKind.Fragment => 5,
            _ => 0
        };

    private static byte ToIrResourceKind(ResourceKindModel kind)
        => kind switch
        {
            ResourceKindModel.Buffer => 1,
            ResourceKindModel.Texture2D => 2,
            ResourceKindModel.Sampler => 3,
            ResourceKindModel.Uniform => 4,
            ResourceKindModel.PushConstant => 5,
            ResourceKindModel.Texture3D => 6,
            _ => 0
        };

    private static byte ToIrResourceAccess(ResourceAccessModel access)
        => access switch
        {
            ResourceAccessModel.Read => 1,
            ResourceAccessModel.Write => 2,
            ResourceAccessModel.ReadWrite => 3,
            ResourceAccessModel.Sample => 4,
            _ => 0
        };

    private static byte ToAssignmentOperation(LoweredElementwiseAssignmentOperation operation)
        => operation switch
        {
            LoweredElementwiseAssignmentOperation.Copy => 1,
            LoweredElementwiseAssignmentOperation.Add => 2,
            LoweredElementwiseAssignmentOperation.Subtract => 3,
            LoweredElementwiseAssignmentOperation.Multiply => 4,
            LoweredElementwiseAssignmentOperation.Divide => 5,
            _ => 0
        };

    private static byte ToAssignmentOperandKind(LoweredElementwiseAssignmentOperandKind kind)
        => kind switch
        {
            LoweredElementwiseAssignmentOperandKind.None => 0,
            LoweredElementwiseAssignmentOperandKind.Resource => 1,
            LoweredElementwiseAssignmentOperandKind.Literal => 2,
            _ => 0
        };

    private static byte ToExpressionNodeKind(LoweredElementwiseExpressionNodeKind kind)
        => kind switch
        {
            LoweredElementwiseExpressionNodeKind.Resource => 1,
            LoweredElementwiseExpressionNodeKind.Literal => 2,
            LoweredElementwiseExpressionNodeKind.Binary => 3,
            LoweredElementwiseExpressionNodeKind.Invocation => 4,
            LoweredElementwiseExpressionNodeKind.PushConstant => 5,
            LoweredElementwiseExpressionNodeKind.Comparison => 6,
            LoweredElementwiseExpressionNodeKind.LocalVariable => 7,
            LoweredElementwiseExpressionNodeKind.ShaderBuiltin => 8,
            LoweredElementwiseExpressionNodeKind.Ternary => 9,
            LoweredElementwiseExpressionNodeKind.Constructor => 10,
            LoweredElementwiseExpressionNodeKind.CallableCall => 11,
            LoweredElementwiseExpressionNodeKind.TextureSample => 12,
            LoweredElementwiseExpressionNodeKind.TextureSampleLevel => 13,
            LoweredElementwiseExpressionNodeKind.GpuStructField => 14,
            _ => 0
        };

    private static byte ToExpressionOperation(LoweredElementwiseExpressionOperation operation)
        => operation switch
        {
            LoweredElementwiseExpressionOperation.None => 0,
            LoweredElementwiseExpressionOperation.Add => 1,
            LoweredElementwiseExpressionOperation.Subtract => 2,
            LoweredElementwiseExpressionOperation.Multiply => 3,
            LoweredElementwiseExpressionOperation.Divide => 4,
            LoweredElementwiseExpressionOperation.Equal => 5,
            LoweredElementwiseExpressionOperation.NotEqual => 6,
            LoweredElementwiseExpressionOperation.Greater => 7,
            LoweredElementwiseExpressionOperation.Less => 8,
            LoweredElementwiseExpressionOperation.GreaterEqual => 9,
            LoweredElementwiseExpressionOperation.LessEqual => 10,
            _ => 0
        };

    private static string NormalizeTypeName(string typeName)
        => typeName.StartsWith("global::", StringComparison.Ordinal) ? typeName.Substring("global::".Length) : typeName;

    private static IEnumerable<SerializedInstruction> BuildInstructions(ShaderModel model, StringTable strings)
    {
        var loweredInstructions = model.LoweredInstructions.Items;
        var entry = model.EntryPointSyntax;
        if (entry?.Body is null)
        {
            yield break;
        }

        foreach (var statement in entry.Body.Statements)
        {
            foreach (var instruction in BuildInstructions(statement, strings, loweredInstructions))
            {
                yield return instruction;
            }
        }
    }

    private static IEnumerable<SerializedElementwiseAssignment> BuildElementwiseAssignments(
        ShaderModel model,
        IReadOnlyList<SerializedInstruction> instructions,
        StringTable strings)
    {
        var resourceBindings = model.Resources.Items.ToDictionary(resource => resource.Name, resource => resource.Binding, StringComparer.Ordinal);
        foreach (var lowered in model.LoweredInstructions.Items)
        {
            if (lowered.Kind != LoweredShaderInstructionKind.ElementwiseAssignment || lowered.ElementwiseAssignment is null)
            {
                continue;
            }

            var instructionIndex = IndexOfInstruction(instructions, lowered.SyntaxStart, IrInstructionOpcode.Assignment);
            if (instructionIndex is null
                || !resourceBindings.TryGetValue(lowered.ElementwiseAssignment.DestinationResourceName, out var destinationBinding)
                || !resourceBindings.TryGetValue(lowered.ElementwiseAssignment.LeftOperand, out var leftBinding))
            {
                continue;
            }

            var rightBinding = NoBinding;
            var rightLiteralStringId = NoString;
            if (lowered.ElementwiseAssignment.RightOperandKind == LoweredElementwiseAssignmentOperandKind.Resource
                && !resourceBindings.TryGetValue(lowered.ElementwiseAssignment.RightOperand, out rightBinding))
            {
                continue;
            }

            if (lowered.ElementwiseAssignment.RightOperandKind == LoweredElementwiseAssignmentOperandKind.Literal)
            {
                rightLiteralStringId = strings.Add(lowered.ElementwiseAssignment.RightOperand);
            }

            yield return new SerializedElementwiseAssignment(
                instructionIndex.Value,
                destinationBinding,
                leftBinding,
                rightBinding,
                ToAssignmentOperation(lowered.ElementwiseAssignment.Operation),
                ToAssignmentOperandKind(lowered.ElementwiseAssignment.RightOperandKind),
                strings.Add(lowered.ElementwiseAssignment.IndexName),
                rightLiteralStringId);
        }
    }

    private static SerializedElementwiseExpressionSection BuildElementwiseExpressionAssignments(
        ShaderModel model,
        IReadOnlyList<SerializedInstruction> instructions,
        StringTable strings)
    {
        var resourceBindings = model.Resources.Items.ToDictionary(resource => resource.Name, resource => resource.Binding, StringComparer.Ordinal);
        var assignments = new List<SerializedElementwiseExpressionAssignment>();
        var nodes = new List<SerializedExpressionNode>();
        var argumentIndices = new List<uint>();
        foreach (var lowered in model.LoweredInstructions.Items)
        {
            if (lowered.Kind != LoweredShaderInstructionKind.ElementwiseAssignment || lowered.ElementwiseExpressionAssignment is null)
            {
                continue;
            }

            var instructionIndex = IndexOfInstruction(instructions, lowered.SyntaxStart, IrInstructionOpcode.Assignment);
            if (instructionIndex is null
                || !resourceBindings.TryGetValue(lowered.ElementwiseExpressionAssignment.DestinationResourceName, out var destinationBinding)
                || !TrySerializeExpressionNode(lowered.ElementwiseExpressionAssignment.Expression, resourceBindings, strings, nodes, argumentIndices, out var rootNodeIndex))
            {
                continue;
            }

            assignments.Add(new SerializedElementwiseExpressionAssignment(
                instructionIndex.Value,
                destinationBinding,
                strings.Add(lowered.ElementwiseExpressionAssignment.IndexName),
                rootNodeIndex));
        }

        return new SerializedElementwiseExpressionSection(assignments, nodes, argumentIndices);
    }

    private static bool TrySerializeExpressionNode(
        LoweredElementwiseExpressionNodeModel node,
        IReadOnlyDictionary<string, uint> resourceBindings,
        StringTable strings,
        List<SerializedExpressionNode> nodes,
        List<uint> argumentIndices,
        out uint nodeIndex)
    {
        nodeIndex = NoNode;
        switch (node.Kind)
        {
            case LoweredElementwiseExpressionNodeKind.Resource:
                if (!resourceBindings.TryGetValue(node.ResourceName, out var binding))
                {
                    return false;
                }

                nodeIndex = AddExpressionNode(nodes, new SerializedExpressionNode(
                    ToExpressionNodeKind(node.Kind),
                    ToExpressionOperation(node.Operation),
                    binding,
                    strings.Add(node.IndexName),
                    NoString,
                    AddOptionalString(strings, NormalizeTypeName(node.TypeName)),
                    NoNode,
                    NoNode,
                    NoString,
                    NoNode,
                    0));
                return true;
            case LoweredElementwiseExpressionNodeKind.Literal:
                nodeIndex = AddExpressionNode(nodes, new SerializedExpressionNode(
                    ToExpressionNodeKind(node.Kind),
                    ToExpressionOperation(node.Operation),
                    NoBinding,
                    NoString,
                    strings.Add(node.Literal),
                    AddOptionalString(strings, NormalizeTypeName(node.TypeName)),
                    NoNode,
                    NoNode,
                    NoString,
                    NoNode,
                    0));
                return true;
            case LoweredElementwiseExpressionNodeKind.PushConstant:
                if (!resourceBindings.TryGetValue(node.ResourceName, out var pushConstantBinding))
                {
                    return false;
                }

                nodeIndex = AddExpressionNode(nodes, new SerializedExpressionNode(
                    ToExpressionNodeKind(node.Kind),
                    ToExpressionOperation(node.Operation),
                    pushConstantBinding,
                    NoString,
                    NoString,
                    AddOptionalString(strings, NormalizeTypeName(node.TypeName)),
                    NoNode,
                    NoNode,
                    NoString,
                    NoNode,
                    0));
                return true;
            case LoweredElementwiseExpressionNodeKind.Binary:
                if (node.Left is null
                    || node.Right is null
                    || !TrySerializeExpressionNode(node.Left, resourceBindings, strings, nodes, argumentIndices, out var leftIndex)
                    || !TrySerializeExpressionNode(node.Right, resourceBindings, strings, nodes, argumentIndices, out var rightIndex))
                {
                    return false;
                }

                nodeIndex = AddExpressionNode(nodes, new SerializedExpressionNode(
                    ToExpressionNodeKind(node.Kind),
                    ToExpressionOperation(node.Operation),
                    NoBinding,
                    NoString,
                    NoString,
                    AddOptionalString(strings, NormalizeTypeName(node.TypeName)),
                    leftIndex,
                    rightIndex,
                    NoString,
                    NoNode,
                    0));
                return true;
            case LoweredElementwiseExpressionNodeKind.Invocation:
                if (string.IsNullOrEmpty(node.SymbolName))
                {
                    return false;
                }

                var firstArgumentIndex = node.Arguments.Items.Count == 0 ? NoNode : (uint)argumentIndices.Count;
                foreach (var argument in node.Arguments.Items)
                {
                    if (!TrySerializeExpressionNode(argument, resourceBindings, strings, nodes, argumentIndices, out var argumentNodeIndex))
                    {
                        return false;
                    }

                    argumentIndices.Add(argumentNodeIndex);
                }

                nodeIndex = AddExpressionNode(nodes, new SerializedExpressionNode(
                    ToExpressionNodeKind(node.Kind),
                    ToExpressionOperation(node.Operation),
                    NoBinding,
                    NoString,
                    NoString,
                    AddOptionalString(strings, NormalizeTypeName(node.TypeName)),
                    NoNode,
                    NoNode,
                    strings.Add(node.SymbolName),
                    firstArgumentIndex,
                    (uint)node.Arguments.Items.Count));
                return true;
            case LoweredElementwiseExpressionNodeKind.Comparison:
                if (node.Left is null || node.Right is null
                    || !TrySerializeExpressionNode(node.Left, resourceBindings, strings, nodes, argumentIndices, out var cmpLeft)
                    || !TrySerializeExpressionNode(node.Right, resourceBindings, strings, nodes, argumentIndices, out var cmpRight))
                    return false;
                nodeIndex = AddExpressionNode(nodes, new SerializedExpressionNode(
                    ToExpressionNodeKind(node.Kind), ToExpressionOperation(node.Operation),
                    NoBinding, NoString, NoString,
                    AddOptionalString(strings, NormalizeTypeName(node.TypeName)),
                    cmpLeft, cmpRight, NoString, NoNode, 0));
                return true;
            case LoweredElementwiseExpressionNodeKind.LocalVariable:
                nodeIndex = AddExpressionNode(nodes, new SerializedExpressionNode(
                    ToExpressionNodeKind(node.Kind), ToExpressionOperation(node.Operation),
                    NoBinding, NoString, NoString,
                    AddOptionalString(strings, NormalizeTypeName(node.TypeName)),
                    NoNode, NoNode, strings.Add(node.ResourceName), NoNode, 0));
                return true;
            case LoweredElementwiseExpressionNodeKind.ShaderBuiltin:
                nodeIndex = AddExpressionNode(nodes, new SerializedExpressionNode(
                    ToExpressionNodeKind(node.Kind), (byte)node.BuiltinKind,
                    NoBinding, NoString, NoString,
                    AddOptionalString(strings, NormalizeTypeName(node.TypeName)),
                    NoNode, NoNode, NoString, NoNode, 0));
                return true;
            case LoweredElementwiseExpressionNodeKind.Ternary:
                if (node.Left is null || node.Right is null || node.Arguments.Items.Count == 0) return false;
                if (!TrySerializeExpressionNode(node.Left, resourceBindings, strings, nodes, argumentIndices, out var tCond)) return false;
                if (!TrySerializeExpressionNode(node.Right, resourceBindings, strings, nodes, argumentIndices, out var tTrue)) return false;
                if (!TrySerializeExpressionNode(node.Arguments.Items[0], resourceBindings, strings, nodes, argumentIndices, out var tFalse)) return false;
                var tArgStart = (uint)argumentIndices.Count;
                argumentIndices.Add(tFalse);
                nodeIndex = AddExpressionNode(nodes, new SerializedExpressionNode(
                    ToExpressionNodeKind(node.Kind), ToExpressionOperation(node.Operation),
                    NoBinding, NoString, NoString,
                    AddOptionalString(strings, NormalizeTypeName(node.TypeName)),
                    tCond, tTrue, NoString, tArgStart, 1));
                return true;
            case LoweredElementwiseExpressionNodeKind.Constructor:
                {
                    var consFirstArg = (uint)argumentIndices.Count;
                    foreach (var arg in node.Arguments.Items)
                    {
                        if (!TrySerializeExpressionNode(arg, resourceBindings, strings, nodes, argumentIndices, out var argIdx))
                            return false;
                        argumentIndices.Add(argIdx);
                    }
                    nodeIndex = AddExpressionNode(nodes, new SerializedExpressionNode(
                        ToExpressionNodeKind(node.Kind), ToExpressionOperation(node.Operation),
                        NoBinding, NoString, NoString,
                        AddOptionalString(strings, NormalizeTypeName(node.TypeName)),
                        NoNode, NoNode, NoString, consFirstArg, (uint)node.Arguments.Items.Count));
                }
                return true;
            case LoweredElementwiseExpressionNodeKind.CallableCall:
                if (string.IsNullOrEmpty(node.SymbolName))
                {
                    return false;
                }

                var callFirstArg = node.Arguments.Items.Count == 0 ? NoNode : (uint)argumentIndices.Count;
                foreach (var argument in node.Arguments.Items)
                {
                    if (!TrySerializeExpressionNode(argument, resourceBindings, strings, nodes, argumentIndices, out var callArgNodeIndex))
                    {
                        return false;
                    }

                    argumentIndices.Add(callArgNodeIndex);
                }

                nodeIndex = AddExpressionNode(nodes, new SerializedExpressionNode(
                    ToExpressionNodeKind(node.Kind),
                    ToExpressionOperation(node.Operation),
                    NoBinding,
                    NoString,
                    NoString,
                    AddOptionalString(strings, NormalizeTypeName(node.TypeName)),
                    NoNode,
                    NoNode,
                    strings.Add(node.SymbolName),
                    callFirstArg,
                    (uint)node.Arguments.Items.Count));
                return true;
            case LoweredElementwiseExpressionNodeKind.TextureSample:
            case LoweredElementwiseExpressionNodeKind.TextureSampleLevel:
            {
                if (string.IsNullOrEmpty(node.ResourceName))
                    return false;

                // ResourceName = texture parameter name → resource binding
                if (!resourceBindings.TryGetValue(node.ResourceName, out var texBinding))
                    return false;
                // SymbolName = sampler parameter name → stored as string table entry
                var samplerNameId = !string.IsNullOrEmpty(node.SymbolName) ? strings.Add(node.SymbolName) : NoString;

                var tsFirstArg = node.Arguments.Items.Count == 0 ? NoNode : (uint)argumentIndices.Count;
                foreach (var argument in node.Arguments.Items)
                {
                    if (!TrySerializeExpressionNode(argument, resourceBindings, strings, nodes, argumentIndices, out var tsArgIdx))
                        return false;
                    argumentIndices.Add(tsArgIdx);
                }

                nodeIndex = AddExpressionNode(nodes, new SerializedExpressionNode(
                    ToExpressionNodeKind(node.Kind),
                    ToExpressionOperation(node.Operation),
                    texBinding,
                    NoString,
                    NoString,
                    AddOptionalString(strings, NormalizeTypeName(node.TypeName)),
                    NoNode,
                    NoNode,
                    samplerNameId, // SymbolStringId = sampler resource name (string table)
                    tsFirstArg,
                    (uint)node.Arguments.Items.Count));
                return true;
            }
            case LoweredElementwiseExpressionNodeKind.GpuStructField:
            {
                if (node.Arguments.Items.Count == 0)
                    return false;
                if (!TrySerializeExpressionNode(node.Arguments.Items[0], resourceBindings, strings, nodes, argumentIndices, out var gfInstance))
                    return false;
                var gfFirstArg = (uint)argumentIndices.Count;
                argumentIndices.Add(gfInstance);
                nodeIndex = AddExpressionNode(nodes, new SerializedExpressionNode(
                    ToExpressionNodeKind(node.Kind),
                    (byte)(int)node.Operation, // field index
                    NoBinding,
                    NoString,
                    NoString,
                    AddOptionalString(strings, NormalizeTypeName(node.TypeName)),
                    NoNode,
                    NoNode,
                    NoString,
                    gfFirstArg,
                    1));
                return true;
            }
            default:
                return false;
        }
    }

    private static uint AddExpressionNode(List<SerializedExpressionNode> nodes, SerializedExpressionNode node)
    {
        var index = (uint)nodes.Count;
        nodes.Add(node);
        return index;
    }

    private static uint AddOptionalString(StringTable strings, string value)
        => string.IsNullOrEmpty(value) ? NoString : strings.Add(value);

    private static SerializedSection? BuildControlFlowExpressionSection(ShaderModel model, IReadOnlyList<SerializedInstruction> instructions, StringTable strings)
    {
        var records = new List<(uint instructionIndex, uint kind, uint rootNodeIndex)>();
        var nodes = new List<SerializedExpressionNode>();
        var argumentIndices = new List<uint>();
        var resourceBindings = model.Resources.Items.ToDictionary(r => r.Name, r => r.Binding, StringComparer.Ordinal);
        foreach (var lowered in model.LoweredInstructions.Items)
        {
            if (lowered.ControlFlowCondition is null) continue;
            var instructionIndex = IndexOfInstruction(instructions, lowered.SyntaxStart,
                lowered.ControlFlowCondition.Role switch
                {
                    LoweredControlFlowRole.IfCondition => IrInstructionOpcode.If,
                    LoweredControlFlowRole.ForCondition or LoweredControlFlowRole.ForInit or LoweredControlFlowRole.ForStep => IrInstructionOpcode.For,
                    LoweredControlFlowRole.WhileCondition => IrInstructionOpcode.While,
                    LoweredControlFlowRole.DoCondition => IrInstructionOpcode.Do,
                    _ => IrInstructionOpcode.Expression
                });
            if (instructionIndex is null) continue;
            if (!TrySerializeExpressionNode(lowered.ControlFlowCondition.Expression, resourceBindings, strings, nodes, argumentIndices, out var rootNodeIndex)) continue;
            uint kind = lowered.ControlFlowCondition.Role switch
            {
                LoweredControlFlowRole.IfCondition => 1, LoweredControlFlowRole.ForCondition => 2,
                LoweredControlFlowRole.ForInit => 3, LoweredControlFlowRole.ForStep => 4,
                LoweredControlFlowRole.WhileCondition => 5, LoweredControlFlowRole.DoCondition => 6,
                _ => 0
            };
            records.Add((instructionIndex.Value, kind, rootNodeIndex));
        }
        if (records.Count == 0) return null;
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8);
        writer.Write((uint)records.Count); writer.Write((uint)nodes.Count); writer.Write((uint)argumentIndices.Count);
        foreach (var (instrIdx, kind, nodeIdx) in records) { writer.Write(instrIdx); writer.Write(kind); writer.Write(nodeIdx); }
        foreach (var node in nodes) { writer.Write(node.Kind); writer.Write(node.Operation); writer.Write((ushort)0); writer.Write(node.ResourceBinding); writer.Write(node.IndexStringId); writer.Write(node.LiteralStringId); writer.Write(node.TypeStringId); writer.Write(node.LeftNodeIndex); writer.Write(node.RightNodeIndex); writer.Write(node.SymbolStringId); writer.Write(node.FirstArgumentIndex); writer.Write(node.ArgumentCount); }
        foreach (var idx in argumentIndices) writer.Write(idx);
        return new SerializedSection(ControlFlowExpressionSectionKind, stream.ToArray());
    }

    private static SerializedSection? BuildAdAnnotationSection(ShaderModel model, StringTable strings)
    {
        var resourceBindings = model.Resources.Items.ToDictionary(r => r.Name, r => r.Binding, StringComparer.Ordinal);
        var parameters = new List<SerializedAdAnnotation>();
        var losses = new List<SerializedAdAnnotation>();
        foreach (var instruction in model.LoweredInstructions.Items)
        {
            if (instruction.AdAnnotation is null) continue;
            var annotation = instruction.AdAnnotation;
            var binding = NoBinding;
            if (!string.IsNullOrEmpty(annotation.ResourceName))
            {
                if (!resourceBindings.TryGetValue(annotation.ResourceName, out binding))
                {
                    binding = NoBinding;
                }
            }

            var record = new SerializedAdAnnotation(
                (uint)annotation.Role,
                binding,
                strings.Add(annotation.Name),
                string.IsNullOrEmpty(annotation.ResourceName) ? NoString : strings.Add(annotation.ResourceName),
                strings.Add(NormalizeTypeName(annotation.TypeName)),
                string.IsNullOrEmpty(annotation.IndexName) ? NoString : strings.Add(annotation.IndexName),
                (uint)annotation.SourceKind,
                0);
            if (annotation.Role == LoweredAdAnnotationRole.Parameter) parameters.Add(record); else losses.Add(record);
        }
        if (parameters.Count == 0 && losses.Count == 0) return null;
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8);
        writer.Write((ushort)2);
        writer.Write((ushort)0);
        writer.Write((uint)parameters.Count);
        writer.Write((uint)losses.Count);
        foreach (var record in parameters) WriteAdRecord(writer, record);
        foreach (var record in losses) WriteAdRecord(writer, record);
        return new SerializedSection(AdAnnotationSectionKind, stream.ToArray());
    }

    private static void WriteAdRecord(BinaryWriter writer, SerializedAdAnnotation record)
    {
        writer.Write(record.Role);
        writer.Write(record.Binding);
        writer.Write(record.NameStringId);
        writer.Write(record.ResourceNameStringId);
        writer.Write(record.TypeNameStringId);
        writer.Write(record.IndexNameStringId);
        writer.Write(record.SourceKind);
        writer.Write(record.ElementCount);
    }

    private static SerializedSection? BuildLocalVariableSection(ShaderModel model, IReadOnlyList<SerializedInstruction> instructions, StringTable strings)
    {
        var decls = new List<(uint instrIdx, uint nameId, uint glslTextId)>();
        foreach (var lowered in model.LoweredInstructions.Items)
        {
            if (lowered.LocalDeclaration is null) continue;
            var d = lowered.LocalDeclaration;
            var instrIdx = IndexOfInstruction(instructions, lowered.SyntaxStart, IrInstructionOpcode.LocalDeclaration);
            if (instrIdx is null) continue;
            var glslDecl = BuildGlslDeclarationText(d);
            decls.Add((instrIdx.Value, strings.Add(d.VariableName), strings.Add(glslDecl)));
        }
        if (decls.Count == 0) return null;
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8);
        writer.Write((uint)decls.Count);
        foreach (var (instrIdx, nameId, glslTextId) in decls) { writer.Write(instrIdx); writer.Write(nameId); writer.Write(glslTextId); }
        return new SerializedSection(LocalVariableSectionKind, stream.ToArray());
    }

    private static string BuildGlslDeclarationText(LoweredLocalDeclarationModel d)
    {
        var sb = new StringBuilder();
        sb.Append(d.GlslTypeName); sb.Append(' '); sb.Append(d.VariableName);
        if (d.Initializer is not null) { sb.Append(" = "); AppendGlslInitializer(sb, d.Initializer, d.GlslTypeName); }
        sb.Append(';');
        return sb.ToString();
    }

    private static void AppendGlslInitializer(StringBuilder sb, LoweredElementwiseExpressionNodeModel expr, string glslType)
    {
        if (TryAppendKnownGlslInitializer(sb, expr, glslType)) return;
        sb.Append(glslType == "float" ? "0.0" : "0");
    }

    private static bool TryAppendKnownGlslInitializer(StringBuilder sb, LoweredElementwiseExpressionNodeModel expr, string glslType)
    {
        switch (expr.Kind)
        {
            case LoweredElementwiseExpressionNodeKind.ShaderBuiltin: AppendBuiltinGlsl(sb, expr.BuiltinKind); return true;
            case LoweredElementwiseExpressionNodeKind.Literal: sb.Append(expr.Literal); return true;
            case LoweredElementwiseExpressionNodeKind.Resource: sb.Append(expr.ResourceName); sb.Append('['); sb.Append(expr.IndexName); sb.Append(']'); return true;
            default: return false;
        }
    }

    private static void AppendBuiltinGlsl(StringBuilder sb, LoweredShaderBuiltinKind kind)
    {
        sb.Append(kind switch {
            LoweredShaderBuiltinKind.ThreadIndexX => "int(gl_GlobalInvocationID.x)",
            LoweredShaderBuiltinKind.ThreadIndexY => "int(gl_GlobalInvocationID.y)",
            LoweredShaderBuiltinKind.ThreadIndexZ => "int(gl_GlobalInvocationID.z)",
            LoweredShaderBuiltinKind.LocalIndexX => "int(gl_LocalInvocationID.x)",
            LoweredShaderBuiltinKind.LocalIndexY => "int(gl_LocalInvocationID.y)",
            LoweredShaderBuiltinKind.LocalIndexZ => "int(gl_LocalInvocationID.z)",
            LoweredShaderBuiltinKind.GroupIdX => "int(gl_WorkGroupID.x)",
            LoweredShaderBuiltinKind.GroupIdY => "int(gl_WorkGroupID.y)",
            LoweredShaderBuiltinKind.GroupIdZ => "int(gl_WorkGroupID.z)",
            LoweredShaderBuiltinKind.VertexIndex => "int(gl_VertexIndex)",
            LoweredShaderBuiltinKind.InstanceIndex => "int(gl_InstanceIndex)",
            LoweredShaderBuiltinKind.FragmentCoordX => "gl_FragCoord",
            _ => "0"});
    }

    private static SerializedSection? BuildCompoundAssignmentSection(ShaderModel model, IReadOnlyList<SerializedInstruction> instructions, StringTable strings)
    {
        var resourceBindings = model.Resources.Items.ToDictionary(r => r.Name, r => r.Binding, StringComparer.Ordinal);
        var records = new List<(uint instrIdx, uint dstBinding, uint idxStrId, uint op, uint rootNode)>();
        var nodes = new List<SerializedExpressionNode>(); var argumentIndices = new List<uint>();
        foreach (var lowered in model.LoweredInstructions.Items)
        {
            if (lowered.CompoundAssignment is null) continue;
            var ca = lowered.CompoundAssignment;
            var instrIdx = IndexOfInstruction(instructions, lowered.SyntaxStart, IrInstructionOpcode.Assignment);
            if (instrIdx is null) continue;
            if (!resourceBindings.TryGetValue(ca.ResourceName, out var binding)) continue;
            if (!TrySerializeExpressionNode(ca.Value, resourceBindings, strings, nodes, argumentIndices, out var root)) continue;
            records.Add((instrIdx.Value, binding, strings.Add(ca.IndexName), (uint)ToAssignmentOperation(ca.Operation), root));
        }
        if (records.Count == 0) return null;
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8);
        writer.Write((uint)records.Count); writer.Write((uint)nodes.Count); writer.Write((uint)argumentIndices.Count);
        foreach (var (instrIdx, dstBinding, idxStrId, op, rootNode) in records) { writer.Write(instrIdx); writer.Write(dstBinding); writer.Write(idxStrId); writer.Write(op); writer.Write(rootNode); }
        foreach (var node in nodes) { writer.Write(node.Kind); writer.Write(node.Operation); writer.Write((ushort)0); writer.Write(node.ResourceBinding); writer.Write(node.IndexStringId); writer.Write(node.LiteralStringId); writer.Write(node.TypeStringId); writer.Write(node.LeftNodeIndex); writer.Write(node.RightNodeIndex); writer.Write(node.SymbolStringId); writer.Write(node.FirstArgumentIndex); writer.Write(node.ArgumentCount); }
        foreach (var idx in argumentIndices) writer.Write(idx);
        return new SerializedSection(CompoundAssignmentSectionKind, stream.ToArray());
    }

    private static IEnumerable<SerializedSection> BuildSections(
        IReadOnlyList<SerializedElementwiseAssignment> elementwiseAssignments,
        SerializedElementwiseExpressionSection expressionAssignments,
        SerializedSection? controlFlowExpressions,
        SerializedSection? adAnnotations,
        SerializedSection? localVars,
        SerializedSection? compoundAssigns)
    {
        if (elementwiseAssignments.Count > 0)
            yield return new SerializedSection(ElementwiseAssignmentSectionKind, WriteElementwiseAssignmentSection(elementwiseAssignments));
        if (expressionAssignments.Assignments.Count > 0)
            yield return new SerializedSection(ElementwiseExpressionAssignmentSectionKind, WriteElementwiseExpressionAssignmentSection(expressionAssignments));
        if (controlFlowExpressions is not null)
            yield return controlFlowExpressions.Value;
        if (adAnnotations is not null)
            yield return adAnnotations.Value;
        if (localVars is not null)
            yield return localVars.Value;
        if (compoundAssigns is not null)
            yield return compoundAssigns.Value;
    }

    private static byte[] WriteElementwiseAssignmentSection(IReadOnlyList<SerializedElementwiseAssignment> assignments)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8);
        writer.Write((uint)assignments.Count);
        foreach (var assignment in assignments)
        {
            // Binding-based records let the native EasyGPU bridge consume semantic assignment data
            // without parsing the transitional ASSIGN1 operand string.
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

    private static byte[] WriteElementwiseExpressionAssignmentSection(SerializedElementwiseExpressionSection section)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8);
        writer.Write((uint)section.Assignments.Count);
        writer.Write((uint)section.Nodes.Count);
        writer.Write((uint)section.ArgumentIndices.Count);
        foreach (var assignment in section.Assignments)
        {
            writer.Write(assignment.InstructionIndex);
            writer.Write(assignment.DestinationBinding);
            writer.Write(assignment.IndexStringId);
            writer.Write(assignment.RootNodeIndex);
        }

        foreach (var node in section.Nodes)
        {
            // Expression records are child-index based so the EasyGPU bridge can build a typed
            // expression graph directly from Roslyn semantic lowering.
            writer.Write(node.Kind);
            writer.Write(node.Operation);
            writer.Write((ushort)0);
            writer.Write(node.ResourceBinding);
            writer.Write(node.IndexStringId);
            writer.Write(node.LiteralStringId);
            writer.Write(node.TypeStringId);
            writer.Write(node.LeftNodeIndex);
            writer.Write(node.RightNodeIndex);
            writer.Write(node.SymbolStringId);
            writer.Write(node.FirstArgumentIndex);
            writer.Write(node.ArgumentCount);
        }

        foreach (var argumentIndex in section.ArgumentIndices)
        {
            writer.Write(argumentIndex);
        }

        return stream.ToArray();
    }

    private static uint? IndexOfInstruction(IReadOnlyList<SerializedInstruction> instructions, int syntaxStart, IrInstructionOpcode opcode)
    {
        for (var i = 0; i < instructions.Count; i++)
        {
            if (instructions[i].SyntaxStart == syntaxStart && instructions[i].Opcode == (byte)opcode)
            {
                return (uint)i;
            }
        }

        return null;
    }

    private static IEnumerable<SerializedInstruction> BuildInstructions(
        StatementSyntax statement,
        StringTable strings,
        IReadOnlyList<LoweredShaderInstructionModel> loweredInstructions)
    {
        switch (statement)
        {
            case LocalDeclarationStatementSyntax local:
                if (IsSharedMemoryDeclaration(local))
                {
                    yield return CreateInstruction(IrInstructionOpcode.SharedMemoryDeclaration, IrOperandKind.Source, local.Declaration, strings);
                }

                yield return CreateInstruction(IrInstructionOpcode.LocalDeclaration, IrOperandKind.Source, local.Declaration, strings);
                break;
            case ExpressionStatementSyntax expression:
                if (expression.Expression is AssignmentExpressionSyntax assignment)
                {
                    yield return CreateAssignmentInstruction(assignment, strings, loweredInstructions);
                }
                else
                {
                    yield return CreateInstruction(IrInstructionOpcode.Expression, IrOperandKind.Source, expression.Expression, strings);
                }

                break;
            case ReturnStatementSyntax @return:
                yield return @return.Expression is not null
                    ? CreateInstruction(IrInstructionOpcode.Return, IrOperandKind.Source, @return.Expression, strings)
                    : CreateInstruction(IrInstructionOpcode.Return, IrOperandKind.Source, @return.ReturnKeyword, strings);
                break;
            case IfStatementSyntax @if:
                yield return CreateInstruction(IrInstructionOpcode.If, IrOperandKind.Source, @if.Condition, strings);
                yield return CreateInstruction(IrInstructionOpcode.BeginBlock, IrOperandKind.Source, @if.IfKeyword, strings);
                foreach (var nested in BuildInstructions(@if.Statement, strings, loweredInstructions))
                {
                    yield return nested;
                }

                yield return CreateInstruction(IrInstructionOpcode.EndBlock, IrOperandKind.Source, @if.IfKeyword, strings);
                if (@if.Else is not null)
                {
                    yield return CreateInstruction(IrInstructionOpcode.Else, IrOperandKind.Source, @if.Else.ElseKeyword, strings);
                    yield return CreateInstruction(IrInstructionOpcode.BeginBlock, IrOperandKind.Source, @if.Else.ElseKeyword, strings);
                    foreach (var nested in BuildInstructions(@if.Else.Statement, strings, loweredInstructions))
                    {
                        yield return nested;
                    }

                    yield return CreateInstruction(IrInstructionOpcode.EndBlock, IrOperandKind.Source, @if.Else.ElseKeyword, strings);
                }

                break;
            case ForStatementSyntax @for:
                yield return CreateInstruction(IrInstructionOpcode.For, IrOperandKind.Source, @for, strings);
                yield return CreateInstruction(IrInstructionOpcode.BeginBlock, IrOperandKind.Source, @for.ForKeyword, strings);
                foreach (var nested in BuildInstructions(@for.Statement, strings, loweredInstructions))
                {
                    yield return nested;
                }

                yield return CreateInstruction(IrInstructionOpcode.EndBlock, IrOperandKind.Source, @for.ForKeyword, strings);
                break;
            case WhileStatementSyntax @while:
                yield return CreateInstruction(IrInstructionOpcode.While, IrOperandKind.Source, @while.Condition, strings);
                yield return CreateInstruction(IrInstructionOpcode.BeginBlock, IrOperandKind.Source, @while.WhileKeyword, strings);
                foreach (var nested in BuildInstructions(@while.Statement, strings, loweredInstructions))
                {
                    yield return nested;
                }

                yield return CreateInstruction(IrInstructionOpcode.EndBlock, IrOperandKind.Source, @while.WhileKeyword, strings);
                break;
            case DoStatementSyntax @do:
                yield return CreateInstruction(IrInstructionOpcode.Do, IrOperandKind.Source, @do.Condition, strings);
                yield return CreateInstruction(IrInstructionOpcode.BeginBlock, IrOperandKind.Source, @do.DoKeyword, strings);
                foreach (var nested in BuildInstructions(@do.Statement, strings, loweredInstructions))
                {
                    yield return nested;
                }

                yield return CreateInstruction(IrInstructionOpcode.EndBlock, IrOperandKind.Source, @do.DoKeyword, strings);
                break;
            case BreakStatementSyntax @break:
                yield return CreateInstruction(IrInstructionOpcode.Break, IrOperandKind.Source, @break.BreakKeyword, strings);
                break;
            case ContinueStatementSyntax @continue:
                yield return CreateInstruction(IrInstructionOpcode.Continue, IrOperandKind.Source, @continue.ContinueKeyword, strings);
                break;
            case BlockSyntax block:
                foreach (var nested in block.Statements)
                {
                    foreach (var instruction in BuildInstructions(nested, strings, loweredInstructions))
                    {
                        yield return instruction;
                    }
                }

                break;
            default:
                yield return CreateInstruction(IrInstructionOpcode.Expression, IrOperandKind.Source, statement, strings);
                break;
        }

        foreach (var invocation in statement.DescendantNodes().OfType<InvocationExpressionSyntax>())
        {
            if (TryCreateSemanticIntrinsicInstruction(invocation, strings, loweredInstructions, out var intrinsic))
            {
                yield return intrinsic;
            }

            yield return CreateInvocationInstruction(invocation, strings, loweredInstructions);
        }

        foreach (var elementAccess in statement.DescendantNodes().OfType<ElementAccessExpressionSyntax>())
        {
            yield return CreateResourceAccessInstruction(elementAccess, strings, loweredInstructions);
        }
    }

    private static SerializedInstruction CreateInstruction(IrInstructionOpcode opcode, IrOperandKind operandKind, SyntaxNodeOrToken operand, StringTable strings)
        => new((byte)opcode, (byte)operandKind, 0, strings.Add(NormalizeSource(operand)), operand.SpanStart);

    private static SerializedInstruction CreateAssignmentInstruction(
        AssignmentExpressionSyntax assignment,
        StringTable strings,
        IReadOnlyList<LoweredShaderInstructionModel> loweredInstructions)
    {
        var hasLoweredAssignment = TryGetLoweredInstruction(
            loweredInstructions,
            assignment.SpanStart,
            LoweredShaderInstructionKind.ElementwiseAssignment,
            out var lowered);
        var payload = hasLoweredAssignment && lowered.ElementwiseAssignment is not null
            ? FormatElementwiseAssignmentPayload(lowered.ElementwiseAssignment)
            : NormalizeSource(assignment);
        var operandKind = hasLoweredAssignment ? IrOperandKind.ElementwiseAssignment : IrOperandKind.Source;
        return new SerializedInstruction((byte)IrInstructionOpcode.Assignment, (byte)operandKind, 0, strings.Add(payload), assignment.SpanStart);
    }

    private static SerializedInstruction CreateInvocationInstruction(
        InvocationExpressionSyntax invocation,
        StringTable strings,
        IReadOnlyList<LoweredShaderInstructionModel> loweredInstructions)
    {
        if (TryGetLoweredInstruction(loweredInstructions, invocation.SpanStart, LoweredShaderInstructionKind.KnownSymbolInvocation, out var lowered))
        {
            return new SerializedInstruction((byte)IrInstructionOpcode.Invocation, (byte)IrOperandKind.Symbol, 0, strings.Add(lowered.Payload), invocation.SpanStart);
        }

        return CreateInstruction(IrInstructionOpcode.Invocation, IrOperandKind.Source, invocation.Expression, strings);
    }

    private static SerializedInstruction CreateResourceAccessInstruction(
        ElementAccessExpressionSyntax elementAccess,
        StringTable strings,
        IReadOnlyList<LoweredShaderInstructionModel> loweredInstructions)
    {
        if (TryGetLoweredInstruction(loweredInstructions, elementAccess.SpanStart, LoweredShaderInstructionKind.ResourceAccess, out var lowered))
        {
            return new SerializedInstruction((byte)IrInstructionOpcode.ResourceAccess, (byte)IrOperandKind.Symbol, 0, strings.Add(lowered.Payload), elementAccess.SpanStart);
        }

        return CreateInstruction(IrInstructionOpcode.ResourceAccess, IrOperandKind.Source, elementAccess.Expression, strings);
    }

    private static bool IsSharedMemoryDeclaration(LocalDeclarationStatementSyntax local)
    {
        if (IsSharedMemoryType(local.Declaration.Type))
        {
            return true;
        }

        // `var shared = new SharedMemory<T>(...)` is the idiomatic C# spelling; keep the IR marker even when
        // the declaration type is inferred from the constructor.
        return local.Declaration.Variables.Any(static variable => variable.Initializer?.Value is ObjectCreationExpressionSyntax creation
            && IsSharedMemoryType(creation.Type));
    }

    private static bool IsSharedMemoryType(TypeSyntax type)
        => type switch
        {
            GenericNameSyntax genericName => genericName.Identifier.ValueText == "SharedMemory",
            QualifiedNameSyntax { Right: GenericNameSyntax genericName } => genericName.Identifier.ValueText == "SharedMemory",
            AliasQualifiedNameSyntax { Name: GenericNameSyntax genericName } => genericName.Identifier.ValueText == "SharedMemory",
            _ => false
        };

    private static bool TryCreateSemanticIntrinsicInstruction(
        InvocationExpressionSyntax invocation,
        StringTable strings,
        IReadOnlyList<LoweredShaderInstructionModel> loweredInstructions,
        out SerializedInstruction instruction)
    {
        if (TryGetSemanticIntrinsicInstruction(loweredInstructions, invocation.SpanStart, out var lowered)
            && TryGetSemanticIntrinsicOpcode(lowered.Kind, out var opcode))
        {
            // Intrinsic opcodes are keyed by Roslyn symbols. This keeps lowering stable across aliases,
            // using-static imports, and other source spellings that refer to the same shader marker method.
            instruction = new SerializedInstruction((byte)opcode, (byte)IrOperandKind.Symbol, 0, strings.Add(lowered.Payload), invocation.SpanStart);
            return true;
        }

        instruction = default;
        return false;
    }

    private static bool TryGetSemanticIntrinsicInstruction(
        IReadOnlyList<LoweredShaderInstructionModel> loweredInstructions,
        int syntaxStart,
        out LoweredShaderInstructionModel instruction)
    {
        foreach (var kind in SemanticIntrinsicKinds)
        {
            if (TryGetLoweredInstruction(loweredInstructions, syntaxStart, kind, out instruction))
            {
                return true;
            }
        }

        instruction = default!;
        return false;
    }

    private static bool TryGetLoweredInstruction(
        IReadOnlyList<LoweredShaderInstructionModel> loweredInstructions,
        int syntaxStart,
        LoweredShaderInstructionKind kind,
        out LoweredShaderInstructionModel instruction)
    {
        foreach (var candidate in loweredInstructions)
        {
            if (candidate.SyntaxStart == syntaxStart && candidate.Kind == kind)
            {
                instruction = candidate;
                return true;
            }
        }

        instruction = default!;
        return false;
    }

    private static bool TryGetSemanticIntrinsicOpcode(LoweredShaderInstructionKind kind, out IrInstructionOpcode opcode)
    {
        opcode = kind switch
        {
            LoweredShaderInstructionKind.WorkgroupBarrier => IrInstructionOpcode.WorkgroupBarrier,
            LoweredShaderInstructionKind.MemoryBarrier => IrInstructionOpcode.MemoryBarrier,
            LoweredShaderInstructionKind.FullBarrier => IrInstructionOpcode.FullBarrier,
            LoweredShaderInstructionKind.AtomicAdd => IrInstructionOpcode.AtomicAdd,
            LoweredShaderInstructionKind.AtomicSub => IrInstructionOpcode.AtomicSub,
            LoweredShaderInstructionKind.AtomicMin => IrInstructionOpcode.AtomicMin,
            LoweredShaderInstructionKind.AtomicMax => IrInstructionOpcode.AtomicMax,
            LoweredShaderInstructionKind.AtomicAnd => IrInstructionOpcode.AtomicAnd,
            LoweredShaderInstructionKind.AtomicOr => IrInstructionOpcode.AtomicOr,
            LoweredShaderInstructionKind.AtomicXor => IrInstructionOpcode.AtomicXor,
            LoweredShaderInstructionKind.AtomicExchange => IrInstructionOpcode.AtomicExchange,
            LoweredShaderInstructionKind.AtomicCompareExchange => IrInstructionOpcode.AtomicCompareExchange,
            _ => IrInstructionOpcode.Expression
        };

        return opcode != IrInstructionOpcode.Expression;
    }

    private static string FormatElementwiseAssignmentPayload(LoweredElementwiseAssignmentModel assignment)
        => string.Join(
            "|",
            "ASSIGN1",
            assignment.DestinationResourceName,
            assignment.IndexName,
            FormatElementwiseOperation(assignment.Operation),
            assignment.LeftOperand,
            assignment.RightOperandKind == LoweredElementwiseAssignmentOperandKind.None ? string.Empty : assignment.RightOperand);

    private static string FormatElementwiseOperation(LoweredElementwiseAssignmentOperation operation)
        => operation switch
        {
            LoweredElementwiseAssignmentOperation.Copy => "copy",
            LoweredElementwiseAssignmentOperation.Add => "add",
            LoweredElementwiseAssignmentOperation.Subtract => "sub",
            LoweredElementwiseAssignmentOperation.Multiply => "mul",
            LoweredElementwiseAssignmentOperation.Divide => "div",
            _ => throw new InvalidOperationException($"Unsupported elementwise assignment operation '{operation}'.")
        };

    private static string NormalizeSource(SyntaxNodeOrToken nodeOrToken)
        => nodeOrToken.ToString()
            .Replace("\r\n", "\n")
            .Replace('\r', '\n')
            .Trim();

    private readonly record struct SerializedResource(uint Binding, byte Kind, byte Access, uint NameStringId, uint ElementTypeStringId);

    private readonly record struct SerializedInstruction(byte Opcode, byte OperandKind, ushort Reserved, uint OperandStringId, int SyntaxStart);

    private readonly record struct SerializedElementwiseAssignment(
        uint InstructionIndex,
        uint DestinationBinding,
        uint LeftBinding,
        uint RightBinding,
        byte Operation,
        byte RightOperandKind,
        uint IndexStringId,
        uint RightLiteralStringId);

    private readonly record struct SerializedElementwiseExpressionSection(
        IReadOnlyList<SerializedElementwiseExpressionAssignment> Assignments,
        IReadOnlyList<SerializedExpressionNode> Nodes,
        IReadOnlyList<uint> ArgumentIndices);

    private readonly record struct SerializedElementwiseExpressionAssignment(
        uint InstructionIndex,
        uint DestinationBinding,
        uint IndexStringId,
        uint RootNodeIndex);

    private readonly record struct SerializedExpressionNode(
        byte Kind,
        byte Operation,
        uint ResourceBinding,
        uint IndexStringId,
        uint LiteralStringId,
        uint TypeStringId,
        uint LeftNodeIndex,
        uint RightNodeIndex,
        uint SymbolStringId,
        uint FirstArgumentIndex,
        uint ArgumentCount);

    private readonly record struct SerializedSection(uint Kind, byte[] Payload);

    private readonly record struct SerializedAdAnnotation(
        uint Role,
        uint Binding,
        uint NameStringId,
        uint ResourceNameStringId,
        uint TypeNameStringId,
        uint IndexNameStringId,
        uint SourceKind,
        uint ElementCount);

    private enum IrInstructionOpcode : byte
    {
        LocalDeclaration = 1,
        Assignment = 2,
        Return = 3,
        If = 4,
        For = 5,
        While = 6,
        Do = 7,
        Break = 8,
        Continue = 9,
        Invocation = 10,
        ResourceAccess = 11,
        Expression = 12,
        BeginBlock = 13,
        Else = 14,
        EndBlock = 15,
        WorkgroupBarrier = 16,
        MemoryBarrier = 17,
        FullBarrier = 18,
        AtomicAdd = 19,
        AtomicSub = 20,
        AtomicMin = 21,
        AtomicMax = 22,
        AtomicAnd = 23,
        AtomicOr = 24,
        AtomicXor = 25,
        AtomicExchange = 26,
        AtomicCompareExchange = 27,
        SharedMemoryDeclaration = 28
    }

    private enum IrOperandKind : byte
    {
        None = 0,
        Source = 1,
        ElementwiseAssignment = 2,
        Symbol = 3
    }

    private sealed class StringTable
    {
        private readonly Dictionary<string, uint> ids = new(StringComparer.Ordinal);
        private readonly List<string> values = [];

        public uint Add(string value)
        {
            if (ids.TryGetValue(value, out var id))
            {
                return id;
            }

            id = (uint)values.Count;
            values.Add(value);
            ids.Add(value, id);
            return id;
        }

        public byte[] ToBytes()
        {
            using var stream = new MemoryStream();
            using var writer = new BinaryWriter(stream, Encoding.UTF8);
            writer.Write((uint)values.Count);
            foreach (var value in values)
            {
                var bytes = Encoding.UTF8.GetBytes(value);
                writer.Write((uint)bytes.Length);
                writer.Write(bytes);
            }

            return stream.ToArray();
        }
    }
}
