using System.Text;

namespace Feather.Generators.Model;

/// <summary>
/// Serializes a typed <see cref="ShaderModuleModel"/> into FEIR section 7.
/// The section is self-describing: every table is addressed by an explicit
/// offset/count pair, and variable-width child/argument lists are represented
/// as ranges into flat arrays.
/// </summary>
internal static class ShaderIrModuleWriter
{
    public const uint SectionKind = 7;

    private const uint NoIdx = uint.MaxValue;
    private const ushort MajorVersion = 1;
    private const ushort MinorVersion = 1;
    private const byte LittleEndian = 1;
    private const int HeaderSize = 104;

    public static byte[] WriteModule(ShaderModuleModel module)
    {
        var strings = new StringTable();
        var types = new TypeTab(module.Structs.Items);
        var functions = new List<SerFuncRec>();
        var parameters = new List<SerParamRec>();
        var statements = new List<SerStmtRec>();
        var expressions = new List<SerExprRec>();
        var lvalues = new List<SerLValRec>();
        var children = new List<uint>();
        var arguments = new List<uint>();

        var entryId = SerFunc(module.EntryPoint, strings, types, functions, parameters, statements, expressions, lvalues, children, arguments);
        foreach (var callable in module.Callables.Items)
        {
            SerFunc(callable, strings, types, functions, parameters, statements, expressions, lvalues, children, arguments);
        }

        types.CompleteStructTypes();
        var structFields = BuildStructFields(types, strings);
        var stringBytes = strings.ToBytes();

        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        writer.Write(new byte[HeaderSize]);

        var functionOffset = WriteTable(stream, () =>
        {
            foreach (var function in functions)
            {
                writer.Write(function.Kind);
                writer.Write(function.NameId);
                writer.Write(function.MangledNameId);
                writer.Write(function.ReturnTypeId);
                writer.Write(function.FirstParameter);
                writer.Write(function.ParameterCount);
                writer.Write(function.BodyStatementId);
            }
        });

        var typeOffset = WriteTable(stream, () =>
        {
            foreach (var type in types.Table)
            {
                writer.Write(type.Kind);
                writer.Write(type.A);
                writer.Write(type.B);
                writer.Write(type.C);
                writer.Write(type.D);
            }
        });

        var structOffset = WriteTable(stream, () =>
        {
            foreach (var structure in structFields.Structs)
            {
                writer.Write(structure.NameId);
                writer.Write(structure.FullyQualifiedNameId);
                writer.Write(structure.FirstField);
                writer.Write(structure.FieldCount);
                writer.Write(structure.SizeInBytes);
                writer.Write(structure.Alignment);
            }
        });

        var structFieldOffset = WriteTable(stream, () =>
        {
            foreach (var field in structFields.Fields)
            {
                writer.Write(field.NameId);
                writer.Write(field.TypeId);
                writer.Write(field.Offset);
                writer.Write(field.SizeInBytes);
                writer.Write(field.Flags);
            }
        });

        var statementOffset = WriteTable(stream, () =>
        {
            foreach (var statement in statements)
            {
                writer.Write(statement.Kind);
                writer.Write(statement.A);
                writer.Write(statement.B);
                writer.Write(statement.C);
                writer.Write(statement.Op);
                writer.Write(statement.NameId);
                writer.Write(statement.FirstChild);
                writer.Write(statement.ChildCount);
            }
        });

        var expressionOffset = WriteTable(stream, () =>
        {
            foreach (var expression in expressions)
            {
                writer.Write(expression.Kind);
                writer.Write(expression.TypeId);
                writer.Write(expression.A);
                writer.Write(expression.B);
                writer.Write(expression.C);
                writer.Write(expression.NameId);
                writer.Write(expression.Op);
                writer.Write(expression.FirstArgument);
                writer.Write(expression.ArgumentCount);
            }
        });

        var lvalueOffset = WriteTable(stream, () =>
        {
            foreach (var lvalue in lvalues)
            {
                writer.Write(lvalue.Kind);
                writer.Write(lvalue.TypeId);
                writer.Write(lvalue.A);
                writer.Write(lvalue.B);
                writer.Write(lvalue.C);
                writer.Write(lvalue.NameId);
            }
        });

        var childOffset = WriteTable(stream, () =>
        {
            foreach (var child in children)
            {
                writer.Write(child);
            }
        });

        var argumentOffset = WriteTable(stream, () =>
        {
            foreach (var argument in arguments)
            {
                writer.Write(argument);
            }
        });

        var parameterOffset = WriteTable(stream, () =>
        {
            foreach (var parameter in parameters)
            {
                writer.Write(parameter.Direction);
                writer.Write(parameter.NameId);
                writer.Write(parameter.TypeId);
            }
        });

        var stringOffset = CheckedOffset(stream);
        writer.Write(stringBytes);
        var stringLength = checked((uint)stringBytes.Length);

        stream.Position = 0;
        writer.Write(Encoding.ASCII.GetBytes("FTIR"));
        writer.Write(MajorVersion);
        writer.Write(MinorVersion);
        writer.Write(LittleEndian);
        writer.Write((byte)0);
        writer.Write((ushort)HeaderSize);
        writer.Write(entryId);
        WriteRange(writer, functionOffset, functions.Count);
        WriteRange(writer, typeOffset, types.Table.Count);
        WriteRange(writer, structOffset, structFields.Structs.Count);
        WriteRange(writer, structFieldOffset, structFields.Fields.Count);
        WriteRange(writer, statementOffset, statements.Count);
        WriteRange(writer, expressionOffset, expressions.Count);
        WriteRange(writer, lvalueOffset, lvalues.Count);
        WriteRange(writer, childOffset, children.Count);
        WriteRange(writer, argumentOffset, arguments.Count);
        WriteRange(writer, parameterOffset, parameters.Count);
        writer.Write(stringOffset);
        writer.Write(stringLength);

        return stream.ToArray();
    }

    private static uint SerFunc(
        ShaderFunctionModel function,
        StringTable strings,
        TypeTab types,
        List<SerFuncRec> functions,
        List<SerParamRec> parameters,
        List<SerStmtRec> statements,
        List<SerExprRec> expressions,
        List<SerLValRec> lvalues,
        List<uint> children,
        List<uint> arguments)
    {
        var firstParameter = function.Parameters.Items.Count == 0 ? NoIdx : checked((uint)parameters.Count);
        foreach (var parameter in function.Parameters.Items)
        {
            parameters.Add(new SerParamRec(
                (byte)parameter.Direction,
                strings.Add(parameter.Name),
                types.Id(parameter.Type)));
        }

        var bodyId = SerBlock(function.Body, strings, types, statements, expressions, lvalues, children, arguments);
        var id = checked((uint)functions.Count);
        functions.Add(new SerFuncRec(
            (byte)function.Kind,
            strings.Add(function.Name),
            strings.Add(function.MangledName),
            types.Id(function.ReturnType),
            firstParameter,
            checked((uint)function.Parameters.Items.Count),
            bodyId));
        return id;
    }

    private static uint SerBlock(
        ShaderBlockStatement block,
        StringTable strings,
        TypeTab types,
        List<SerStmtRec> statements,
        List<SerExprRec> expressions,
        List<SerLValRec> lvalues,
        List<uint> children,
        List<uint> arguments)
    {
        var childIds = new List<uint>();
        foreach (var statement in block.Statements.Items)
        {
            childIds.Add(SerStmt(statement, strings, types, statements, expressions, lvalues, children, arguments));
        }

        var firstChild = childIds.Count == 0 ? NoIdx : checked((uint)children.Count);
        children.AddRange(childIds);
        return AddS(statements, new SerStmtRec
        {
            Kind = 1,
            FirstChild = firstChild,
            ChildCount = checked((uint)childIds.Count)
        });
    }

    private static uint SerStmt(
        ShaderStatement statement,
        StringTable strings,
        TypeTab types,
        List<SerStmtRec> statements,
        List<SerExprRec> expressions,
        List<SerLValRec> lvalues,
        List<uint> children,
        List<uint> arguments)
    {
        switch (statement)
        {
            case ShaderBlockStatement block:
                return SerBlock(block, strings, types, statements, expressions, lvalues, children, arguments);
            case ShaderLocalDeclarationStatement declaration:
                return AddS(statements, new SerStmtRec
                {
                    Kind = 2,
                    Op = types.Id(declaration.Type),
                    NameId = strings.Add(declaration.VariableName),
                    A = declaration.Initializer is { } initializer ? SerExpr(initializer, strings, types, expressions, lvalues, arguments) : NoIdx
                });
            case ShaderAssignmentStatement assignment:
                return AddS(statements, new SerStmtRec
                {
                    Kind = 3,
                    A = SerLVal(assignment.Target, strings, types, expressions, lvalues, arguments),
                    B = SerExpr(assignment.Value, strings, types, expressions, lvalues, arguments)
                });
            case ShaderCompoundAssignmentStatement compound:
                return AddS(statements, new SerStmtRec
                {
                    Kind = 4,
                    A = SerLVal(compound.Target, strings, types, expressions, lvalues, arguments),
                    B = SerExpr(compound.Value, strings, types, expressions, lvalues, arguments),
                    Op = (uint)compound.Operator
                });
            case ShaderIfStatement ifStatement:
                return AddS(statements, new SerStmtRec
                {
                    Kind = 5,
                    A = SerExpr(ifStatement.Condition, strings, types, expressions, lvalues, arguments),
                    B = SerBlock(ifStatement.Then, strings, types, statements, expressions, lvalues, children, arguments),
                    C = ifStatement.Else is { } elseBlock ? SerBlock(elseBlock, strings, types, statements, expressions, lvalues, children, arguments) : NoIdx
                });
            case ShaderForStatement forStatement:
                return AddS(statements, new SerStmtRec
                {
                    Kind = 6,
                    A = forStatement.Init is { } init ? SerStmt(init, strings, types, statements, expressions, lvalues, children, arguments) : NoIdx,
                    B = forStatement.Condition is { } condition ? SerExpr(condition, strings, types, expressions, lvalues, arguments) : NoIdx,
                    C = forStatement.Step is { } step ? SerStmt(step, strings, types, statements, expressions, lvalues, children, arguments) : NoIdx,
                    Op = SerBlock(forStatement.Body, strings, types, statements, expressions, lvalues, children, arguments)
                });
            case ShaderWhileStatement whileStatement:
                return AddS(statements, new SerStmtRec
                {
                    Kind = 7,
                    A = SerExpr(whileStatement.Condition, strings, types, expressions, lvalues, arguments),
                    B = SerBlock(whileStatement.Body, strings, types, statements, expressions, lvalues, children, arguments)
                });
            case ShaderDoWhileStatement doWhileStatement:
                return AddS(statements, new SerStmtRec
                {
                    Kind = 8,
                    A = SerBlock(doWhileStatement.Body, strings, types, statements, expressions, lvalues, children, arguments),
                    B = SerExpr(doWhileStatement.Condition, strings, types, expressions, lvalues, arguments)
                });
            case ShaderBreakStatement:
                return AddS(statements, new SerStmtRec { Kind = 9 });
            case ShaderContinueStatement:
                return AddS(statements, new SerStmtRec { Kind = 10 });
            case ShaderReturnStatement returnStatement:
                return AddS(statements, new SerStmtRec
                {
                    Kind = 11,
                    A = returnStatement.Value is { } value ? SerExpr(value, strings, types, expressions, lvalues, arguments) : NoIdx
                });
            case ShaderExpressionStatement expressionStatement:
                return AddS(statements, new SerStmtRec
                {
                    Kind = 12,
                    A = SerExpr(expressionStatement.Expression, strings, types, expressions, lvalues, arguments)
                });
            case ShaderBarrierStatement barrier:
                return AddS(statements, new SerStmtRec { Kind = 13, Op = (uint)barrier.Kind });
            case ShaderIncrementDecrementStatement increment:
                return AddS(statements, new SerStmtRec
                {
                    Kind = 14,
                    A = SerLVal(increment.Target, strings, types, expressions, lvalues, arguments),
                    Op = (uint)(increment.IsIncrement ? 1 : 0) | (uint)(increment.IsPrefix ? 2 : 0)
                });
            case ShaderSharedMemoryDeclarationStatement shared:
                return AddS(statements, new SerStmtRec
                {
                    Kind = 15,
                    A = checked((uint)shared.Length),
                    Op = types.Id(shared.ElementType),
                    NameId = strings.Add(shared.VariableName)
                });
            default:
                throw new InvalidOperationException($"Unsupported shader statement '{statement.GetType().Name}'.");
        }
    }

    private static uint SerExpr(
        ShaderExpression expression,
        StringTable strings,
        TypeTab types,
        List<SerExprRec> expressions,
        List<SerLValRec> lvalues,
        List<uint> arguments)
    {
        switch (expression)
        {
            case ShaderLiteralExpression literal:
                return AddE(expressions, new SerExprRec { Kind = 1, TypeId = types.Id(literal.Type), NameId = strings.Add(literal.ValueText) });
            case ShaderLocalReferenceExpression local:
                return AddE(expressions, new SerExprRec { Kind = 2, TypeId = types.Id(local.Type), NameId = strings.Add(local.Name) });
            case ShaderParameterReferenceExpression parameter:
                return AddE(expressions, new SerExprRec { Kind = 3, TypeId = types.Id(parameter.Type), NameId = strings.Add(parameter.Name) });
            case ShaderFieldReferenceExpression field:
                return AddE(expressions, new SerExprRec
                {
                    Kind = 4,
                    TypeId = types.Id(field.Type),
                    A = SerExpr(field.Instance, strings, types, expressions, lvalues, arguments),
                    NameId = strings.Add(field.Field.Name)
                });
            case ShaderResourceElementExpression resource:
                return AddE(expressions, new SerExprRec
                {
                    Kind = 5,
                    TypeId = types.Id(resource.Type),
                    A = SerExpr(resource.Index, strings, types, expressions, lvalues, arguments),
                    NameId = strings.Add(resource.ResourceName)
                });
            case ShaderUnaryExpression unary:
                return AddE(expressions, new SerExprRec
                {
                    Kind = 6,
                    TypeId = types.Id(unary.Type),
                    Op = (uint)unary.Operator,
                    A = SerExpr(unary.Operand, strings, types, expressions, lvalues, arguments)
                });
            case ShaderBinaryExpression binary:
                return AddE(expressions, new SerExprRec
                {
                    Kind = 7,
                    TypeId = types.Id(binary.Type),
                    Op = (uint)binary.Operator,
                    A = SerExpr(binary.Left, strings, types, expressions, lvalues, arguments),
                    B = SerExpr(binary.Right, strings, types, expressions, lvalues, arguments)
                });
            case ShaderComparisonExpression comparison:
                return AddE(expressions, new SerExprRec
                {
                    Kind = 8,
                    TypeId = types.Id(comparison.Type),
                    Op = (uint)comparison.Operator,
                    A = SerExpr(comparison.Left, strings, types, expressions, lvalues, arguments),
                    B = SerExpr(comparison.Right, strings, types, expressions, lvalues, arguments)
                });
            case ShaderLogicalExpression logical:
                return AddE(expressions, new SerExprRec
                {
                    Kind = 9,
                    TypeId = types.Id(logical.Type),
                    Op = (uint)logical.Operator,
                    A = SerExpr(logical.Left, strings, types, expressions, lvalues, arguments),
                    B = SerExpr(logical.Right, strings, types, expressions, lvalues, arguments)
                });
            case ShaderConditionalExpression conditional:
                return AddE(expressions, new SerExprRec
                {
                    Kind = 10,
                    TypeId = types.Id(conditional.Type),
                    A = SerExpr(conditional.Condition, strings, types, expressions, lvalues, arguments),
                    B = SerExpr(conditional.WhenTrue, strings, types, expressions, lvalues, arguments),
                    C = SerExpr(conditional.WhenFalse, strings, types, expressions, lvalues, arguments)
                });
            case ShaderConversionExpression conversion:
                return AddE(expressions, new SerExprRec
                {
                    Kind = 11,
                    TypeId = types.Id(conversion.Type),
                    A = SerExpr(conversion.Operand, strings, types, expressions, lvalues, arguments)
                });
            case ShaderConstructorExpression constructor:
                return SerCallLikeExpression(12, constructor.Type, NoIdx, 0, constructor.Arguments.Items, strings, types, expressions, lvalues, arguments);
            case ShaderIntrinsicCallExpression intrinsic:
                return SerCallLikeExpression(13, intrinsic.Type, strings.Add(intrinsic.IntrinsicName), 0, intrinsic.Arguments.Items, strings, types, expressions, lvalues, arguments);
            case ShaderCallableCallExpression call:
                return SerCallLikeExpression(14, call.Type, strings.Add(call.CallableName), 0, call.Arguments.Items, strings, types, expressions, lvalues, arguments);
            case ShaderAtomicExpression atomic:
                return SerAtomicExpression(atomic, strings, types, expressions, lvalues, arguments);
            case ShaderTextureSampleExpression sample:
                return SerTextureSampleExpression(sample, strings, types, expressions, lvalues, arguments);
            case ShaderSwizzleExpression swizzle:
                return AddE(expressions, new SerExprRec
                {
                    Kind = 15,
                    TypeId = types.Id(swizzle.Type),
                    A = SerExpr(swizzle.Vector, strings, types, expressions, lvalues, arguments),
                    NameId = strings.Add(swizzle.SwizzleComponents)
                });
            case ShaderMemberAccessExpression member:
                return AddE(expressions, new SerExprRec
                {
                    Kind = 16,
                    TypeId = types.Id(member.Type),
                    A = SerExpr(member.Instance, strings, types, expressions, lvalues, arguments),
                    NameId = strings.Add(member.Field.Name)
                });
            case ShaderIndexAccessExpression index:
                return AddE(expressions, new SerExprRec
                {
                    Kind = 17,
                    TypeId = types.Id(index.Type),
                    A = SerExpr(index.Array, strings, types, expressions, lvalues, arguments),
                    B = SerExpr(index.Index, strings, types, expressions, lvalues, arguments)
                });
            case ShaderBuiltinExpression builtin:
                return AddE(expressions, new SerExprRec { Kind = 18, TypeId = types.Id(builtin.Type), Op = (uint)builtin.Kind });
            case ShaderPushConstantExpression pushConstant:
                return AddE(expressions, new SerExprRec
                {
                    Kind = 19,
                    TypeId = types.Id(pushConstant.Type),
                    NameId = strings.Add(pushConstant.ResourceName),
                    Op = pushConstant.Binding
                });
            case ShaderMatrixColumnExpression matrixColumn:
                return AddE(expressions, new SerExprRec
                {
                    Kind = 20,
                    TypeId = types.Id(matrixColumn.Type),
                    A = SerExpr(matrixColumn.Matrix, strings, types, expressions, lvalues, arguments),
                    B = SerExpr(matrixColumn.ColumnIndex, strings, types, expressions, lvalues, arguments)
                });
            case ShaderSharedMemoryElementExpression shared:
                return AddE(expressions, new SerExprRec
                {
                    Kind = 21,
                    TypeId = types.Id(shared.Type),
                    A = SerExpr(shared.Index, strings, types, expressions, lvalues, arguments),
                    NameId = strings.Add(shared.Name)
                });
            default:
                throw new InvalidOperationException($"Unsupported shader expression '{expression.GetType().Name}'.");
        }
    }

    private static uint SerAtomicExpression(
        ShaderAtomicExpression atomic,
        StringTable strings,
        TypeTab types,
        List<SerExprRec> expressions,
        List<SerLValRec> lvalues,
        List<uint> arguments)
    {
        var argumentIds = new List<uint>(atomic.Arguments.Items.Count);
        foreach (var argument in atomic.Arguments.Items)
        {
            argumentIds.Add(SerExpr(argument, strings, types, expressions, lvalues, arguments));
        }

        var firstArgument = argumentIds.Count == 0 ? NoIdx : checked((uint)arguments.Count);
        arguments.AddRange(argumentIds);
        return AddE(expressions, new SerExprRec
        {
            Kind = 22,
            TypeId = types.Id(atomic.Type),
            A = SerLVal(atomic.Target, strings, types, expressions, lvalues, arguments),
            Op = (uint)atomic.Operation,
            FirstArgument = firstArgument,
            ArgumentCount = checked((uint)argumentIds.Count)
        });
    }

    private static uint SerTextureSampleExpression(
        ShaderTextureSampleExpression sample,
        StringTable strings,
        TypeTab types,
        List<SerExprRec> expressions,
        List<SerLValRec> lvalues,
        List<uint> arguments)
    {
        var sourceArguments = new List<ShaderExpression> { sample.Texture, sample.Sampler, sample.Uv };
        if (sample.Lod is { } lod)
        {
            sourceArguments.Add(lod);
        }
        if (sample.Ddx is { } ddx)
        {
            sourceArguments.Add(ddx);
        }
        if (sample.Ddy is { } ddy)
        {
            sourceArguments.Add(ddy);
        }

        return SerCallLikeExpression(
            23,
            sample.Type,
            NoIdx,
            (uint)sample.Operation,
            sourceArguments,
            strings,
            types,
            expressions,
            lvalues,
            arguments);
    }

    private static uint SerCallLikeExpression(
        byte kind,
        ShaderType type,
        uint nameId,
        uint op,
        IReadOnlyList<ShaderExpression> sourceArguments,
        StringTable strings,
        TypeTab types,
        List<SerExprRec> expressions,
        List<SerLValRec> lvalues,
        List<uint> arguments)
    {
        var argumentIds = new List<uint>(sourceArguments.Count);
        foreach (var argument in sourceArguments)
        {
            argumentIds.Add(SerExpr(argument, strings, types, expressions, lvalues, arguments));
        }

        var firstArgument = argumentIds.Count == 0 ? NoIdx : checked((uint)arguments.Count);
        arguments.AddRange(argumentIds);
        return AddE(expressions, new SerExprRec
        {
            Kind = kind,
            TypeId = types.Id(type),
            NameId = nameId,
            Op = op,
            FirstArgument = firstArgument,
            ArgumentCount = checked((uint)argumentIds.Count)
        });
    }

    private static uint SerLVal(
        ShaderLValue lvalue,
        StringTable strings,
        TypeTab types,
        List<SerExprRec> expressions,
        List<SerLValRec> lvalues,
        List<uint> arguments)
    {
        switch (lvalue)
        {
            case ShaderLocalLValue local:
                return AddL(lvalues, new SerLValRec { Kind = 1, TypeId = types.Id(local.Type), NameId = strings.Add(local.Name) });
            case ShaderParameterLValue parameter:
                return AddL(lvalues, new SerLValRec { Kind = 2, TypeId = types.Id(parameter.Type), NameId = strings.Add(parameter.Name) });
            case ShaderFieldLValue field:
                return AddL(lvalues, new SerLValRec
                {
                    Kind = 3,
                    TypeId = types.Id(field.Type),
                    A = field.Instance is { } instance ? SerLVal(instance, strings, types, expressions, lvalues, arguments) : NoIdx,
                    NameId = strings.Add(field.Field.Name)
                });
            case ShaderResourceElementLValue resource:
                return AddL(lvalues, new SerLValRec
                {
                    Kind = 4,
                    TypeId = types.Id(resource.Type),
                    A = SerExpr(resource.Index, strings, types, expressions, lvalues, arguments),
                    NameId = strings.Add(resource.ResourceName)
                });
            case ShaderSwizzleLValue swizzle:
                return AddL(lvalues, new SerLValRec
                {
                    Kind = 5,
                    TypeId = types.Id(swizzle.Type),
                    A = SerExpr(swizzle.Vector, strings, types, expressions, lvalues, arguments),
                    NameId = strings.Add(swizzle.SwizzleComponents)
                });
            case ShaderMemberAccessLValue member:
                return AddL(lvalues, new SerLValRec
                {
                    Kind = 6,
                    TypeId = types.Id(member.Type),
                    A = SerLVal(member.Instance, strings, types, expressions, lvalues, arguments),
                    NameId = strings.Add(member.Field.Name)
                });
            case ShaderIndexAccessLValue index:
                return AddL(lvalues, new SerLValRec
                {
                    Kind = 7,
                    TypeId = types.Id(index.Type),
                    A = SerLVal(index.Array, strings, types, expressions, lvalues, arguments),
                    B = SerExpr(index.Index, strings, types, expressions, lvalues, arguments)
                });
            case ShaderMatrixColumnLValue matrixColumn:
                return AddL(lvalues, new SerLValRec
                {
                    Kind = 8,
                    TypeId = types.Id(matrixColumn.Type),
                    A = SerExpr(matrixColumn.Matrix, strings, types, expressions, lvalues, arguments),
                    B = SerExpr(matrixColumn.ColumnIndex, strings, types, expressions, lvalues, arguments)
                });
            case ShaderSharedMemoryElementLValue shared:
                return AddL(lvalues, new SerLValRec
                {
                    Kind = 9,
                    TypeId = types.Id(shared.Type),
                    A = SerExpr(shared.Index, strings, types, expressions, lvalues, arguments),
                    NameId = strings.Add(shared.Name)
                });
            default:
                throw new InvalidOperationException($"Unsupported shader l-value '{lvalue.GetType().Name}'.");
        }
    }

    private static StructTables BuildStructFields(TypeTab types, StringTable strings)
    {
        var structs = new List<SerStructRec>();
        var fields = new List<SerStructFieldRec>();
        foreach (var structure in types.Structs)
        {
            var fieldCount = structure.Fields.Items.Count;
            var firstField = fieldCount == 0 ? NoIdx : checked((uint)fields.Count);
            foreach (var field in structure.Fields.Items)
            {
                fields.Add(new SerStructFieldRec(
                    strings.Add(field.Name),
                    types.Id(field.Type),
                    checked((uint)field.Offset),
                    checked((uint)field.SizeInBytes),
                    (uint)field.Flags));
            }

            structs.Add(new SerStructRec(
                strings.Add(structure.Name),
                strings.Add(structure.FullyQualifiedMetadataName),
                firstField,
                checked((uint)fieldCount),
                checked((uint)structure.SizeInBytes),
                checked((uint)structure.Alignment)));
        }

        return new StructTables(structs, fields);
    }

    private static uint AddS(List<SerStmtRec> statements, SerStmtRec statement)
    {
        var id = checked((uint)statements.Count);
        statements.Add(statement);
        return id;
    }

    private static uint AddE(List<SerExprRec> expressions, SerExprRec expression)
    {
        var id = checked((uint)expressions.Count);
        expressions.Add(expression);
        return id;
    }

    private static uint AddL(List<SerLValRec> lvalues, SerLValRec lvalue)
    {
        var id = checked((uint)lvalues.Count);
        lvalues.Add(lvalue);
        return id;
    }

    private static uint WriteTable(MemoryStream stream, Action write)
    {
        var offset = CheckedOffset(stream);
        write();
        return offset;
    }

    private static uint CheckedOffset(MemoryStream stream)
        => checked((uint)stream.Position);

    private static void WriteRange(BinaryWriter writer, uint offset, int count)
    {
        writer.Write(offset);
        writer.Write(checked((uint)count));
    }

    private sealed record SerFuncRec(
        byte Kind,
        uint NameId,
        uint MangledNameId,
        uint ReturnTypeId,
        uint FirstParameter,
        uint ParameterCount,
        uint BodyStatementId);

    private sealed record SerParamRec(byte Direction, uint NameId, uint TypeId);

    private sealed class SerStmtRec
    {
        public byte Kind;
        public uint A = NoIdx;
        public uint B = NoIdx;
        public uint C = NoIdx;
        public uint Op;
        public uint NameId = NoIdx;
        public uint FirstChild = NoIdx;
        public uint ChildCount;
    }

    private sealed class SerExprRec
    {
        public byte Kind;
        public uint TypeId = NoIdx;
        public uint A = NoIdx;
        public uint B = NoIdx;
        public uint C = NoIdx;
        public uint NameId = NoIdx;
        public uint Op;
        public uint FirstArgument = NoIdx;
        public uint ArgumentCount;
    }

    private sealed class SerLValRec
    {
        public byte Kind;
        public uint TypeId = NoIdx;
        public uint A = NoIdx;
        public uint B = NoIdx;
        public uint C = NoIdx;
        public uint NameId = NoIdx;
    }

    private sealed record SerStructRec(
        uint NameId,
        uint FullyQualifiedNameId,
        uint FirstField,
        uint FieldCount,
        uint SizeInBytes,
        uint Alignment);

    private sealed record SerStructFieldRec(uint NameId, uint TypeId, uint Offset, uint SizeInBytes, uint Flags);

    private sealed record StructTables(IReadOnlyList<SerStructRec> Structs, IReadOnlyList<SerStructFieldRec> Fields);

    private sealed class TypeTab
    {
        public readonly List<SerTypeRec> Table = new();
        public readonly List<ShaderStructType> Structs = new();

        private readonly Dictionary<ShaderType, uint> _ids = new();
        private readonly Dictionary<ShaderStructType, uint> _structIds = new();

        public TypeTab(IEnumerable<ShaderStructType> structs)
        {
            foreach (var structure in structs)
            {
                StructId(structure);
            }
        }

        public uint Id(ShaderType type)
        {
            if (_ids.TryGetValue(type, out var existing))
            {
                return existing;
            }

            SerTypeRec record;
            switch (type)
            {
                case ShaderPrimitiveType primitive:
                    record = new SerTypeRec(1, (uint)primitive.Kind, PrimitiveBitWidth(primitive), 0, 0);
                    break;
                case ShaderVectorType vector:
                    record = new SerTypeRec(2, Id(vector.ElementType), checked((uint)vector.ComponentCount), 0, 0);
                    break;
                case ShaderMatrixType matrix:
                    record = new SerTypeRec(3, Id(matrix.ElementType), checked((uint)matrix.Rows), checked((uint)matrix.Columns), 0);
                    break;
                case ShaderStructType structure:
                    record = new SerTypeRec(4, StructId(structure), 0, 0, 0);
                    break;
                case ShaderArrayType array:
                    record = new SerTypeRec(5, Id(array.ElementType), array.Length is { } length ? checked((uint)length) : NoIdx, 0, 0);
                    break;
                case ShaderResourceWrapperType resource:
                    record = new SerTypeRec(6, (uint)resource.Kind, Id(resource.ElementType), (uint)resource.Access, 0);
                    break;
                case ShaderVoidType:
                    record = new SerTypeRec(7, 0, 0, 0, 0);
                    break;
                default:
                    throw new InvalidOperationException($"Unsupported shader type '{type.GetType().Name}'.");
            }

            var id = checked((uint)Table.Count);
            _ids[type] = id;
            Table.Add(record);
            return id;
        }

        public void CompleteStructTypes()
        {
            for (var i = 0; i < Structs.Count; i++)
            {
                foreach (var field in Structs[i].Fields.Items)
                {
                    Id(field.Type);
                }
            }
        }

        private uint StructId(ShaderStructType structure)
        {
            if (_structIds.TryGetValue(structure, out var existing))
            {
                return existing;
            }

            var id = checked((uint)Structs.Count);
            _structIds[structure] = id;
            Structs.Add(structure);
            return id;
        }

        private static uint PrimitiveBitWidth(ShaderPrimitiveType primitive)
            => primitive.CSharpTypeName switch
            {
                "byte" or "sbyte" => 8,
                "short" or "ushort" => 16,
                _ => 32
            };
    }

    private readonly record struct SerTypeRec(byte Kind, uint A, uint B, uint C, uint D);

    internal sealed class StringTable
    {
        private readonly Dictionary<string, uint> _ids = new(StringComparer.Ordinal);
        private readonly List<string> _values = new();

        public uint Add(string value)
        {
            if (_ids.TryGetValue(value, out var id))
            {
                return id;
            }

            id = checked((uint)_values.Count);
            _values.Add(value);
            _ids[value] = id;
            return id;
        }

        public byte[] ToBytes()
        {
            using var stream = new MemoryStream();
            using var writer = new BinaryWriter(stream, Encoding.UTF8);
            writer.Write(checked((uint)_values.Count));
            foreach (var value in _values)
            {
                var bytes = Encoding.UTF8.GetBytes(value);
                writer.Write(checked((uint)bytes.Length));
                writer.Write(bytes);
            }

            return stream.ToArray();
        }
    }
}
