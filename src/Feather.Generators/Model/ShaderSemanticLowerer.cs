using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Operations;

namespace Feather.Generators.Model;

internal static class ShaderSemanticLowerer
{
    public static EquatableArray<LoweredShaderInstructionModel> Lower(
        ShaderModel model,
        SemanticModel semanticModel,
        CancellationToken cancellationToken)
    {
        var instructions = new List<LoweredShaderInstructionModel>();
        var entry = model.EntryPointSyntax;
        if (entry?.Body is null)
        {
            return new EquatableArray<LoweredShaderInstructionModel>(instructions);
        }

        // The IR writer still aligns semantic records with syntax spans, so keep discovery ordered by source node kind
        // while delegating resource, expression, and invocation meaning to focused lowerers.
        foreach (var assignment in entry.Body.DescendantNodes().OfType<AssignmentExpressionSyntax>())
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (semanticModel.GetOperation(assignment, cancellationToken) is IAssignmentOperation operation
                && TryLowerElementwiseAssignment(operation, out var loweredAssignment, out var loweredExpressionAssignment))
            {
                instructions.Add(new LoweredShaderInstructionModel(
                    assignment.SpanStart,
                    LoweredShaderInstructionKind.ElementwiseAssignment,
                    string.Empty,
                    loweredAssignment,
                    loweredExpressionAssignment));
            }
        }

        foreach (var invocation in entry.Body.DescendantNodes().OfType<InvocationExpressionSyntax>())
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (semanticModel.GetOperation(invocation, cancellationToken) is IInvocationOperation operation
                && ShaderInvocationLowerer.TryLowerKnownInvocation(operation, out var kind))
            {
                instructions.Add(new LoweredShaderInstructionModel(
                    invocation.SpanStart,
                    kind,
                    ShaderSemanticFacts.GetMethodMetadataName(operation.TargetMethod),
                    AdAnnotation: TryLowerAdAnnotation(operation, kind, semanticModel, cancellationToken)));
            }
        }

        foreach (var elementAccess in entry.Body.DescendantNodes().OfType<ElementAccessExpressionSyntax>())
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (semanticModel.GetOperation(elementAccess, cancellationToken) is IPropertyReferenceOperation operation
                && ShaderResourceElementLowerer.TryLower(operation, out var element))
            {
                instructions.Add(new LoweredShaderInstructionModel(
                    elementAccess.SpanStart,
                    LoweredShaderInstructionKind.ResourceAccess,
                    ShaderResourceElementLowerer.FormatPayload(element)));
            }
        }


        // Lower local variable declarations.
        foreach (var localDecl in entry.Body.DescendantNodes().OfType<LocalDeclarationStatementSyntax>())
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (semanticModel.GetOperation(localDecl, cancellationToken) is IVariableDeclarationGroupOperation group)
            {
                foreach (var declaration in group.Declarations)
                {
                    foreach (var declarator in declaration.Declarators)
                    {
                        var glslType = GetGlslTypeName(declarator.Symbol.Type);
                        if (glslType is null) continue;
                        LoweredElementwiseExpressionNodeModel? initializer = null;
                        if (declarator.Initializer?.Value is not null
                            && ShaderElementwiseExpressionLowerer.TryLower(declarator.Initializer.Value, null, out var init))
                            initializer = init;
                        instructions.Add(new LoweredShaderInstructionModel(
                            localDecl.SpanStart, LoweredShaderInstructionKind.LocalDeclaration,
                            string.Empty, LocalDeclaration: new LoweredLocalDeclarationModel(
                                declarator.Symbol.Name, glslType, initializer)));
                    }
                }
            }
        }

        // Lower compound assignments (output[i] += value).
        foreach (var assignment in entry.Body.DescendantNodes().OfType<AssignmentExpressionSyntax>())
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (semanticModel.GetOperation(assignment, cancellationToken) is ICompoundAssignmentOperation compoundOp
                && ShaderResourceElementLowerer.TryLower(compoundOp.Target, out var target)
                && TryLowerAssignmentOperation(compoundOp.OperatorKind, out var op)
                && ShaderElementwiseExpressionLowerer.TryLower(compoundOp.Value, target.IndexSymbol, out var value))
            {
                instructions.Add(new LoweredShaderInstructionModel(
                    assignment.SpanStart, LoweredShaderInstructionKind.CompoundAssignment,
                    string.Empty, CompoundAssignment: new LoweredCompoundAssignmentModel(
                        target.ResourceName, target.IndexName, op, value)));
            }
        }

        // Lower assignments to local variables.
        foreach (var assignment in entry.Body.DescendantNodes().OfType<AssignmentExpressionSyntax>())
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (semanticModel.GetOperation(assignment, cancellationToken) is IAssignmentOperation assignOp
                && IsLocalTarget(assignOp.Target, out var localName)
                && !ShaderResourceElementLowerer.TryLower(assignOp.Target, out _))
            {
                if (ShaderElementwiseExpressionLowerer.TryLower(assignOp.Value, null, out var value))
                {
                    instructions.Add(new LoweredShaderInstructionModel(
                        assignment.SpanStart, LoweredShaderInstructionKind.LocalAssignment,
                        string.Empty, LocalAssignment: new LoweredLocalAssignmentModel(localName, value)));
                }
            }
        }

        // Lower if conditions.
        foreach (var ifStatement in entry.Body.DescendantNodes().OfType<IfStatementSyntax>())
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (semanticModel.GetOperation(ifStatement, cancellationToken) is IConditionalOperation { Condition: not null } ifOp
                && ShaderElementwiseExpressionLowerer.TryLower(ifOp.Condition, null, out var condition))
            {
                var condSpan = ifStatement.Condition.SpanStart;
                instructions.Add(new LoweredShaderInstructionModel(
                    condSpan, default, string.Empty,
                    ControlFlowCondition: new LoweredControlFlowConditionModel(
                        condSpan, LoweredControlFlowRole.IfCondition, condition)));
            }
        }

        // Lower for-loop conditions and steps.
        foreach (var forStatement in entry.Body.DescendantNodes().OfType<ForStatementSyntax>())
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (semanticModel.GetOperation(forStatement, cancellationToken) is IForLoopOperation forLoopOp)
            {
                if (forLoopOp.Condition is not null
                    && ShaderElementwiseExpressionLowerer.TryLower(forLoopOp.Condition, null, out var forCondition))
                {
                    instructions.Add(new LoweredShaderInstructionModel(
                        forLoopOp.Condition.Syntax.SpanStart, default, string.Empty,
                        ControlFlowCondition: new LoweredControlFlowConditionModel(
                            forLoopOp.Condition.Syntax.SpanStart, LoweredControlFlowRole.ForCondition, forCondition)));
                }
                foreach (var stepOp in forLoopOp.AtLoopBottom)
                {
                    if (stepOp is IAssignmentOperation assignOp2
                        && ShaderElementwiseExpressionLowerer.TryLower(assignOp2.Value, null, out var stepExpr))
                    {
                        instructions.Add(new LoweredShaderInstructionModel(
                            stepOp.Syntax.SpanStart, default, string.Empty,
                            ControlFlowCondition: new LoweredControlFlowConditionModel(
                                stepOp.Syntax.SpanStart, LoweredControlFlowRole.ForStep, stepExpr)));
                    }
                }
            }
        }

        // Lower while conditions.
        foreach (var whileStatement in entry.Body.DescendantNodes().OfType<WhileStatementSyntax>())
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (semanticModel.GetOperation(whileStatement, cancellationToken) is IConditionalOperation { Condition: not null } whileOp
                && ShaderElementwiseExpressionLowerer.TryLower(whileOp.Condition, null, out var whileCondition))
            {
                var condSpan = whileStatement.Condition.SpanStart;
                instructions.Add(new LoweredShaderInstructionModel(
                    condSpan, default, string.Empty,
                    ControlFlowCondition: new LoweredControlFlowConditionModel(
                        condSpan, LoweredControlFlowRole.WhileCondition, whileCondition)));
            }
        }

        // Lower do-while conditions.
        foreach (var doStatement in entry.Body.DescendantNodes().OfType<DoStatementSyntax>())
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (semanticModel.GetOperation(doStatement, cancellationToken) is IConditionalOperation { Condition: not null } doOp
                && ShaderElementwiseExpressionLowerer.TryLower(doOp.Condition, null, out var doCondition))
            {
                var condSpan = doStatement.Condition.SpanStart;
                instructions.Add(new LoweredShaderInstructionModel(
                    condSpan, default, string.Empty,
                    ControlFlowCondition: new LoweredControlFlowConditionModel(
                        condSpan, LoweredControlFlowRole.DoCondition, doCondition)));
            }
        }

        return new EquatableArray<LoweredShaderInstructionModel>(instructions);
    }

    private static bool TryLowerElementwiseAssignment(
        IAssignmentOperation assignment,
        out LoweredElementwiseAssignmentModel? lowered,
        out LoweredElementwiseExpressionAssignmentModel? expressionLowered)
    {
        lowered = null;
        expressionLowered = null;
        if (!ShaderResourceElementLowerer.TryLower(assignment.Target, out var destination))
        {
            return false;
        }

        if (ShaderElementwiseExpressionLowerer.TryLower(assignment.Value, destination.IndexSymbol, out var expression))
        {
            expressionLowered = new LoweredElementwiseExpressionAssignmentModel(
                destination.ResourceName,
                destination.IndexName,
                expression);
        }

        if (ShaderResourceElementLowerer.TryLower(assignment.Value, out var copySource))
        {
            if (!SymbolEqualityComparer.Default.Equals(destination.IndexSymbol, copySource.IndexSymbol))
            {
                return expressionLowered is not null;
            }

            lowered = new LoweredElementwiseAssignmentModel(
                destination.ResourceName,
                destination.IndexName,
                LoweredElementwiseAssignmentOperation.Copy,
                copySource.ResourceName,
                LoweredElementwiseAssignmentOperandKind.None,
                string.Empty);
            return true;
        }

        if (!ShaderSemanticFacts.TryUnwrapConversion(assignment.Value, out var value) || value is not IBinaryOperation binary)
        {
            return expressionLowered is not null;
        }

        if (!TryLowerAssignmentOperation(binary.OperatorKind, out var operation))
        {
            return expressionLowered is not null;
        }

        if (ShaderResourceElementLowerer.TryLower(binary.LeftOperand, out var left)
            && SymbolEqualityComparer.Default.Equals(destination.IndexSymbol, left.IndexSymbol))
        {
            if (ShaderResourceElementLowerer.TryLower(binary.RightOperand, out var right)
                && SymbolEqualityComparer.Default.Equals(destination.IndexSymbol, right.IndexSymbol))
            {
                lowered = new LoweredElementwiseAssignmentModel(
                    destination.ResourceName,
                    destination.IndexName,
                    operation,
                    left.ResourceName,
                    LoweredElementwiseAssignmentOperandKind.Resource,
                    right.ResourceName);
                return true;
            }

            if (ShaderSemanticFacts.TryFormatNumericLiteral(binary.RightOperand, out var literal))
            {
                lowered = new LoweredElementwiseAssignmentModel(
                    destination.ResourceName,
                    destination.IndexName,
                    operation,
                    left.ResourceName,
                    LoweredElementwiseAssignmentOperandKind.Literal,
                    literal);
                return true;
            }
        }

        // Addition and multiplication are commutative, so the literal can appear on either side without changing IR.
        if (operation is LoweredElementwiseAssignmentOperation.Add or LoweredElementwiseAssignmentOperation.Multiply
            && ShaderSemanticFacts.TryFormatNumericLiteral(binary.LeftOperand, out var leftLiteral)
            && ShaderResourceElementLowerer.TryLower(binary.RightOperand, out var rightResource)
            && SymbolEqualityComparer.Default.Equals(destination.IndexSymbol, rightResource.IndexSymbol))
        {
            lowered = new LoweredElementwiseAssignmentModel(
                destination.ResourceName,
                destination.IndexName,
                operation,
                rightResource.ResourceName,
                LoweredElementwiseAssignmentOperandKind.Literal,
                leftLiteral);
            return true;
        }

        return expressionLowered is not null;
    }

    private static bool TryLowerAssignmentOperation(
        BinaryOperatorKind operatorKind,
        out LoweredElementwiseAssignmentOperation operation)
    {
        operation = operatorKind switch
        {
            BinaryOperatorKind.Add => LoweredElementwiseAssignmentOperation.Add,
            BinaryOperatorKind.Subtract => LoweredElementwiseAssignmentOperation.Subtract,
            BinaryOperatorKind.Multiply => LoweredElementwiseAssignmentOperation.Multiply,
            BinaryOperatorKind.Divide => LoweredElementwiseAssignmentOperation.Divide,
            _ => default
        };
        return operatorKind is BinaryOperatorKind.Add or BinaryOperatorKind.Subtract or BinaryOperatorKind.Multiply or BinaryOperatorKind.Divide;
    }

    private static string? GetGlslTypeName(ITypeSymbol? type)
    {
        if (type is null) return null;
        var typeName = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        return typeName switch
        {
            "float" or "System.Single" => "float",
            "int" or "System.Int32" => "int",
            "bool" or "System.Boolean" => "bool",
            "global::Feather.Math.float2" => "vec2",
            "global::Feather.Math.float3" => "vec3",
            "global::Feather.Math.float4" => "vec4",
            "global::Feather.Math.int2" => "ivec2",
            "global::Feather.Math.int3" => "ivec3",
            "global::Feather.Math.int4" => "ivec4",
            _ => null
        };
    }

    private static bool IsLocalTarget(IOperation target, out string localName)
    {
        localName = string.Empty;
        if (!ShaderSemanticFacts.TryUnwrapConversion(target, out var unwrapped)) return false;
        switch (unwrapped)
        {
            case ILocalReferenceOperation local: localName = local.Local.Name; return true;
            case IParameterReferenceOperation param: localName = param.Parameter.Name; return true;
            case IFieldReferenceOperation { Field.IsStatic: false } field: localName = field.Field.Name; return true;
            default: return false;
        }
    }

    private static LoweredAdAnnotationModel? TryLowerAdAnnotation(
        IInvocationOperation operation,
        LoweredShaderInstructionKind kind,
        SemanticModel semanticModel,
        CancellationToken cancellationToken)
    {
        if (kind is not (LoweredShaderInstructionKind.AdParameter or LoweredShaderInstructionKind.AdLoss)) return null;
        if (operation.Arguments.Length == 0) return null;
        var role = kind == LoweredShaderInstructionKind.AdParameter ? LoweredAdAnnotationRole.Parameter : LoweredAdAnnotationRole.Loss;
        var arg = operation.Arguments[0].Value;
        if (!ShaderSemanticFacts.TryUnwrapConversion(arg, out var unwrapped)) return null;
        if (role == LoweredAdAnnotationRole.Parameter &&
            operation.Arguments[0].Syntax is ArgumentSyntax { Expression: var argumentSyntax } &&
            ShaderModelFactory.TryTraceAdParameterSource(argumentSyntax, semanticModel, cancellationToken, out var source))
        {
            return new LoweredAdAnnotationModel(
                role,
                source.ResourceName,
                source.ResourceName,
                source.TypeName,
                source.IndexName,
                LoweredAdSourceKind.BufferElement);
        }

        if (ShaderResourceElementLowerer.TryLower(unwrapped, out var element))
        {
            return new LoweredAdAnnotationModel(
                role,
                element.ResourceName,
                element.ResourceName,
                GetTypeName(unwrapped.Type),
                element.IndexName,
                LoweredAdSourceKind.BufferElement);
        }

        var localName = unwrapped switch
        {
            ILocalReferenceOperation local => local.Local.Name,
            IParameterReferenceOperation param => param.Parameter.Name,
            IFieldReferenceOperation field => field.Field.Name,
            _ => null
        };
        return localName is not null
            ? new LoweredAdAnnotationModel(role, localName, string.Empty, GetTypeName(unwrapped.Type), string.Empty, LoweredAdSourceKind.Local)
            : null;
    }

    private static string GetTypeName(ITypeSymbol? type)
        => type?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) ?? string.Empty;
}
