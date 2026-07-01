using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Operations;

namespace Feather.Generators.Model;

internal static class ShaderElementwiseExpressionLowerer
{
    public static bool TryLower(
        IOperation operation,
        ISymbol? expectedIndexSymbol,
        out LoweredElementwiseExpressionNodeModel expression,
        IReadOnlyDictionary<IParameterSymbol, IOperation>? paramSubst = null,
        HashSet<IMethodSymbol>? recursionGuard = null)
    {
        expression = default!;
        if (!ShaderSemanticFacts.TryUnwrapConversion(operation, out var unwrapped))
        {
            return false;
        }

        // Parameter substitution for callable inline expansion.
        if (paramSubst is not null
            && unwrapped is IParameterReferenceOperation paramRef
            && paramSubst.TryGetValue(paramRef.Parameter, out var subst))
        {
            return TryLower(subst, expectedIndexSymbol, out expression, paramSubst, recursionGuard);
        }

        if (ShaderResourceElementLowerer.TryLower(unwrapped, out var resource))
        {
            // null expectedIndexSymbol means "match any index" (used by control flow condition lowering)
            if (expectedIndexSymbol is not null
                && !SymbolEqualityComparer.Default.Equals(expectedIndexSymbol, resource.IndexSymbol))
            {
                return false;
            }

            expression = new LoweredElementwiseExpressionNodeModel(
                LoweredElementwiseExpressionNodeKind.Resource,
                LoweredElementwiseExpressionOperation.None,
                resource.ResourceName,
                resource.IndexName,
                string.Empty,
                string.Empty,
                ShaderSemanticFacts.GetTypeName(unwrapped.Type));
            return true;
        }

        if (ShaderSemanticFacts.TryFormatNumericLiteral(unwrapped, out var literal))
        {
            expression = new LoweredElementwiseExpressionNodeModel(
                LoweredElementwiseExpressionNodeKind.Literal,
                LoweredElementwiseExpressionOperation.None,
                string.Empty,
                string.Empty,
                literal,
                string.Empty,
                ShaderSemanticFacts.GetTypeName(unwrapped.Type));
            return true;
        }

        if (TryLowerPushConstant(unwrapped, out var pushConstant))
        {
            expression = pushConstant;
            return true;
        }

        if (unwrapped is IInvocationOperation invocation
            && TryLowerInvocation(invocation, expectedIndexSymbol, out expression, paramSubst, recursionGuard))
        {
            return true;
        }

        // Texture Sample / SampleLevel calls
        if (unwrapped is IInvocationOperation texCall
            && TryLowerTextureCall(texCall, expectedIndexSymbol, out expression, paramSubst, recursionGuard))
        {
            return true;
        }

        // Callable method calls — inline-expand the callable body.
        if (unwrapped is IInvocationOperation callInvocation
            && ShaderSemanticFacts.IsCallableMethod(callInvocation.TargetMethod))
        {
            return TryLowerCallableCall(callInvocation, expectedIndexSymbol, paramSubst, recursionGuard, out expression);
        }

        // Comparison binary: lowered BEFORE arithmetic binary since it uses different operation codes
        if (unwrapped is IBinaryOperation cmpBinary
            && IsComparisonOp(cmpBinary.OperatorKind)
            && TryLower(cmpBinary.LeftOperand, expectedIndexSymbol, out var cmpLeft, paramSubst, recursionGuard)
            && TryLower(cmpBinary.RightOperand, expectedIndexSymbol, out var cmpRight, paramSubst, recursionGuard))
        {
            expression = new LoweredElementwiseExpressionNodeModel(
                LoweredElementwiseExpressionNodeKind.Comparison,
                MapComparisonOp(cmpBinary.OperatorKind),
                string.Empty, string.Empty, string.Empty, string.Empty,
                ShaderSemanticFacts.GetTypeName(unwrapped.Type), cmpLeft, cmpRight);
            return true;
        }

        if (unwrapped is IBinaryOperation binary
            && TryLowerBinaryOperation(binary.OperatorKind, out var operationKind)
            && TryLower(binary.LeftOperand, expectedIndexSymbol, out var left, paramSubst, recursionGuard)
            && TryLower(binary.RightOperand, expectedIndexSymbol, out var right, paramSubst, recursionGuard))
        {
            expression = new LoweredElementwiseExpressionNodeModel(
                LoweredElementwiseExpressionNodeKind.Binary, operationKind,
                string.Empty, string.Empty, string.Empty, string.Empty,
                ShaderSemanticFacts.GetTypeName(unwrapped.Type), left, right);
            return true;
        }

        if (unwrapped is ILocalReferenceOperation localRef && !string.IsNullOrEmpty(localRef.Local.Name))
        {
            expression = new LoweredElementwiseExpressionNodeModel(
                LoweredElementwiseExpressionNodeKind.LocalVariable,
                LoweredElementwiseExpressionOperation.None,
                localRef.Local.Name, string.Empty, string.Empty, string.Empty,
                ShaderSemanticFacts.GetTypeName(unwrapped.Type));
            return true;
        }

        // GpuStruct field access: result.R, result.G, etc.
        if (unwrapped is IPropertyReferenceOperation gpuProp
            && ShaderSemanticFacts.TryGetGpuStructFieldIndex(gpuProp, out var fieldIndex, out var fieldTypeName))
        {
            if (gpuProp.Instance is not null
                && TryLower(gpuProp.Instance, expectedIndexSymbol, out var instanceExpr, paramSubst, recursionGuard))
            {
                expression = new LoweredElementwiseExpressionNodeModel(
                    LoweredElementwiseExpressionNodeKind.GpuStructField,
                    (LoweredElementwiseExpressionOperation)fieldIndex,
                    string.Empty, string.Empty, string.Empty, string.Empty,
                    fieldTypeName,
                    Arguments: new EquatableArray<LoweredElementwiseExpressionNodeModel>([instanceExpr]));
                return true;
            }
        }

        if (unwrapped is IFieldReferenceOperation gpuField
            && ShaderSemanticFacts.TryGetGpuStructFieldIndex(gpuField, out var gpuFieldIndex, out var gpuFieldTypeName))
        {
            if (gpuField.Instance is not null
                && TryLower(gpuField.Instance, expectedIndexSymbol, out var instanceExpr, paramSubst, recursionGuard))
            {
                expression = new LoweredElementwiseExpressionNodeModel(
                    LoweredElementwiseExpressionNodeKind.GpuStructField,
                    (LoweredElementwiseExpressionOperation)gpuFieldIndex,
                    string.Empty, string.Empty, string.Empty, string.Empty,
                    gpuFieldTypeName,
                    Arguments: new EquatableArray<LoweredElementwiseExpressionNodeModel>([instanceExpr]));
                return true;
            }
        }

        if (unwrapped is IPropertyReferenceOperation propRef
            && ShaderSemanticFacts.TryGetBuiltinKind(propRef, out var builtinKind))
        {
            expression = new LoweredElementwiseExpressionNodeModel(
                LoweredElementwiseExpressionNodeKind.ShaderBuiltin,
                LoweredElementwiseExpressionOperation.None,
                string.Empty, string.Empty, string.Empty, string.Empty,
                ShaderSemanticFacts.GetTypeName(unwrapped.Type),
                BuiltinKind: builtinKind);
            return true;
        }

        // Vector/matrix constructor: new float3(x, y, z)
        if (unwrapped is IObjectCreationOperation ctor
            && ShaderSemanticFacts.IsFeatherVectorType(ctor.Type)
            && ctor.Arguments.Length >= 1)
        {
            var args = new List<LoweredElementwiseExpressionNodeModel>(ctor.Arguments.Length);
            foreach (var arg in ctor.Arguments)
            {
                if (!TryLower(arg.Value, expectedIndexSymbol, out var loweredArg, paramSubst, recursionGuard)) return false;
                args.Add(loweredArg);
            }
            expression = new LoweredElementwiseExpressionNodeModel(
                LoweredElementwiseExpressionNodeKind.Constructor,
                LoweredElementwiseExpressionOperation.None,
                string.Empty, string.Empty, string.Empty, string.Empty,
                ShaderSemanticFacts.GetTypeName(unwrapped.Type),
                Arguments: new EquatableArray<LoweredElementwiseExpressionNodeModel>(args));
            return true;
        }

        // Ternary conditional (a ? b : c)
        if (unwrapped is IConditionalOperation ternary
            && ternary.WhenFalse is not null
            && TryLower(ternary.Condition, expectedIndexSymbol, out var cond, paramSubst, recursionGuard)
            && TryLower(ternary.WhenTrue, expectedIndexSymbol, out var whenTrue, paramSubst, recursionGuard)
            && TryLower(ternary.WhenFalse, expectedIndexSymbol, out var whenFalse, paramSubst, recursionGuard))
        {
            expression = new LoweredElementwiseExpressionNodeModel(
                LoweredElementwiseExpressionNodeKind.Ternary,
                LoweredElementwiseExpressionOperation.None,
                string.Empty, string.Empty, string.Empty, string.Empty,
                ShaderSemanticFacts.GetTypeName(unwrapped.Type),
                cond, whenTrue,
                new EquatableArray<LoweredElementwiseExpressionNodeModel>([whenFalse]));
            return true;
        }

        return false;
    }

    private static bool IsComparisonOp(BinaryOperatorKind k) => k is
        BinaryOperatorKind.Equals or BinaryOperatorKind.NotEquals
        or BinaryOperatorKind.GreaterThan or BinaryOperatorKind.GreaterThanOrEqual
        or BinaryOperatorKind.LessThan or BinaryOperatorKind.LessThanOrEqual;

    private static LoweredElementwiseExpressionOperation MapComparisonOp(BinaryOperatorKind k) => k switch
    {
        BinaryOperatorKind.Equals => LoweredElementwiseExpressionOperation.Equal,
        BinaryOperatorKind.NotEquals => LoweredElementwiseExpressionOperation.NotEqual,
        BinaryOperatorKind.GreaterThan => LoweredElementwiseExpressionOperation.Greater,
        BinaryOperatorKind.LessThan => LoweredElementwiseExpressionOperation.Less,
        BinaryOperatorKind.GreaterThanOrEqual => LoweredElementwiseExpressionOperation.GreaterEqual,
        BinaryOperatorKind.LessThanOrEqual => LoweredElementwiseExpressionOperation.LessEqual,
        _ => LoweredElementwiseExpressionOperation.None
    };

    private static bool TryLowerPushConstant(
        IOperation operation,
        out LoweredElementwiseExpressionNodeModel expression)
    {
        expression = default!;
        if (TryLowerUniformValue(operation, out var uniformResourceName))
        {
            expression = new LoweredElementwiseExpressionNodeModel(
                LoweredElementwiseExpressionNodeKind.PushConstant,
                LoweredElementwiseExpressionOperation.None,
                uniformResourceName,
                string.Empty,
                string.Empty,
                string.Empty,
                ShaderSemanticFacts.GetTypeName(operation.Type));
            return true;
        }

        if (TryGetPushConstantName(operation, out var resourceName))
        {
            expression = new LoweredElementwiseExpressionNodeModel(
                LoweredElementwiseExpressionNodeKind.PushConstant,
                LoweredElementwiseExpressionOperation.None,
                resourceName,
                string.Empty,
                string.Empty,
                string.Empty,
                ShaderSemanticFacts.GetTypeName(operation.Type));
            return true;
        }

        return false;
    }

    private static bool TryLowerUniformValue(IOperation operation, out string resourceName)
    {
        resourceName = string.Empty;
        if (operation is not IPropertyReferenceOperation { Property.Name: "Value" } property
            || !ShaderSemanticFacts.IsUniformResourceType(property.Instance?.Type))
        {
            return false;
        }

        return TryGetPushConstantName(property.Instance, out resourceName);
    }

    private static bool TryGetPushConstantName(IOperation? operation, out string resourceName)
    {
        resourceName = string.Empty;
        if (operation is null
            || !ShaderSemanticFacts.TryUnwrapConversion(operation, out var unwrapped)
            || !IsPushConstantResourceType(unwrapped.Type))
        {
            return false;
        }

        resourceName = unwrapped switch
        {
            IParameterReferenceOperation parameter => parameter.Parameter.Name,
            IFieldReferenceOperation field => field.Field.Name,
            ILocalReferenceOperation local => local.Local.Name,
            _ => string.Empty
        };
        return resourceName.Length > 0;
    }

    private static bool TryLowerInvocation(
        IInvocationOperation invocation,
        ISymbol? expectedIndexSymbol,
        out LoweredElementwiseExpressionNodeModel expression,
        IReadOnlyDictionary<IParameterSymbol, IOperation>? paramSubst = null,
        HashSet<IMethodSymbol>? recursionGuard = null)
    {
        expression = default!;
        if (!IsSupportedElementwiseMathInvocation(invocation.TargetMethod))
        {
            return false;
        }

        var arguments = new List<LoweredElementwiseExpressionNodeModel>(invocation.Arguments.Length);
        foreach (var argument in invocation.Arguments)
        {
            if (!TryLower(argument.Value, expectedIndexSymbol, out var loweredArgument, paramSubst, recursionGuard))
            {
                return false;
            }

            arguments.Add(loweredArgument);
        }

        expression = new LoweredElementwiseExpressionNodeModel(
            LoweredElementwiseExpressionNodeKind.Invocation,
            LoweredElementwiseExpressionOperation.None,
            string.Empty,
            string.Empty,
            string.Empty,
            ShaderSemanticFacts.GetMethodMetadataName(invocation.TargetMethod),
            ShaderSemanticFacts.GetTypeName(invocation.Type),
            Arguments: new EquatableArray<LoweredElementwiseExpressionNodeModel>(arguments));
        return true;
    }

    /// <summary>
    /// Lowers SampledTexture2D&lt;T&gt;.Sample and .SampleLevel calls to typed expression nodes.
    /// </summary>
    private static bool TryLowerTextureCall(
        IInvocationOperation invocation,
        ISymbol? expectedIndexSymbol,
        out LoweredElementwiseExpressionNodeModel expression,
        IReadOnlyDictionary<IParameterSymbol, IOperation>? paramSubst = null,
        HashSet<IMethodSymbol>? recursionGuard = null)
    {
        expression = default!;
        var method = invocation.TargetMethod;
        var containingType = method.ContainingType?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        if (containingType is null || !containingType.StartsWith("global::Feather.Resources.SampledTexture2D<"))
            return false;

        LoweredElementwiseExpressionNodeKind kind;
        int minArgs;
        if (method.Name == "Sample")
        {
            kind = LoweredElementwiseExpressionNodeKind.TextureSample;
            minArgs = 2; // (SamplerState sampler, float2 uv)
        }
        else if (method.Name == "SampleLevel")
        {
            kind = LoweredElementwiseExpressionNodeKind.TextureSampleLevel;
            minArgs = 3; // (SamplerState sampler, float2 uv, float lod)
        }
        else
        {
            return false;
        }

        if (invocation.Arguments.Length < minArgs)
            return false;

        // Extract texture resource name from the instance (e.g. "tex" in tex.Sample(...))
        string? textureName = null;
        if (invocation.Instance is not null)
            textureName = GetResourceNameFromOperation(invocation.Instance);

        if (textureName is not { Length: > 0 })
            return false;

        // Sampler resource name from first argument
        string? samplerName = GetResourceNameFromOperation(invocation.Arguments[0].Value);

        // Lower remaining arguments (uv, optional lod)
        var args = new List<LoweredElementwiseExpressionNodeModel>();
        for (int i = 1; i < invocation.Arguments.Length; i++)
        {
            if (!TryLower(invocation.Arguments[i].Value, expectedIndexSymbol, out var lowered, paramSubst, recursionGuard))
                return false;
            args.Add(lowered);
        }

        expression = new LoweredElementwiseExpressionNodeModel(
            kind,
            LoweredElementwiseExpressionOperation.None,
            textureName,
            string.Empty,
            string.Empty,
            samplerName ?? string.Empty,
            ShaderSemanticFacts.GetTypeName(invocation.Type),
            Arguments: new EquatableArray<LoweredElementwiseExpressionNodeModel>(args));
        return true;
    }

    /// <summary>Extracts the resource name from a parameter/field/local reference.</summary>
    private static string? GetResourceNameFromOperation(IOperation? operation)
    {
        if (operation is null || !ShaderSemanticFacts.TryUnwrapConversion(operation, out var unwrapped))
            return null;
        return unwrapped switch
        {
            IParameterReferenceOperation p => p.Parameter.Name,
            IFieldReferenceOperation f => f.Field.Name,
            ILocalReferenceOperation l => l.Local.Name,
            _ => null
        };
    }

    /// <summary>
    /// Inline-expands a [Callable] method by lowering its return expression with
    /// parameter references replaced by the actual argument expressions.
    /// Rejects recursive or control-flow-heavy callables that cannot be lowered to a single expression.
    /// </summary>
    private static bool TryLowerCallableCall(
        IInvocationOperation invocation,
        ISymbol? expectedIndexSymbol,
        IReadOnlyDictionary<IParameterSymbol, IOperation>? parentParamSubst,
        HashSet<IMethodSymbol>? parentRecursionGuard,
        out LoweredElementwiseExpressionNodeModel expression)
    {
        expression = default!;
        var method = invocation.TargetMethod;

        // ── Recursion guard: reject direct or mutual recursion ──────────────
        var recursionGuard = parentRecursionGuard ?? new HashSet<IMethodSymbol>(SymbolEqualityComparer.Default);
        if (!recursionGuard.Add(method))
            return false;

        try
        {
            // Build the parameter → argument mapping.
            var substitutions = new Dictionary<IParameterSymbol, IOperation>(SymbolEqualityComparer.Default);
            for (int i = 0; i < invocation.Arguments.Length; i++)
            {
                if (invocation.Arguments[i].Parameter is { } p)
                    substitutions[p] = invocation.Arguments[i].Value;
            }

            // Merge parent substitutions (for nested callable calls).
            if (parentParamSubst is not null)
            {
                foreach (var kv in parentParamSubst)
                {
                    if (!substitutions.ContainsKey(kv.Key))
                        substitutions.Add(kv.Key, kv.Value);
                }
            }

            // Get the callable's method body.
            var syntaxRef = method.DeclaringSyntaxReferences.FirstOrDefault();
            if (syntaxRef?.GetSyntax() is not MethodDeclarationSyntax methodSyntax)
                return false;

            // Extract the return expression (handles both block-body and expression-body).
            ExpressionSyntax? returnExprSyntax = null;

            if (methodSyntax.ExpressionBody is { } arrowBody)
            {
                // Expression-bodied: float Scale(float x) => x * 3.0f;
                returnExprSyntax = arrowBody.Expression;
            }
            else if (methodSyntax.Body is { } blockBody)
            {
                // Block-bodied: find the single return statement.
                var returnStatements = blockBody.DescendantNodes()
                    .OfType<ReturnStatementSyntax>()
                    .ToArray();
                if (returnStatements.Length != 1 || returnStatements[0].Expression is null)
                    return false;
                returnExprSyntax = returnStatements[0].Expression;
            }
            else
            {
                return false;
            }

            if (returnExprSyntax is null)
                return false;

            // Get the semantic model for the callable's syntax tree.
            var compilation = invocation.SemanticModel?.Compilation;
            if (compilation is null)
                return false;

            var callableSemanticModel = compilation.GetSemanticModel(methodSyntax.SyntaxTree);
            var returnOp = callableSemanticModel.GetOperation(returnExprSyntax);
            if (returnOp is null)
                return false;

            // Lower the return expression with parameter substitutions applied.
            return TryLower(returnOp, expectedIndexSymbol, out expression, substitutions, recursionGuard);
        }
        finally
        {
            recursionGuard.Remove(method);
        }
    }

    private static bool TryLowerBinaryOperation(
        BinaryOperatorKind operatorKind,
        out LoweredElementwiseExpressionOperation operation)
    {
        operation = operatorKind switch
        {
            BinaryOperatorKind.Add => LoweredElementwiseExpressionOperation.Add,
            BinaryOperatorKind.Subtract => LoweredElementwiseExpressionOperation.Subtract,
            BinaryOperatorKind.Multiply => LoweredElementwiseExpressionOperation.Multiply,
            BinaryOperatorKind.Divide => LoweredElementwiseExpressionOperation.Divide,
            _ => LoweredElementwiseExpressionOperation.None
        };
        return operation != LoweredElementwiseExpressionOperation.None;
    }

    private static bool IsSupportedElementwiseMathInvocation(IMethodSymbol method)
    {
        var containingType = method.ContainingType?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        if (containingType is not ("global::Feather.Math.ShaderMath" or "global::Feather.Math.Hlsl"))
        {
            return false;
        }

        return method.Name switch
        {
            "Sin" or "Cos" or "Tan" or "Exp" or "Log" or "Sqrt" or "InverseSqrt"
                => ShaderSemanticFacts.HasFloatSignature(method, 1),
            "Abs" or "Floor" or "Ceil" or "Round" or "Fract" or "Saturate"
                => ShaderSemanticFacts.HasFloatOrMatchingFloatVectorUnarySignature(method),
            "Length" => method.Parameters.Length == 1 && method.ReturnType.SpecialType == SpecialType.System_Single
                && ShaderSemanticFacts.IsFeatherVectorType(method.Parameters[0].Type),
            "Normalize" => method.Parameters.Length == 1
                && ShaderSemanticFacts.IsFeatherVectorType(method.ReturnType)
                && ShaderSemanticFacts.IsFeatherVectorType(method.Parameters[0].Type),
            "Pow" => ShaderSemanticFacts.HasFloatSignature(method, 2),
            "Min" or "Max" => ShaderSemanticFacts.HasFloatOrMatchingFloatVectorBinarySignature(method),
            "Clamp" => ShaderSemanticFacts.HasFloatOrMatchingFloatVectorClampSignature(method),
            "Lerp" or "Mix" => ShaderSemanticFacts.HasFloatOrMatchingFloatVectorLerpSignature(method),
            "Smoothstep" => ShaderSemanticFacts.HasFloatSignature(method, 3),
            "Dot" => ShaderSemanticFacts.HasFloatVectorDotSignature(method),
            "Cross" => ShaderSemanticFacts.HasFloat3CrossSignature(method),
            "Mul" => ShaderSemanticFacts.HasMatrixMulSignature(method),
            "Transpose" or "Inverse" => ShaderSemanticFacts.HasMatrixTransformSignature(method),
            "Determinant" => ShaderSemanticFacts.HasMatrixScalarSignature(method),
            "Hadamard" => ShaderSemanticFacts.HasMatrixHadamardSignature(method),
            _ => false
        };
    }

    private static bool IsPushConstantResourceType(ITypeSymbol? type)
        => type is not null
            && !ShaderSemanticFacts.IsShaderResourceType(type)
            && (ShaderSemanticFacts.IsUniformResourceType(type) || ShaderSemanticFacts.IsSupportedScalarPushConstantType(type));
}
