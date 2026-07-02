using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Operations;

namespace Feather.Generators.Model;

internal sealed class ShaderIrLoweringException : Exception
{
    public ShaderIrLoweringException(string message, Location? location = null)
        : base(message)
    {
        Location = location;
    }

    public Location? Location { get; }
}

/// <summary>
/// Lowers a ShaderModel into a typed ShaderModuleModel using ordered IOperation traversal.
/// Preserves block nesting, statement order, scope, and semantic symbol identity.
/// </summary>
internal static class ShaderIrLowerer
{
    public static ShaderModuleModel? Lower(ShaderModel model, SemanticModel semanticModel, CancellationToken ct)
    {
        var entry = model.EntryPointSyntax;
        if (entry is null || (entry.Body is null && entry.ExpressionBody is null))
        {
            throw new ShaderIrLoweringException("entry point must have a block body or expression body", entry?.GetLocation() ?? model.Syntax.Identifier.GetLocation());
        }

        if (semanticModel.GetDeclaredSymbol(entry, ct) is not IMethodSymbol entrySymbol)
        {
            throw new ShaderIrLoweringException("entry point symbol could not be read from Roslyn semantics", entry.GetLocation());
        }

        var ctx = new LowerCtx(model, semanticModel, ct);
        foreach (var parameter in entrySymbol.Parameters)
        {
            ctx.Register(parameter, ToTy(parameter.Type, entry));
        }

        var body = LowerMethodBody(entry, semanticModel, ctx, ct);
        var kind = model.Kind switch
        {
            ShaderKind.Compute1D => ShaderFunctionKind.Compute1D,
            ShaderKind.Compute2D => ShaderFunctionKind.Compute2D,
            ShaderKind.Compute3D => ShaderFunctionKind.Compute3D,
            ShaderKind.Vertex => ShaderFunctionKind.Vertex,
            ShaderKind.Fragment => ShaderFunctionKind.Fragment,
            _ => ShaderFunctionKind.Compute1D
        };
        var entryReturnType = ShaderTypeFactory.FromTypeSymbol(entrySymbol.ReturnType) ?? ShaderTypeFactory.Void;
        var entryParameters = entrySymbol.Parameters.Select(p =>
            new ShaderParameterModel(
                p.Name,
                ToTy(p.Type, entry),
                p.RefKind is RefKind.Ref or RefKind.Out ? ShaderParameterDirection.InOut : ShaderParameterDirection.In))
            .ToArray();

        var entryFunc = new ShaderFunctionModel(
            model.Name, model.Name, kind,
            entryReturnType,
            new EquatableArray<ShaderParameterModel>(entryParameters),
            body);

        // Lower callable method bodies
        var callableFuncs = new List<ShaderFunctionModel>();
        foreach (var callable in model.Callables.Items)
        {
            ct.ThrowIfCancellationRequested();
            if (callable.Syntax.Body is null && callable.Syntax.ExpressionBody is null)
            {
                throw new ShaderIrLoweringException($"callable '{callable.Name}' must have a block body", callable.Syntax.GetLocation());
            }

            var callableSemanticModel = callable.Syntax.SyntaxTree == semanticModel.SyntaxTree
                ? semanticModel
                : semanticModel.Compilation.GetSemanticModel(callable.Syntax.SyntaxTree);
            var callCtx = new LowerCtx(model, callableSemanticModel, ct);
            foreach (var parameter in callable.Symbol.Parameters)
            {
                callCtx.Register(parameter, ToTy(parameter.Type, callable.Syntax));
            }

            var callBlock = LowerMethodBody(callable.Syntax, callableSemanticModel, callCtx, ct);
            var retType = ShaderTypeFactory.FromTypeSymbol(callable.Symbol.ReturnType) ?? ShaderTypeFactory.Void;
            var pars = callable.Parameters.Items.Select(p =>
                new ShaderParameterModel(p.Name,
                    ShaderTypeFactory.FromTypeName(p.TypeName) ?? ShaderTypeFactory.Float,
                    p.IsRef ? ShaderParameterDirection.InOut : ShaderParameterDirection.In))
                .ToArray();

            callableFuncs.Add(new ShaderFunctionModel(
                callable.Name, callable.MangledName, ShaderFunctionKind.Callable,
                retType, new EquatableArray<ShaderParameterModel>(pars), callBlock));
        }

        return new ShaderModuleModel(
            entryFunc, new EquatableArray<ShaderFunctionModel>(callableFuncs),
            model.Resources, new EquatableArray<ShaderStructType>(),
            model.ThreadGroup, model.Name, model.Namespace);
    }

    private static ShaderBlockStatement LowerMethodBody(MethodDeclarationSyntax method, SemanticModel semanticModel, LowerCtx ctx, CancellationToken ct)
    {
        var op = semanticModel.GetOperation(method, ct);
        IBlockOperation? block = op switch
        {
            IMethodBodyOperation { BlockBody: not null } mb => mb.BlockBody,
            IMethodBodyOperation { ExpressionBody: not null } mb => mb.ExpressionBody,
            IBlockOperation b => b,
            _ => null
        };

        if (block is null)
        {
            throw new ShaderIrLoweringException($"method '{method.Identifier.ValueText}' body could not be read from Roslyn operations", method.GetLocation());
        }

        return LowerBlock(block, ctx);
    }

    private static ShaderBlockStatement LowerBlock(IBlockOperation block, LowerCtx ctx)
    {
        var stmts = new List<ShaderStatement>();
        var prevScope = ctx.PushScope();
        foreach (var op in block.Operations)
        {
            ctx.CT.ThrowIfCancellationRequested();
            stmts.AddRange(LowerStmt(op, ctx));
        }
        ctx.PopScope(prevScope);
        return new ShaderBlockStatement(new EquatableArray<ShaderStatement>(stmts));
    }

    private static IReadOnlyList<ShaderStatement> LowerStmt(IOperation op, LowerCtx ctx)
    {
        return op switch
        {
            IBlockOperation b => [LowerBlock(b, ctx)],
            IVariableDeclarationGroupOperation g => LowerVarDecl(g, ctx),
            IExpressionStatementOperation { Operation: var operation } when ContainsAdMarkerInvocation(operation) => [],
            IExpressionStatementOperation e => [LowerExpressionStatement(e.Operation, ctx)],
            ISimpleAssignmentOperation a => [LowerSimpleAssign(a, ctx)],
            ICompoundAssignmentOperation c => [LowerCompoundAssign(c, ctx)],
            IIncrementOrDecrementOperation inc => [LowerIncDec(inc, ctx)],
            IConditionalOperation { IsRef: false } cond => [LowerIf(cond, ctx)],
            IForLoopOperation f => [LowerFor(f, ctx)],
            IWhileLoopOperation w => [LowerWhile(w, ctx)],
            IReturnOperation r => [new ShaderReturnStatement(r.ReturnedValue is { } v ? LowerExpr(v, ctx) : null)],
            IBranchOperation { BranchKind: BranchKind.Break } => [new ShaderBreakStatement()],
            IBranchOperation { BranchKind: BranchKind.Continue } => [new ShaderContinueStatement()],
            _ => throw Unsupported(op, $"unsupported statement operation '{op.Kind}'")
        };
    }

    private static IReadOnlyList<ShaderStatement> LowerVarDecl(IVariableDeclarationGroupOperation g, LowerCtx ctx)
    {
        var declarations = new List<ShaderStatement>();
        foreach (var d in g.Declarations)
        {
            foreach (var decl in d.Declarators)
            {
                var t = ToTy(decl.Symbol.Type, decl.Syntax);
                if (TryCreateSharedMemoryDeclaration(decl, t, ctx, out var shared))
                {
                    declarations.Add(shared);
                    ctx.Register(decl.Symbol, t);
                    if (t is ShaderArrayType { ElementType: var elementType })
                    {
                        ctx.RegisterShared(decl.Symbol.Name, elementType);
                    }
                    continue;
                }

                ShaderExpression? init = decl.Initializer?.Value is { } v ? LowerExpr(v, ctx) : null;
                declarations.Add(new ShaderLocalDeclarationStatement(decl.Symbol.Name, t, init, decl.Symbol));
                ctx.Register(decl.Symbol, t);
            }
        }

        return declarations;
    }

    private static ShaderStatement LowerExpressionStatement(IOperation operation, LowerCtx ctx)
    {
        if (ContainsAdMarkerInvocation(operation))
        {
            return new ShaderBlockStatement(new EquatableArray<ShaderStatement>());
        }

        return operation switch
        {
            ISimpleAssignmentOperation a => LowerSimpleAssign(a, ctx),
            ICompoundAssignmentOperation c => LowerCompoundAssign(c, ctx),
            IIncrementOrDecrementOperation inc => LowerIncDec(inc, ctx),
            IInvocationOperation inv when TryLowerBarrier(inv, out var barrier) => new ShaderBarrierStatement(barrier),
            _ => new ShaderExpressionStatement(LowerExpr(operation, ctx))
        };
    }

    private static ShaderStatement LowerSimpleAssign(ISimpleAssignmentOperation a, LowerCtx ctx)
    {
        if (a.Target is IDiscardOperation)
        {
            return new ShaderExpressionStatement(LowerExpr(a.Value, ctx));
        }

        var lv = LowerLVal(a.Target, ctx);
        if (lv is null)
        {
            throw Unsupported(a.Target, "assignment target is not a supported shader l-value");
        }

        return new ShaderAssignmentStatement(lv, LowerExpr(a.Value, ctx));
    }

    private static ShaderStatement LowerCompoundAssign(ICompoundAssignmentOperation a, LowerCtx ctx)
    {
        var lv = LowerLVal(a.Target, ctx);
        if (lv is null)
        {
            throw Unsupported(a.Target, "compound assignment target is not a supported shader l-value");
        }

        var op = MapBinaryOp(a.OperatorKind);
        if (op is null)
        {
            throw Unsupported(a, $"unsupported compound assignment operator '{a.OperatorKind}'");
        }

        return new ShaderCompoundAssignmentStatement(lv, op.Value, LowerExpr(a.Value, ctx));
    }

    private static ShaderStatement LowerIncDec(IIncrementOrDecrementOperation inc, LowerCtx ctx)
    {
        var lv = LowerLVal(inc.Target, ctx);
        if (lv is null)
        {
            throw Unsupported(inc.Target, "increment/decrement target is not a supported shader l-value");
        }

        return new ShaderIncrementDecrementStatement(lv,
            inc.Kind == OperationKind.Increment, !inc.IsPostfix);
    }

    private static ShaderStatement LowerIf(IConditionalOperation cond, LowerCtx ctx)
    {
        var c = LowerExpr(cond.Condition, ctx);
        var t = LowerOneOrBlock(cond.WhenTrue, ctx);
        var e = cond.WhenFalse is { } f ? LowerOneOrBlock(f, ctx) : null;
        return new ShaderIfStatement(c, t, e);
    }

    private static ShaderStatement LowerFor(IForLoopOperation f, LowerCtx ctx)
    {
        var prevScope = ctx.PushScope();
        var init = LowerOptionalStatementSequence(f.Before, ctx);
        ShaderExpression? cond = f.Condition is { } c ? LowerExpr(c, ctx) : null;
        var step = LowerOptionalStatementSequence(f.AtLoopBottom, ctx);
        var body = LowerOneOrBlock(f.Body, ctx);
        ctx.PopScope(prevScope);
        return new ShaderForStatement(init, cond, step, body);
    }

    private static ShaderStatement LowerWhile(IWhileLoopOperation w, LowerCtx ctx)
    {
        if (w.Condition is null)
        {
            throw Unsupported(w, "while and do-while loops must have a condition");
        }

        var cond = LowerExpr(w.Condition, ctx);
        var body = LowerOneOrBlock(w.Body, ctx);
        return w.ConditionIsTop
            ? new ShaderWhileStatement(cond, body)
            : new ShaderDoWhileStatement(body, cond);
    }

    private static ShaderBlockStatement LowerOneOrBlock(IOperation op, LowerCtx ctx)
        => op is IBlockOperation b ? LowerBlock(b, ctx)
            : new ShaderBlockStatement(new EquatableArray<ShaderStatement>(
                LowerStmt(op, ctx)));

    private static ShaderStatement? LowerOptionalStatementSequence(IEnumerable<IOperation> operations, LowerCtx ctx)
    {
        var statements = operations.SelectMany(operation => LowerStmt(operation, ctx)).ToArray();
        return statements.Length switch
        {
            0 => null,
            1 => statements[0],
            _ => new ShaderBlockStatement(new EquatableArray<ShaderStatement>(statements))
        };
    }

    // ── Expression lowering ──────────────────────────────────────────────

    private static ShaderExpression LowerExpr(IOperation op, LowerCtx ctx)
    {
        if (op is IConversionOperation conversion && IsSemanticConversion(conversion))
        {
            return new ShaderConversionExpression(ToTy(conversion.Type), LowerExpr(conversion.Operand, ctx));
        }

        var u = ShaderSemanticFacts.TryUnwrapConversion(op, out var uw) ? uw : op;
        if (TryLowerConstantExpression(u, ctx, out var constant))
        {
            return constant;
        }

        return u switch
        {
            ILiteralOperation lit => new ShaderLiteralExpression(ToTy(lit.Type), FormatLit(lit)),
            ILocalReferenceOperation loc when ctx.IsResource(loc.Local.Name)
                => new ShaderLocalReferenceExpression(ToTy(loc.Type), loc.Local.Name, loc.Local),
            ILocalReferenceOperation loc => new ShaderLocalReferenceExpression(ToTy(loc.Type), loc.Local.Name, loc.Local),
            IParameterReferenceOperation p when ctx.IsResource(p.Parameter.Name)
                => new ShaderParameterReferenceExpression(ToTy(p.Type), p.Parameter.Name, p.Parameter),
            IParameterReferenceOperation p => new ShaderParameterReferenceExpression(ToTy(p.Type), p.Parameter.Name, p.Parameter),
            IFieldReferenceOperation f => LowerFieldRef(f, ctx),
            IArrayElementReferenceOperation a => LowerArrayElem(a, ctx),
            IInlineArrayAccessOperation a => LowerInlineArrayElem(a, ctx),
            IBinaryOperation bin => LowerBinary(bin, ctx),
            IUnaryOperation un => LowerUnary(un, ctx),
            IInvocationOperation inv => LowerInvoke(inv, ctx),
            IPropertyReferenceOperation prop => LowerPropRef(prop, ctx),
            IObjectCreationOperation ctor => LowerCtor(ctor, ctx),
            IConditionalOperation { IsRef: false } tern => LowerTernary(tern, ctx),
            _ => throw Unsupported(u, $"unsupported expression operation '{u.Kind}'")
        };
    }

    private static ShaderExpression LowerFieldRef(IFieldReferenceOperation f, LowerCtx ctx)
    {
        if (TryLowerConstantExpression(f, ctx, out var constant))
            return constant;
        if (f.Field.IsStatic && ShaderSemanticFacts.IsBuiltinField(f.Field, out var bk))
            return new ShaderBuiltinExpression(ToTy(f.Type), bk);
        if (f.Instance is IInstanceReferenceOperation && ctx.IsResource(f.Field.Name))
            return new ShaderLocalReferenceExpression(ToTy(f.Type), f.Field.Name, f.Field);
        var inst = f.Instance is { } i ? LowerExpr(i, ctx)
            : new ShaderLocalReferenceExpression(ShaderTypeFactory.Void, "this", f.Field);
        return new ShaderFieldReferenceExpression(ToTy(f.Type), inst, f.Field);
    }

    private static bool TryLowerConstantExpression(IOperation operation, LowerCtx ctx, out ShaderExpression expression)
    {
        expression = null!;
        var type = ShaderTypeFactory.FromTypeSymbol(operation.Type);
        if (type is null)
        {
            return false;
        }

        if (operation.ConstantValue.HasValue && operation.ConstantValue.Value is { } constantValue)
        {
            expression = new ShaderLiteralExpression(type, FormatConstantValue(constantValue));
            return true;
        }

        if (operation is IFieldReferenceOperation { Field.IsStatic: true } fieldReference &&
            TryGetStaticFieldInitializerConstant(fieldReference.Field, ctx, out constantValue))
        {
            expression = new ShaderLiteralExpression(type, FormatConstantValue(constantValue));
            return true;
        }

        return false;
    }

    private static bool TryGetStaticFieldInitializerConstant(
        IFieldSymbol field,
        LowerCtx ctx,
        out object constantValue)
    {
        constantValue = null!;
        if (!field.IsReadOnly || field.DeclaringSyntaxReferences.Length == 0)
        {
            return false;
        }

        foreach (var syntaxReference in field.DeclaringSyntaxReferences)
        {
            if (syntaxReference.GetSyntax(ctx.CT) is not VariableDeclaratorSyntax { Initializer.Value: { } initializer })
            {
                continue;
            }

            var value = ctx.SemanticModel.GetConstantValue(initializer, ctx.CT);
            if (value.HasValue && value.Value is not null)
            {
                constantValue = value.Value;
                return true;
            }
        }

        return false;
    }

    private static ShaderExpression LowerArrayElem(IArrayElementReferenceOperation a, LowerCtx ctx)
    {
        if (a.Indices.Length != 1)
            throw Unsupported(a, "multi-dimensional array access is not supported");
        var arr = LowerExpr(a.ArrayReference, ctx);
        var idx = LowerExpr(a.Indices[0], ctx);
        if (arr is ShaderLocalReferenceExpression { Name: var n } && ctx.IsResource(n))
            return new ShaderResourceElementExpression(ToTy(a.Type), n, idx, null!);
        if (arr is ShaderFieldReferenceExpression { Instance: ShaderLocalReferenceExpression { Name: var fn } } && ctx.IsResource(fn))
            return new ShaderResourceElementExpression(ToTy(a.Type), fn, idx, null!);
        if (arr is ShaderLocalReferenceExpression { Name: var sharedName } && ctx.IsShared(sharedName))
            return new ShaderSharedMemoryElementExpression(ToTy(a.Type), sharedName, idx);
        return new ShaderIndexAccessExpression(ToTy(a.Type), arr, idx);
    }

    private static ShaderExpression LowerInlineArrayElem(IInlineArrayAccessOperation a, LowerCtx ctx)
    {
        var arr = LowerExpr(a.Instance, ctx);
        var idx = LowerExpr(a.Argument, ctx);
        return new ShaderIndexAccessExpression(ToTy(a.Type), arr, idx);
    }

    private static ShaderExpression LowerBinary(IBinaryOperation bin, LowerCtx ctx)
    {
        var l = LowerExpr(bin.LeftOperand, ctx);
        var r = LowerExpr(bin.RightOperand, ctx);
        var t = ToTy(bin.Type);
        if (IsCmp(bin.OperatorKind))
        {
            var op = MapCmpOp(bin.OperatorKind);
            if (op is null) throw Unsupported(bin, $"unsupported comparison operator '{bin.OperatorKind}'");
            return new ShaderComparisonExpression(t, op.Value, l, r);
        }
        if (IsLog(bin.OperatorKind))
        {
            var op = MapLogOp(bin.OperatorKind);
            if (op is null) throw Unsupported(bin, $"unsupported logical operator '{bin.OperatorKind}'");
            return new ShaderLogicalExpression(t, op.Value, l, r);
        }
        var bop = MapBinaryOp(bin.OperatorKind);
        if (bop is null) throw Unsupported(bin, $"unsupported binary operator '{bin.OperatorKind}'");
        return new ShaderBinaryExpression(t, bop.Value, l, r);
    }

    private static ShaderExpression LowerUnary(IUnaryOperation un, LowerCtx ctx)
    {
        var operand = LowerExpr(un.Operand, ctx);
        var t = ToTy(un.Type);
        var op = un.OperatorKind switch
        {
            UnaryOperatorKind.Minus => ShaderUnaryOperator.Negate,
            UnaryOperatorKind.Not => ShaderUnaryOperator.Not,
            UnaryOperatorKind.BitwiseNegation => ShaderUnaryOperator.BitwiseNot,
            _ => throw Unsupported(un, $"unsupported unary operator '{un.OperatorKind}'")
        };
        return new ShaderUnaryExpression(t, op, operand);
    }

    private static ShaderExpression LowerInvoke(IInvocationOperation inv, LowerCtx ctx)
    {
        var mn = ShaderSemanticFacts.GetMethodMetadataName(inv.TargetMethod);
        var ret = ToTy(inv.Type);
        if (TryLowerAtomic(inv, ret, ctx, out var atomic))
            return atomic;

        var args = LowerInvocationArguments(inv, ctx);
        if (TryLowerTextureSample(inv, ret, args, out var textureSample))
            return textureSample;
        if (ShaderInvocationLowerer.TryLowerKnownInvocation(inv, out _))
            return new ShaderIntrinsicCallExpression(ret, mn, new EquatableArray<ShaderExpression>(args));
        if (IsCallable(inv.TargetMethod))
            return new ShaderCallableCallExpression(ret, ShaderModelFactory.GetCallableMangledName(inv.TargetMethod), inv.TargetMethod,
                new EquatableArray<ShaderExpression>(args));
        throw Unsupported(inv, $"unsupported invocation '{mn}'");
    }

    private static bool TryLowerTextureSample(
        IInvocationOperation inv,
        ShaderType returnType,
        ShaderExpression[] args,
        out ShaderTextureSampleExpression expression)
    {
        expression = null!;
        if (!IsTextureSampleInvocation(inv))
        {
            return false;
        }

        var expectedArgumentCount = inv.TargetMethod.Name switch
        {
            "SampleLevel" => 4,
            "SampleGrad" => 5,
            _ => 3
        };
        if (args.Length != expectedArgumentCount)
        {
            throw Unsupported(inv, $"{inv.TargetMethod.Name} expects {expectedArgumentCount - 1} explicit arguments");
        }

        if (args[0].Type is not ShaderResourceWrapperType { Kind: ShaderResourceKind.Texture2D, Access: ShaderResourceAccess.Sample } ||
            args[1].Type is not ShaderResourceWrapperType { Kind: ShaderResourceKind.Sampler } ||
            args[2].Type != ShaderTypeFactory.Float2 ||
            (inv.TargetMethod.Name == "SampleLevel" && args[3].Type != ShaderTypeFactory.Float) ||
            (inv.TargetMethod.Name == "SampleGrad" && (args[3].Type != ShaderTypeFactory.Float2 || args[4].Type != ShaderTypeFactory.Float2)))
        {
            throw Unsupported(inv, "texture sampling requires SampledTexture2D<T>, SamplerState, float2 coordinates, optional float LOD, or float2 explicit gradients");
        }

        expression = new ShaderTextureSampleExpression(
            returnType,
            inv.TargetMethod.Name switch
            {
                "SampleLevel" => ShaderTextureSampleOperation.SampleLevel,
                "SampleGrad" => ShaderTextureSampleOperation.SampleGrad,
                _ => ShaderTextureSampleOperation.Sample
            },
            args[0],
            args[1],
            args[2],
            inv.TargetMethod.Name == "SampleLevel" ? args[3] : null,
            inv.TargetMethod.Name == "SampleGrad" ? args[3] : null,
            inv.TargetMethod.Name == "SampleGrad" ? args[4] : null);
        return true;
    }

    private static bool TryLowerAtomic(IInvocationOperation inv, ShaderType returnType, LowerCtx ctx, out ShaderAtomicExpression expression)
    {
        expression = null!;
        if (inv.TargetMethod.ContainingType?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) != "global::Feather.GpuAtomic")
        {
            return false;
        }

        var operation = inv.TargetMethod.Name switch
        {
            "Add" => ShaderAtomicOperation.Add,
            "Sub" => ShaderAtomicOperation.Sub,
            "Min" => ShaderAtomicOperation.Min,
            "Max" => ShaderAtomicOperation.Max,
            "And" => ShaderAtomicOperation.And,
            "Or" => ShaderAtomicOperation.Or,
            "Xor" => ShaderAtomicOperation.Xor,
            "Exchange" => ShaderAtomicOperation.Exchange,
            "CompareExchange" => ShaderAtomicOperation.CompareExchange,
            _ => (ShaderAtomicOperation?)null
        };
        if (operation is null)
        {
            return false;
        }

        var expectedArgumentCount = operation == ShaderAtomicOperation.CompareExchange ? 3 : 2;
        if (inv.Arguments.Length != expectedArgumentCount)
        {
            throw Unsupported(inv, $"atomic '{inv.TargetMethod.Name}' has an invalid argument count");
        }

        var target = LowerLVal(inv.Arguments[0].Value, ctx);
        if (target is null)
        {
            throw Unsupported(inv.Arguments[0].Value, $"atomic target for '{inv.TargetMethod.Name}' is not a supported shader l-value");
        }

        if (!IsAtomicStorageTarget(target))
        {
            throw Unsupported(inv.Arguments[0].Value,
                $"atomic target for '{inv.TargetMethod.Name}' must be rooted in a buffer or shared-memory element; {DescribeAtomicTarget(target)} is not addressable GPU storage");
        }

        if (!IsIntType(target.Type) || !IsIntType(returnType))
        {
            throw Unsupported(inv, "only int atomics are supported in the basic shader DSL");
        }

        var operands = inv.Arguments.Skip(1).Select(argument => LowerExpr(argument.Value, ctx)).ToArray();
        if (operands.Any(operand => !IsIntType(operand.Type)))
        {
            throw Unsupported(inv, "atomic operands must be int values");
        }

        expression = new ShaderAtomicExpression(returnType, operation.Value, target, new EquatableArray<ShaderExpression>(operands));
        return true;
    }

    private static bool IsAtomicStorageTarget(ShaderLValue target)
        => target switch
        {
            ShaderResourceElementLValue => true,
            ShaderSharedMemoryElementLValue => true,
            ShaderFieldLValue { Instance: { } instance } => IsAtomicStorageTarget(instance),
            ShaderMemberAccessLValue member => IsAtomicStorageTarget(member.Instance),
            ShaderIndexAccessLValue index => IsAtomicStorageTarget(index.Array),
            _ => false
        };

    private static string DescribeAtomicTarget(ShaderLValue target)
        => target switch
        {
            ShaderLocalLValue local => $"local '{local.Name}'",
            ShaderParameterLValue parameter => $"parameter '{parameter.Name}'",
            ShaderFieldLValue field => $"field '{field.Field.Name}'",
            ShaderMemberAccessLValue member => $"field '{member.Field.Name}'",
            ShaderIndexAccessLValue => "indexed local value",
            ShaderSwizzleLValue swizzle => $"swizzle '{swizzle.SwizzleComponents}'",
            ShaderMatrixColumnLValue => "matrix column",
            _ => "target"
        };

    private static ShaderExpression[] LowerInvocationArguments(IInvocationOperation inv, LowerCtx ctx)
    {
        var args = new List<ShaderExpression>();
        if (inv.Instance is not null && !IsCallable(inv.TargetMethod))
        {
            args.Add(LowerExpr(inv.Instance, ctx));
        }

        args.AddRange(inv.Arguments.Select(argument => LowerExpr(argument.Value, ctx)));
        return args.ToArray();
    }

    private static bool IsIntType(ShaderType type)
        => type is ShaderPrimitiveType { Kind: ShaderPrimitiveKind.Int };

    private static bool IsAdMarkerInvocation(IInvocationOperation inv)
        => inv.TargetMethod.ContainingType?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) == "global::Feather.AD.AD"
            && inv.TargetMethod.Name is "Parameter" or "Loss";

    private static bool ContainsAdMarkerInvocation(IOperation operation)
    {
        if (operation is IInvocationOperation invocation && IsAdMarkerInvocation(invocation))
        {
            return true;
        }

        foreach (var child in operation.ChildOperations)
        {
            if (ContainsAdMarkerInvocation(child))
            {
                return true;
            }
        }

        return false;
    }

    private static ShaderExpression LowerPropRef(IPropertyReferenceOperation prop, LowerCtx ctx)
    {
        if (prop.Arguments.Length > 0)
        {
            return LowerIndexerExpression(prop, ctx);
        }

        if (ShaderSemanticFacts.TryGetBuiltinKind(prop, out var lbk))
            return new ShaderBuiltinExpression(ToTy(prop.Type), (ShaderBuiltinKind)(byte)lbk);
        if (TryLowerBuiltinVector(prop, ctx, out var builtinVector))
            return builtinVector;
        if (prop.Property.Name == "Value" && prop.Instance is { } inst
            && ShaderSemanticFacts.IsUniformResourceType(inst.Type))
        {
            var rn = ResName(inst);
            return new ShaderPushConstantExpression(ToTy(prop.Type), rn, ctx.Binding(rn));
        }
        if (prop.Instance is { } matrixInstance && TryGetMatrixColumnIndex(ToTy(matrixInstance.Type), prop.Property.Name, out var column))
        {
            return new ShaderMatrixColumnExpression(
                ToTy(prop.Type),
                LowerExpr(matrixInstance, ctx),
                new ShaderLiteralExpression(ShaderTypeFactory.Int, column.ToString(System.Globalization.CultureInfo.InvariantCulture)));
        }
        if (prop.Instance is { } structInstance && TryGetStructField(ToTy(structInstance.Type), prop.Property.Name, out var field))
            return new ShaderMemberAccessExpression(field.Type, LowerExpr(structInstance, ctx), field);
        if (prop.Instance is { } i && IsSwiz(prop.Property.Name, ToTy(i.Type)))
            return new ShaderSwizzleExpression(ToTy(prop.Type), LowerExpr(i, ctx), prop.Property.Name.ToUpperInvariant());
        throw Unsupported(prop, $"unsupported property '{prop.Property.Name}'");
    }

    private static ShaderExpression LowerCtor(IObjectCreationOperation ctor, LowerCtx ctx)
    {
        var t = ToTy(ctor.Type);
        if (t is ShaderStructType structType && ctor.Initializer is { } initializer)
        {
            var values = new Dictionary<string, ShaderExpression>(StringComparer.Ordinal);
            foreach (var operation in initializer.Initializers)
            {
                if (operation is not ISimpleAssignmentOperation assignment)
                {
                    throw Unsupported(operation, "GPU struct object initializers support only simple member assignments");
                }

                var target = ShaderSemanticFacts.TryUnwrapConversion(assignment.Target, out var unwrappedTarget)
                    ? unwrappedTarget
                    : assignment.Target;
                var name = target switch
                {
                    IFieldReferenceOperation field => field.Field.Name,
                    IPropertyReferenceOperation property => property.Property.Name,
                    _ => throw Unsupported(target, "GPU struct object initializer target must be a field or property")
                };
                values[name] = LowerExpr(assignment.Value, ctx);
            }

            var initializerArgs = new List<ShaderExpression>(structType.Fields.Items.Count);
            foreach (var field in structType.Fields.Items)
            {
                if (!values.TryGetValue(field.Name, out var value))
                {
                    throw Unsupported(ctor, $"GPU struct object initializer is missing field '{field.Name}'");
                }

                initializerArgs.Add(value);
            }

            return new ShaderConstructorExpression(t, new EquatableArray<ShaderExpression>(initializerArgs));
        }

        var args = ctor.Arguments.Select(a => LowerExpr(a.Value, ctx)).ToArray();
        return new ShaderConstructorExpression(t, new EquatableArray<ShaderExpression>(args));
    }

    private static ShaderExpression LowerTernary(IConditionalOperation tern, LowerCtx ctx)
    {
        if (tern.WhenFalse is null)
        {
            throw Unsupported(tern, "conditional expressions must have a false branch");
        }

        return new ShaderConditionalExpression(ToTy(tern.Type),
            LowerExpr(tern.Condition, ctx), LowerExpr(tern.WhenTrue, ctx), LowerExpr(tern.WhenFalse, ctx));
    }

    // ── L-value lowering ─────────────────────────────────────────────────

    private static ShaderLValue? LowerLVal(IOperation target, LowerCtx ctx)
    {
        var u = ShaderSemanticFacts.TryUnwrapConversion(target, out var uw) ? uw : target;
        return u switch
        {
            ILocalReferenceOperation l => new ShaderLocalLValue(ToTy(l.Type), l.Local.Name, l.Local),
            IParameterReferenceOperation p => new ShaderParameterLValue(ToTy(p.Type), p.Parameter.Name, p.Parameter),
            IFieldReferenceOperation f => f.Instance is IInstanceReferenceOperation && ctx.IsResource(f.Field.Name)
                ? new ShaderLocalLValue(ToTy(f.Type), f.Field.Name, f.Field)
                : f.Instance is { } i ? new ShaderFieldLValue(ToTy(f.Type), LowerLVal(i, ctx), f.Field)
                : new ShaderFieldLValue(ToTy(f.Type), null, f.Field),
            IArrayElementReferenceOperation a => LowerArrayLVal(a, ctx),
            IInlineArrayAccessOperation a => LowerInlineArrayLVal(a, ctx),
            IPropertyReferenceOperation prop => LowerPropLVal(prop, ctx),
            _ => null
        };
    }

    private static ShaderLValue? LowerArrayLVal(IArrayElementReferenceOperation a, LowerCtx ctx)
    {
        if (a.Indices.Length != 1) return null;
        var idx = LowerExpr(a.Indices[0], ctx);
        if (a.ArrayReference is ILocalReferenceOperation lr && ctx.IsResource(lr.Local.Name))
            return new ShaderResourceElementLValue(ToTy(a.Type), lr.Local.Name, idx, null!);
        if (a.ArrayReference is ILocalReferenceOperation shared && ctx.IsShared(shared.Local.Name))
            return new ShaderSharedMemoryElementLValue(ToTy(a.Type), shared.Local.Name, idx);
        var arrLv = LowerLVal(a.ArrayReference, ctx);
        return arrLv is { } ? new ShaderIndexAccessLValue(ToTy(a.Type), arrLv, idx) : null;
    }

    private static ShaderLValue? LowerInlineArrayLVal(IInlineArrayAccessOperation a, LowerCtx ctx)
    {
        var idx = LowerExpr(a.Argument, ctx);
        var arrLv = LowerLVal(a.Instance, ctx);
        return arrLv is { } ? new ShaderIndexAccessLValue(ToTy(a.Type), arrLv, idx) : null;
    }

    private static ShaderLValue? LowerPropLVal(IPropertyReferenceOperation prop, LowerCtx ctx)
    {
        if (prop.Arguments.Length > 0)
        {
            return LowerIndexerLValue(prop, ctx);
        }

        if (prop.Instance is { } structInstance && TryGetStructField(ToTy(structInstance.Type), prop.Property.Name, out var field))
        {
            var instance = LowerLVal(structInstance, ctx);
            return instance is null ? null : new ShaderMemberAccessLValue(field.Type, instance, field);
        }

        if (prop.Instance is { } matrixInstance && TryGetMatrixColumnIndex(ToTy(matrixInstance.Type), prop.Property.Name, out var column))
        {
            return new ShaderMatrixColumnLValue(
                ToTy(prop.Type),
                LowerExpr(matrixInstance, ctx),
                new ShaderLiteralExpression(ShaderTypeFactory.Int, column.ToString(System.Globalization.CultureInfo.InvariantCulture)));
        }

        if (prop.Instance is { } i && IsSwiz(prop.Property.Name, ToTy(i.Type)))
            return new ShaderSwizzleLValue(ToTy(prop.Type), LowerExpr(i, ctx), prop.Property.Name.ToUpperInvariant());
        return null;
    }

    // ── helpers ──────────────────────────────────────────────────────────

    private static ShaderExpression LowerIndexerExpression(IPropertyReferenceOperation prop, LowerCtx ctx)
    {
        if (prop.Arguments.Length != 1)
        {
            throw Unsupported(prop, "only single-argument shader indexers are supported");
        }

        if (prop.Instance is null)
        {
            throw Unsupported(prop, "static indexers are not supported in shader code");
        }

        var instance = LowerExpr(prop.Instance, ctx);
        var index = LowerExpr(prop.Arguments[0].Value, ctx);
        if (TryResourceName(instance, ctx, out var resourceName))
        {
            return new ShaderResourceElementExpression(ToTy(prop.Type), resourceName, index, null!);
        }
        if (instance is ShaderLocalReferenceExpression { Name: var sharedName } && ctx.IsShared(sharedName))
        {
            return new ShaderSharedMemoryElementExpression(ToTy(prop.Type), sharedName, index);
        }

        return new ShaderIndexAccessExpression(ToTy(prop.Type), instance, index);
    }

    private static ShaderLValue? LowerIndexerLValue(IPropertyReferenceOperation prop, LowerCtx ctx)
    {
        if (prop.Arguments.Length != 1 || prop.Instance is null)
        {
            return null;
        }

        var instance = LowerExpr(prop.Instance, ctx);
        var index = LowerExpr(prop.Arguments[0].Value, ctx);
        if (TryResourceName(instance, ctx, out var resourceName))
        {
            return new ShaderResourceElementLValue(ToTy(prop.Type), resourceName, index, null!);
        }
        if (instance is ShaderLocalReferenceExpression { Name: var sharedName } && ctx.IsShared(sharedName))
        {
            return new ShaderSharedMemoryElementLValue(ToTy(prop.Type), sharedName, index);
        }

        var array = LowerLVal(prop.Instance, ctx);
        return array is null ? null : new ShaderIndexAccessLValue(ToTy(prop.Type), array, index);
    }

    private static bool TryCreateSharedMemoryDeclaration(
        IVariableDeclaratorOperation decl,
        ShaderType declaredType,
        LowerCtx ctx,
        out ShaderSharedMemoryDeclarationStatement statement)
    {
        statement = null!;
        if (decl.Initializer?.Value is not IObjectCreationOperation ctor ||
            declaredType is not ShaderArrayType { ElementType: var elementType } ||
            !IsSharedMemoryType(ctor.Type) ||
            ctor.Arguments.Length != 1)
        {
            return false;
        }

        if (ctor.Arguments[0].Value.ConstantValue.Value is not int length || length <= 0)
        {
            throw Unsupported(ctor.Arguments[0].Value, "shared memory length must be a positive compile-time int constant");
        }

        statement = new ShaderSharedMemoryDeclarationStatement(decl.Symbol.Name, elementType, length, decl.Symbol);
        return true;
    }

    private static bool IsSharedMemoryType(ITypeSymbol? type)
        => type is INamedTypeSymbol { IsGenericType: true } named &&
           named.ConstructedFrom.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) == "global::Feather.SharedMemory<T>";

    private static bool TryResourceName(ShaderExpression expression, LowerCtx ctx, out string name)
    {
        name = expression switch
        {
            ShaderLocalReferenceExpression local => local.Name,
            ShaderParameterReferenceExpression parameter => parameter.Name,
            ShaderFieldReferenceExpression { Instance: ShaderLocalReferenceExpression { Name: "this" }, Field: var field } => field.Name,
            _ => string.Empty
        };

        return name.Length > 0 && ctx.IsResource(name);
    }

    private static bool TryLowerBuiltinVector(IPropertyReferenceOperation prop, LowerCtx ctx, out ShaderExpression expression)
    {
        expression = null!;
        if (prop.Property.Name is not ("XY" or "XYZ"))
        {
            return false;
        }

        var containingType = prop.Property.ContainingType?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        var first = containingType switch
        {
            "global::Feather.ThreadIds" => ShaderBuiltinKind.ThreadIndexX,
            "global::Feather.LocalIds" => ShaderBuiltinKind.LocalIndexX,
            "global::Feather.GroupIds" => ShaderBuiltinKind.GroupIdX,
            "global::Feather.DispatchSize" => ShaderBuiltinKind.DispatchSizeX,
            "global::Feather.GroupSize" => ShaderBuiltinKind.GroupSizeX,
            "global::Feather.Graphics.VertexIds" => ShaderBuiltinKind.VertexIndex,
            "global::Feather.Graphics.FragmentIds" => ShaderBuiltinKind.FragmentCoordX,
            _ => default
        };

        if (first == default)
        {
            return false;
        }

        var count = prop.Property.Name.Length;
        var args = Enumerable.Range(0, count)
            .Select(offset => new ShaderBuiltinExpression(ShaderTypeFactory.Int, (ShaderBuiltinKind)((byte)first + offset)))
            .Cast<ShaderExpression>()
            .ToArray();
        expression = new ShaderConstructorExpression(ToTy(prop.Type), new EquatableArray<ShaderExpression>(args));
        return true;
    }

    private static bool TryLowerBarrier(IInvocationOperation inv, out ShaderBarrierKind kind)
    {
        kind = default;
        var containingType = inv.TargetMethod.ContainingType?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        if (containingType != "global::Feather.GpuBarrier")
        {
            return false;
        }

        kind = inv.TargetMethod.Name switch
        {
            "Workgroup" => ShaderBarrierKind.Workgroup,
            "Memory" => ShaderBarrierKind.Memory,
            "Full" => ShaderBarrierKind.Full,
            _ => default
        };
        return inv.TargetMethod.Name is "Workgroup" or "Memory" or "Full";
    }

    private static bool IsSemanticConversion(IConversionOperation conversion)
    {
        if (conversion.Type is null || conversion.Operand.Type is null)
        {
            return false;
        }

        if (conversion.Conversion.IsIdentity)
        {
            return false;
        }

        var sourceType = ShaderTypeFactory.FromTypeSymbol(conversion.Operand.Type);
        var targetType = ShaderTypeFactory.FromTypeSymbol(conversion.Type);
        return sourceType is not null && targetType is not null && sourceType != targetType;
    }

    private static bool IsTextureSampleInvocation(IInvocationOperation inv)
    {
        if (inv.TargetMethod.Name is not ("Sample" or "SampleLevel" or "SampleGrad"))
        {
            return false;
        }

        return ShaderTypeFactory.FromTypeSymbol(inv.TargetMethod.ContainingType) is
            ShaderResourceWrapperType { Kind: ShaderResourceKind.Texture2D, Access: ShaderResourceAccess.Sample };
    }

    private static bool TryGetStructField(ShaderType instanceType, string fieldName, out ShaderStructField field)
    {
        if (instanceType is ShaderStructType structType)
        {
            foreach (var candidate in structType.Fields.Items)
            {
                if (StringComparer.Ordinal.Equals(candidate.Name, fieldName))
                {
                    field = candidate;
                    return true;
                }
            }
        }

        field = null!;
        return false;
    }

    private static bool TryGetMatrixColumnIndex(ShaderType instanceType, string propertyName, out int column)
    {
        column = propertyName switch
        {
            "C0" => 0,
            "C1" => 1,
            "C2" => 2,
            "C3" => 3,
            _ => -1
        };

        return instanceType is ShaderMatrixType matrix && column >= 0 && column < matrix.Columns;
    }

    private static ShaderIrLoweringException Unsupported(IOperation operation, string message)
        => new(message, operation.Syntax.GetLocation());

    private static ShaderType ToTy(ITypeSymbol? t) => ToTy(t, null);

    private static ShaderType ToTy(ITypeSymbol? t, SyntaxNode? syntax)
        => ShaderTypeFactory.FromTypeSymbol(t)
            ?? throw new ShaderIrLoweringException(
                $"unsupported shader type '{t?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) ?? "<unknown>"}'",
                syntax?.GetLocation() ?? t?.Locations.FirstOrDefault());

    private static string FormatLit(ILiteralOperation lit)
    {
        if (!lit.ConstantValue.HasValue || lit.ConstantValue.Value is null) return "0";
        return FormatConstantValue(lit.ConstantValue.Value);
    }

    private static string FormatConstantValue(object v)
    {
        return v switch
        {
            float f => f.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
            double d => d.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
            int i => i.ToString(System.Globalization.CultureInfo.InvariantCulture),
            uint u => u.ToString(System.Globalization.CultureInfo.InvariantCulture),
            bool b => b ? "true" : "false",
            _ => v.ToString() ?? "0"
        };
    }

    private static string ResName(IOperation inst)
    {
        var u = ShaderSemanticFacts.TryUnwrapConversion(inst, out var uw) ? uw : inst;
        return u switch
        {
            ILocalReferenceOperation l => l.Local.Name,
            IParameterReferenceOperation p => p.Parameter.Name,
            IFieldReferenceOperation f => f.Field.Name,
            _ => "unknown"
        };
    }

    private static ShaderBinaryOperator? MapBinaryOp(BinaryOperatorKind k) => k switch
    {
        BinaryOperatorKind.Add => ShaderBinaryOperator.Add,
        BinaryOperatorKind.Subtract => ShaderBinaryOperator.Subtract,
        BinaryOperatorKind.Multiply => ShaderBinaryOperator.Multiply,
        BinaryOperatorKind.Divide => ShaderBinaryOperator.Divide,
        BinaryOperatorKind.Remainder => ShaderBinaryOperator.Modulo,
        BinaryOperatorKind.And => ShaderBinaryOperator.BitwiseAnd,
        BinaryOperatorKind.Or => ShaderBinaryOperator.BitwiseOr,
        BinaryOperatorKind.ExclusiveOr => ShaderBinaryOperator.BitwiseXor,
        BinaryOperatorKind.LeftShift => ShaderBinaryOperator.ShiftLeft,
        BinaryOperatorKind.RightShift => ShaderBinaryOperator.ShiftRight,
        _ => null
    };

    private static bool IsCmp(BinaryOperatorKind k) => k is BinaryOperatorKind.Equals or BinaryOperatorKind.NotEquals
        or BinaryOperatorKind.GreaterThan or BinaryOperatorKind.GreaterThanOrEqual
        or BinaryOperatorKind.LessThan or BinaryOperatorKind.LessThanOrEqual;

    private static ShaderCompareOperator? MapCmpOp(BinaryOperatorKind k) => k switch
    {
        BinaryOperatorKind.Equals => ShaderCompareOperator.Equal,
        BinaryOperatorKind.NotEquals => ShaderCompareOperator.NotEqual,
        BinaryOperatorKind.LessThan => ShaderCompareOperator.Less,
        BinaryOperatorKind.LessThanOrEqual => ShaderCompareOperator.LessEqual,
        BinaryOperatorKind.GreaterThan => ShaderCompareOperator.Greater,
        BinaryOperatorKind.GreaterThanOrEqual => ShaderCompareOperator.GreaterEqual,
        _ => null
    };

    private static bool IsLog(BinaryOperatorKind k) => k is BinaryOperatorKind.ConditionalAnd or BinaryOperatorKind.ConditionalOr;

    private static ShaderLogicalOperator? MapLogOp(BinaryOperatorKind k) => k switch
    {
        BinaryOperatorKind.ConditionalAnd => ShaderLogicalOperator.And,
        BinaryOperatorKind.ConditionalOr => ShaderLogicalOperator.Or,
        _ => null
    };

    private static bool IsCallable(IMethodSymbol m)
        => m.GetAttributes().Any(a => a.AttributeClass?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)
            is "global::Feather.CallableAttribute" or "global::Feather.ShaderFunctionAttribute");

    private static bool IsSwiz(string name, ShaderType? vt)
    {
        if (vt is not ShaderVectorType vec || name.Length < 1 || name.Length > 4) return false;
        var coordinateComponents = vec.ComponentCount switch
        {
            2 => "XYxy",
            3 => "XYZxyz",
            4 => "XYZWxyzw",
            _ => null
        };
        var colorComponents = vec.ComponentCount switch
        {
            2 => "RGrg",
            3 => "RGBrgb",
            4 => "RGBArgba",
            _ => null
        };
        var textureCoordinateComponents = vec.ComponentCount switch
        {
            2 => "STst",
            3 => "STPstp",
            4 => "STPQstpq",
            _ => null
        };

        return (coordinateComponents is not null && name.All(c => coordinateComponents.Contains(c)))
            || (colorComponents is not null && name.All(c => colorComponents.Contains(c)))
            || (textureCoordinateComponents is not null && name.All(c => textureCoordinateComponents.Contains(c)));
    }

    private sealed class LowerCtx
    {
        private readonly HashSet<string> _resources;
        private readonly Dictionary<string, ShaderType> _shared = new(StringComparer.Ordinal);
        private readonly Dictionary<string, uint> _bindings;
        private readonly Dictionary<string, ShaderType> _locals = new(StringComparer.Ordinal);
        public SemanticModel SemanticModel { get; }
        public CancellationToken CT { get; }

        public LowerCtx(ShaderModel model, SemanticModel semanticModel, CancellationToken ct)
        {
            SemanticModel = semanticModel;
            CT = ct;
            _resources = new HashSet<string>(model.Resources.Items.Select(r => r.Name), StringComparer.Ordinal);
            _bindings = model.Resources.Items.ToDictionary(r => r.Name, r => r.Binding, StringComparer.Ordinal);
        }

        public void Register(ISymbol s, ShaderType t) => _locals[s.Name] = t;
        public void RegisterShared(string name, ShaderType elementType) => _shared[name] = elementType;
        public bool IsResource(string n) => _resources.Contains(n);
        public bool IsShared(string n) => _shared.ContainsKey(n);
        public uint Binding(string n) => _bindings.TryGetValue(n, out var b) ? b : uint.MaxValue;
        public List<string> PushScope() => new(_locals.Keys.Concat(_shared.Keys.Select(static key => "$shared:" + key)));
        public void PopScope(List<string> prev)
        {
            var previousLocals = new HashSet<string>(
                prev.Where(static key => !key.StartsWith("$shared:", StringComparison.Ordinal)),
                StringComparer.Ordinal);
            var previousShared = new HashSet<string>(
                prev.Where(static key => key.StartsWith("$shared:", StringComparison.Ordinal))
                    .Select(static key => key.Substring("$shared:".Length)),
                StringComparer.Ordinal);
            foreach (var k in _locals.Keys.Except(previousLocals).ToList()) _locals.Remove(k);
            foreach (var k in _shared.Keys.Except(previousShared).ToList()) _shared.Remove(k);
        }
    }
}
