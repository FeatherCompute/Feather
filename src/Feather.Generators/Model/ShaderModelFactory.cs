using Feather.Generators.Diagnostics;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Operations;

namespace Feather.Generators.Model;

internal static class ShaderModelFactory
{
    public static ShaderModel? Create(GeneratorAttributeSyntaxContext context, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (context.TargetNode is not StructDeclarationSyntax syntax || context.TargetSymbol is not INamedTypeSymbol symbol)
        {
            return null;
        }

        var model = CreateCore(syntax, symbol, context.SemanticModel, cancellationToken);
        if (model is null) return null;
        var lowered = ShaderSemanticLowerer.Lower(model, context.SemanticModel, cancellationToken);
        var typedIrDiagnostics = new List<TypedIrDiagnosticModel>();
        var typedIr = TryLowerTypedIr(model, context.SemanticModel, cancellationToken, typedIrDiagnostics);
        var bodyDiagnostics = CollectShaderBodyDiagnostics(model, context.SemanticModel, cancellationToken)
            .ToArray();
        return model with
        {
            LoweredInstructions = lowered,
            TypedIrSection = typedIr,
            TypedIrDiagnostics = new EquatableArray<TypedIrDiagnosticModel>(typedIrDiagnostics),
            BodyDiagnostics = new EquatableArray<ShaderBodyDiagnosticModel>(bodyDiagnostics)
        };
    }

    private static byte[]? TryLowerTypedIr(
        ShaderModel model,
        SemanticModel semanticModel,
        CancellationToken cancellationToken,
        List<TypedIrDiagnosticModel> diagnostics)
    {
        try
        {
            if (model.EntryPointSyntax is null || model.EntryPointSymbol is null)
            {
                return null;
            }

            var typedModule = ShaderIrLowerer.Lower(model, semanticModel, cancellationToken);
            if (typedModule is null)
            {
                return null;
            }

            return ShaderIrModuleWriter.WriteModule(typedModule);
        }
        catch (ShaderIrLoweringException ex)
        {
            if (model.Kind is ShaderKind.Compute1D or ShaderKind.Compute2D or ShaderKind.Compute3D)
            {
                diagnostics.Add(new TypedIrDiagnosticModel(
                    ex.Location ?? model.Syntax.Identifier.GetLocation(),
                    ex.Message));
            }
            return null;
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            if (model.Kind is ShaderKind.Compute1D or ShaderKind.Compute2D or ShaderKind.Compute3D)
            {
                diagnostics.Add(new TypedIrDiagnosticModel(model.Syntax.Identifier.GetLocation(), ex.Message));
            }
            return null;
        }
    }

    public static ShaderModel? Create(StructDeclarationSyntax syntax, INamedTypeSymbol symbol)
    {
        return CreateCore(syntax, symbol, null, CancellationToken.None);
    }

    public static ShaderModel? Create(StructDeclarationSyntax syntax, INamedTypeSymbol symbol, SemanticModel semanticModel, CancellationToken cancellationToken)
    {
        var model = CreateCore(syntax, symbol, semanticModel, cancellationToken);
        if (model is null)
        {
            return null;
        }

        var bodyDiagnostics = CollectShaderBodyDiagnostics(model, semanticModel, cancellationToken)
            .ToArray();
        return model with
        {
            BodyDiagnostics = new EquatableArray<ShaderBodyDiagnosticModel>(bodyDiagnostics)
        };
    }

    private static ShaderModel? CreateCore(
        StructDeclarationSyntax syntax,
        INamedTypeSymbol symbol,
        SemanticModel? semanticModel,
        CancellationToken cancellationToken)
    {
        var kind = GetShaderKind(symbol);
        if (kind is null)
        {
            return null;
        }

        var namespaceName = symbol.ContainingNamespace.IsGlobalNamespace ? string.Empty : symbol.ContainingNamespace.ToDisplayString();
        var resources = GetPrimaryConstructorResources(syntax, symbol);
        var threadGroup = ReadThreadGroup(symbol);
        var entryPoint = ResolveEntryPoint(syntax, symbol);
        var callables = semanticModel is null
            ? DiscoverDeclaredCallables(syntax, symbol, entryPoint.Symbol)
            : DiscoverCallables(syntax, symbol, entryPoint.Syntax, entryPoint.Symbol, semanticModel, cancellationToken);
        return new ShaderModel(
            syntax,
            symbol,
            kind.Value,
            namespaceName,
            symbol.Name,
            symbol.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat),
            entryPoint.Syntax,
            entryPoint.Symbol,
            threadGroup,
            BoundsCheck: !HasKernelBoundsCheckFalse(symbol),
            AutoDiff: HasAttribute(symbol, "Feather.AutoDiffAttribute"),
            new EquatableArray<ResourceModel>(resources),
            new EquatableArray<LoweredShaderInstructionModel>(Array.Empty<LoweredShaderInstructionModel>()),
            Callables: callables);
    }

    public static IEnumerable<Diagnostic> Validate(ShaderModel model)
    {
        foreach (var diagnostic in model.BodyDiagnostics.Items)
        {
            yield return Diagnostic.Create(
                diagnostic.Descriptor,
                diagnostic.Location ?? model.Syntax.Identifier.GetLocation(),
                diagnostic.Arguments.Items.Cast<object>().ToArray());
        }

        foreach (var diagnostic in model.TypedIrDiagnostics.Items)
        {
            yield return Diagnostic.Create(
                FeatherDiagnostics.TypedIrLoweringFailed,
                diagnostic.Location ?? model.Syntax.Identifier.GetLocation(),
                model.Name,
                diagnostic.Message);
        }

        var modifiers = model.Syntax.Modifiers;
        var isPartial = modifiers.Any(SyntaxKind.PartialKeyword);
        var isReadonly = modifiers.Any(SyntaxKind.ReadOnlyKeyword);
        if (!isPartial || !isReadonly)
        {
            yield return Diagnostic.Create(FeatherDiagnostics.ShaderTypeShape, model.Syntax.Identifier.GetLocation(), model.Name);
        }

        if (!IsCompatibleEntryPoint(model))
        {
            yield return Diagnostic.Create(FeatherDiagnostics.ComputeExecute, model.Syntax.Identifier.GetLocation(), model.Name);
        }

        foreach (var resource in model.Resources.Items)
        {
            if (resource.Kind == ResourceKindModel.PushConstant && !resource.IsPushConstantSupported)
            {
                yield return Diagnostic.Create(
                    FeatherDiagnostics.UnsupportedConstructorParameter,
                    model.Syntax.ParameterList?.Parameters.FirstOrDefault(p => p.Identifier.ValueText == resource.Name)?.GetLocation() ?? model.Syntax.Identifier.GetLocation(),
                    resource.Name,
                    resource.TypeName);
            }

            if (resource.Kind is ResourceKindModel.Buffer or ResourceKindModel.Texture2D or ResourceKindModel.Texture3D &&
                !resource.IsComputeStorageElementSupported)
            {
                yield return Diagnostic.Create(
                    FeatherDiagnostics.UnsupportedConstructorParameter,
                    model.Syntax.ParameterList?.Parameters.FirstOrDefault(p => p.Identifier.ValueText == resource.Name)?.GetLocation() ?? model.Syntax.Identifier.GetLocation(),
                    resource.Name,
                    resource.ElementTypeName);
            }
        }

        foreach (var field in model.Symbol.GetMembers().OfType<IFieldSymbol>().Where(static field => !field.IsStatic && !field.IsImplicitlyDeclared))
        {
            if (!IsSupportedCapturedField(field.Type))
            {
                yield return Diagnostic.Create(
                    FeatherDiagnostics.UnsupportedCapturedField,
                    field.Locations.FirstOrDefault() ?? model.Syntax.Identifier.GetLocation(),
                    field.Name,
                    field.Type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat));
            }
        }

        if (model.ThreadGroup.X <= 0 || model.ThreadGroup.Y <= 0 || model.ThreadGroup.Z <= 0)
        {
            yield return Diagnostic.Create(FeatherDiagnostics.ThreadGroupSizeInvalid, model.Syntax.Identifier.GetLocation(), model.Name);
        }

        if (model.Kind is ShaderKind.Vertex or ShaderKind.Fragment)
        {
            var varyingType = GetGraphicsVaryingType(model);
            if (varyingType is not null && !IsSupportedGraphicsVarying(varyingType))
            {
                yield return Diagnostic.Create(
                    FeatherDiagnostics.GraphicsVaryingUnsupported,
                    model.Syntax.Identifier.GetLocation(),
                    varyingType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat));
            }
        }

        if (model.Kind == ShaderKind.Fragment)
        {
            var outputType = GetFragmentOutputType(model);
            if (outputType is not null && !IsSupportedFragmentOutput(outputType, out _))
            {
                yield return Diagnostic.Create(
                    FeatherDiagnostics.FragmentOutputUnsupported,
                    model.Syntax.Identifier.GetLocation(),
                    outputType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat));
            }
        }
    }

    private static IEnumerable<Diagnostic> ValidateShaderBody(ShaderModel model, SemanticModel semanticModel, CancellationToken cancellationToken)
    {
        foreach (var diagnostic in CollectShaderBodyDiagnostics(model, semanticModel, cancellationToken))
        {
            yield return Diagnostic.Create(
                diagnostic.Descriptor,
                diagnostic.Location ?? model.Syntax.Identifier.GetLocation(),
                diagnostic.Arguments.Items.Cast<object>().ToArray());
        }
    }

    private static IEnumerable<ShaderBodyDiagnosticModel> CollectShaderBodyDiagnostics(
        ShaderModel model,
        SemanticModel semanticModel,
        CancellationToken cancellationToken)
    {
        foreach (var diagnostic in GetCallableRecursionDiagnostics(model, semanticModel, cancellationToken))
        {
            yield return diagnostic;
        }

        foreach (var diagnostic in ValidateAutoDiffMarkers(model, semanticModel, cancellationToken))
        {
            yield return diagnostic;
        }

        foreach (var method in GetShaderMethods(model))
        {
            var methodSemanticModel = GetSemanticModelForMethod(semanticModel, method);
            var symbol = methodSemanticModel.GetDeclaredSymbol(method, cancellationToken);
            if (symbol is not null && IsCallable(symbol))
            {
                foreach (var diagnostic in ValidateCallableDeclaration(method, symbol))
                {
                    yield return diagnostic;
                }

            }

            foreach (var diagnostic in ValidateShaderMethod(method, methodSemanticModel, cancellationToken))
            {
                yield return diagnostic;
            }
        }
    }

    private static IEnumerable<ShaderBodyDiagnosticModel> ValidateAutoDiffMarkers(
        ShaderModel model,
        SemanticModel semanticModel,
        CancellationToken cancellationToken)
    {
        var entry = model.EntryPointSyntax;
        if (entry is null)
        {
            yield break;
        }

        var lossCount = 0;
        var root = (SyntaxNode?)entry.Body ?? entry.ExpressionBody?.Expression;
        if (root is null)
        {
            yield break;
        }

        foreach (var invocation in root.DescendantNodes().OfType<InvocationExpressionSyntax>())
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!TryGetAutoDiffMarkerName(invocation, semanticModel, cancellationToken, out var markerName))
            {
                continue;
            }

            if (model.Kind is not ShaderKind.Compute1D)
            {
                yield return BodyDiagnostic(
                    FeatherDiagnostics.AutoDiffMarkerUnsupported,
                    invocation.GetLocation(),
                    markerName,
                    "the MVP supports markers only in one-dimensional compute kernels");
                continue;
            }

            if (!model.AutoDiff)
            {
                yield return BodyDiagnostic(
                    FeatherDiagnostics.AutoDiffMarkerUnsupported,
                    invocation.GetLocation(),
                    markerName,
                    "the containing kernel must be annotated with [AutoDiff]");
                continue;
            }

            if (invocation.ArgumentList.Arguments.Count != 1)
            {
                yield return BodyDiagnostic(
                    FeatherDiagnostics.AutoDiffMarkerUnsupported,
                    invocation.GetLocation(),
                    markerName,
                    "marker calls must have exactly one argument");
                continue;
            }

            var argument = invocation.ArgumentList.Arguments[0].Expression;
            var argumentType = semanticModel.GetTypeInfo(argument, cancellationToken).Type;
            if (markerName == "Loss")
            {
                lossCount++;
                if (lossCount > 1)
                {
                    yield return BodyDiagnostic(
                        FeatherDiagnostics.AutoDiffMarkerUnsupported,
                        invocation.GetLocation(),
                        markerName,
                        "only one scalar loss marker is supported");
                }

                if (argumentType?.SpecialType != SpecialType.System_Single)
                {
                    yield return BodyDiagnostic(
                        FeatherDiagnostics.AutoDiffSourceUnsupported,
                        argument.GetLocation(),
                        markerName,
                        "loss must be a scalar float");
                }

                continue;
            }

            if (!IsSupportedAutoDiffValueType(argumentType))
            {
                yield return BodyDiagnostic(
                    FeatherDiagnostics.AutoDiffSourceUnsupported,
                    argument.GetLocation(),
                    markerName,
                    $"parameter type '{argumentType?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) ?? "<unknown>"}' is not supported");
                continue;
            }

            if (!IsBufferElementArgument(argument, semanticModel, cancellationToken, out var sourceType))
            {
                yield return BodyDiagnostic(
                    FeatherDiagnostics.AutoDiffSourceUnsupported,
                    argument.GetLocation(),
                    markerName,
                    "parameter values must be directly traceable to a captured buffer element");
                continue;
            }

            if (sourceType is not null && GetShaderResourceViewKind(sourceType) != ShaderResourceViewKind.Buffer)
            {
                yield return BodyDiagnostic(
                    FeatherDiagnostics.AutoDiffSourceUnsupported,
                    argument.GetLocation(),
                    markerName,
                    "texture, sampler, and non-buffer parameter sources are not supported");
            }
        }
    }

    private static bool TryGetAutoDiffMarkerName(
        InvocationExpressionSyntax invocation,
        SemanticModel semanticModel,
        CancellationToken cancellationToken,
        out string markerName)
    {
        var symbolInfo = semanticModel.GetSymbolInfo(invocation, cancellationToken);
        if (symbolInfo.Symbol is IMethodSymbol method && IsAutoDiffMarkerMethod(method))
        {
            markerName = method.Name;
            return true;
        }

        foreach (var candidate in symbolInfo.CandidateSymbols.OfType<IMethodSymbol>())
        {
            if (IsAutoDiffMarkerMethod(candidate))
            {
                markerName = candidate.Name;
                return true;
            }
        }

        markerName = string.Empty;
        return false;
    }

    private static bool IsAutoDiffMarkerMethod(IMethodSymbol method)
        => method.Name is "Parameter" or "Loss" && IsAutoDiffMarkerType(method.ContainingType);

    private static bool IsAutoDiffMarkerType(ITypeSymbol? type)
        => type?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) == "global::Feather.AD.AD";

    internal readonly record struct TraceableAdSource(
        string ResourceName,
        ITypeSymbol? ResourceType,
        string TypeName,
        string IndexName);

    private static IEnumerable<MethodDeclarationSyntax> GetShaderMethods(ShaderModel model)
    {
        var entry = model.EntryPointSyntax;
        if (entry is not null)
        {
            yield return entry;
        }

        foreach (var method in model.Syntax.Members.OfType<MethodDeclarationSyntax>())
        {
            if (method == entry)
            {
                continue;
            }

            var symbol = model.Symbol.GetMembers(method.Identifier.ValueText).OfType<IMethodSymbol>().FirstOrDefault();
            if (symbol is not null && IsCallable(symbol))
            {
                yield return method;
            }
        }

        foreach (var callable in model.Callables.Items)
        {
            if (callable.Syntax == entry || callable.Syntax.Ancestors().OfType<StructDeclarationSyntax>().FirstOrDefault() == model.Syntax)
            {
                continue;
            }

            yield return callable.Syntax;
        }
    }

    private static SemanticModel GetSemanticModelForMethod(SemanticModel rootSemanticModel, MethodDeclarationSyntax method)
        => method.SyntaxTree == rootSemanticModel.SyntaxTree
            ? rootSemanticModel
            : rootSemanticModel.Compilation.GetSemanticModel(method.SyntaxTree);

    private static IEnumerable<ShaderBodyDiagnosticModel> ValidateShaderMethod(MethodDeclarationSyntax method, SemanticModel semanticModel, CancellationToken cancellationToken)
    {
        SyntaxNode? root = method.Body;
        if (root is null)
        {
            root = method.ExpressionBody?.Expression;
        }

        if (root is null)
        {
            yield break;
        }

        if (method.Modifiers.Any(SyntaxKind.AsyncKeyword))
        {
            yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedAsync, method.Identifier.GetLocation());
        }

        foreach (var node in root.DescendantNodesAndSelf())
        {
            cancellationToken.ThrowIfCancellationRequested();

            switch (node)
            {
                case TryStatementSyntax tryStatement:
                    yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedExceptionHandling, tryStatement.TryKeyword.GetLocation());
                    break;
                case ThrowStatementSyntax throwStatement:
                    yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedExceptionHandling, throwStatement.ThrowKeyword.GetLocation());
                    break;
                case AwaitExpressionSyntax awaitExpression:
                    yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedAsync, awaitExpression.GetLocation());
                    break;
                case AnonymousFunctionExpressionSyntax lambda:
                    yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedExpression, lambda.GetLocation(), "lambda");
                    break;
                case SwitchStatementSyntax switchStatement:
                    yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedControlFlow, switchStatement.SwitchKeyword.GetLocation(), "switch");
                    break;
                case SwitchExpressionSyntax switchExpression:
                    yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedControlFlow, switchExpression.GetLocation(), "switch expression");
                    break;
                case ForEachStatementSyntax forEachStatement:
                    yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedControlFlow, forEachStatement.ForEachKeyword.GetLocation(), "foreach");
                    break;
                case GotoStatementSyntax gotoStatement:
                    yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedControlFlow, gotoStatement.GotoKeyword.GetLocation(), "goto");
                    break;
                case UnsafeStatementSyntax unsafeStatement:
                    yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedStatement, unsafeStatement.UnsafeKeyword.GetLocation(), "unsafe");
                    break;
                case FixedStatementSyntax fixedStatement:
                    yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedStatement, fixedStatement.FixedKeyword.GetLocation(), "fixed");
                    break;
                case ObjectCreationExpressionSyntax creation:
                    var type = semanticModel.GetTypeInfo(creation, cancellationToken).Type;
                    if (type is not null && !type.IsUnmanagedType)
                    {
                        yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedAllocation, creation.NewKeyword.GetLocation());
                    }

                    break;
                case ArrayCreationExpressionSyntax arrayCreation:
                    yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedAllocation, arrayCreation.NewKeyword.GetLocation());
                    break;
                case ImplicitArrayCreationExpressionSyntax implicitArrayCreation:
                    yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedAllocation, implicitArrayCreation.NewKeyword.GetLocation());
                    break;
                case StackAllocArrayCreationExpressionSyntax stackAlloc:
                    yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedExpression, stackAlloc.StackAllocKeyword.GetLocation(), "stackalloc");
                    break;
                case LocalDeclarationStatementSyntax localDeclaration:
                    foreach (var diagnostic in ValidateLocalDeclaration(localDeclaration, semanticModel, cancellationToken))
                    {
                        yield return diagnostic;
                    }

                    break;
                case IdentifierNameSyntax identifier:
                    foreach (var diagnostic in ValidateIdentifierReference(identifier, method, semanticModel, cancellationToken))
                    {
                        yield return diagnostic;
                    }

                    break;
                case InvocationExpressionSyntax invocation:
                    foreach (var diagnostic in ValidateCall(invocation, semanticModel, cancellationToken))
                    {
                        yield return diagnostic;
                    }

                    break;
                case ElementAccessExpressionSyntax elementAccess:
                    foreach (var diagnostic in ValidateElementAccess(elementAccess, semanticModel, cancellationToken))
                    {
                        yield return diagnostic;
                    }

                    break;
                case AssignmentExpressionSyntax assignment:
                    foreach (var diagnostic in ValidateAssignment(assignment, semanticModel, cancellationToken))
                    {
                        yield return diagnostic;
                    }

                    break;
            }
        }
    }

    private static IEnumerable<ShaderBodyDiagnosticModel> ValidateIdentifierReference(
        IdentifierNameSyntax identifier,
        MethodDeclarationSyntax method,
        SemanticModel semanticModel,
        CancellationToken cancellationToken)
    {
        if (semanticModel.GetSymbolInfo(identifier, cancellationToken).Symbol is not ILocalSymbol local)
        {
            yield break;
        }

        if (semanticModel.GetDeclaredSymbol(method, cancellationToken) is not IMethodSymbol methodSymbol)
        {
            yield break;
        }

        if (!SymbolEqualityComparer.Default.Equals(local.ContainingSymbol, methodSymbol) &&
            local.ContainingSymbol is IMethodSymbol { MethodKind: MethodKind.Ordinary, ContainingType.TypeKind: TypeKind.Submission or TypeKind.Class } &&
            local.ContainingSymbol.Name.StartsWith("<", StringComparison.Ordinal) &&
            local.Locations.Any(static location => location.SourceTree is not null))
        {
            yield return BodyDiagnostic(
                FeatherDiagnostics.TopLevelLocalUnsupported,
                identifier.GetLocation(),
                local.Name);
        }
    }

    private static IEnumerable<ShaderBodyDiagnosticModel> ValidateLocalDeclaration(
        LocalDeclarationStatementSyntax localDeclaration,
        SemanticModel semanticModel,
        CancellationToken cancellationToken)
    {
        foreach (var variable in localDeclaration.Declaration.Variables)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var symbol = semanticModel.GetDeclaredSymbol(variable, cancellationToken) as ILocalSymbol;
            if (symbol?.Type is null)
            {
                continue;
            }

            if (symbol.Type.TypeKind == TypeKind.Array)
            {
                yield return BodyDiagnostic(
                    FeatherDiagnostics.UnsupportedAllocation,
                    variable.Identifier.GetLocation());
                continue;
            }

            if (!IsSupportedComputeValueType(symbol.Type))
            {
                yield return BodyDiagnostic(
                    FeatherDiagnostics.UnsupportedExpression,
                    variable.Identifier.GetLocation(),
                    $"unsupported shader type '{symbol.Type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)}'");
                continue;
            }

            if (symbol.Type.IsReferenceType)
            {
                yield return BodyDiagnostic(
                    FeatherDiagnostics.UnsupportedExpression,
                    variable.Identifier.GetLocation(),
                    "reference type local");
            }
        }
    }

    private static IEnumerable<ShaderBodyDiagnosticModel> ValidateCall(InvocationExpressionSyntax invocation, SemanticModel semanticModel, CancellationToken cancellationToken)
    {
        var symbol = semanticModel.GetSymbolInfo(invocation, cancellationToken).Symbol as IMethodSymbol;
        if (symbol is null)
        {
            yield break;
        }

        foreach (var diagnostic in ValidateTextureSampleCall(invocation, symbol, semanticModel, cancellationToken))
        {
            yield return diagnostic;
        }

        if (symbol.IsGenericMethod)
        {
            yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedGenericUsage, invocation.GetLocation(), symbol.Name);
            yield break;
        }

        var containing = symbol.ContainingType?.ToDisplayString();
        if (containing is "Feather.Math.ShaderMath" or "Feather.Math.Hlsl" or "Feather.AD.AD" or "Feather.GpuBarrier" or "Feather.GpuAtomic")
        {
            yield break;
        }

        if (IsRecursiveShaderCall(symbol, invocation, semanticModel, cancellationToken))
        {
            yield return BodyDiagnostic(FeatherDiagnostics.RecursiveShaderFunction, invocation.GetLocation(), symbol.Name);
            yield break;
        }

        if ((symbol.IsVirtual || symbol.IsOverride || symbol.ContainingType?.TypeKind == TypeKind.Interface) && !IsAllowedShaderViewCall(symbol))
        {
            yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedVirtualCall, invocation.GetLocation(), symbol.Name);
            yield break;
        }

        if (IsCallable(symbol))
        {
            foreach (var diagnostic in ValidateCallableInvocation(invocation, symbol, semanticModel, cancellationToken))
            {
                yield return diagnostic;
            }

            yield break;
        }

        if (symbol.MethodKind is MethodKind.PropertyGet or MethodKind.PropertySet)
        {
            yield break;
        }

        if (IsAllowedShaderViewCall(symbol))
        {
            yield break;
        }

        yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedCall, invocation.GetLocation(), symbol.Name);
    }

    private static IEnumerable<ShaderBodyDiagnosticModel> ValidateCallableDeclaration(MethodDeclarationSyntax method, IMethodSymbol symbol)
    {
        if (method.Body is null && method.ExpressionBody is null)
        {
            yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedCall, method.Identifier.GetLocation(), $"{symbol.Name} must have a source body");
        }

        if (symbol.IsGenericMethod)
        {
            yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedGenericUsage, method.Identifier.GetLocation(), symbol.Name);
        }

        if (IsShaderLibraryCallable(symbol) && !symbol.IsStatic)
        {
            yield return BodyDiagnostic(
                FeatherDiagnostics.UnsupportedCall,
                method.Identifier.GetLocation(),
                $"{symbol.Name} must be static because it belongs to a [ShaderLibrary]");
        }

        if (!IsSupportedCallableType(symbol.ReturnType))
        {
            yield return BodyDiagnostic(
                FeatherDiagnostics.UnsupportedCall,
                method.ReturnType.GetLocation(),
                symbol.Name);
        }

        foreach (var parameter in symbol.Parameters)
        {
            if (parameter.RefKind is RefKind.Ref or RefKind.Out or RefKind.In || !IsSupportedCallableType(parameter.Type))
            {
                var parameterSyntax = method.ParameterList.Parameters
                    .FirstOrDefault(candidate => candidate.Identifier.ValueText == parameter.Name);
                yield return BodyDiagnostic(
                    FeatherDiagnostics.UnsupportedCall,
                    parameterSyntax?.GetLocation() ?? method.Identifier.GetLocation(),
                    symbol.Name);
            }
        }
    }

    private static IEnumerable<ShaderBodyDiagnosticModel> ValidateElementAccess(ElementAccessExpressionSyntax elementAccess, SemanticModel semanticModel, CancellationToken cancellationToken)
    {
        var expressionType = semanticModel.GetTypeInfo(elementAccess.Expression, cancellationToken).Type;
        if (expressionType is null || elementAccess.ArgumentList.Arguments.Count != 1)
        {
            yield break;
        }

        var resourceKind = GetShaderResourceViewKind(expressionType);
        if (resourceKind == ShaderResourceViewKind.None)
        {
            yield break;
        }

        var indexType = semanticModel.GetTypeInfo(elementAccess.ArgumentList.Arguments[0].Expression, cancellationToken).Type;
        var resourceName = elementAccess.Expression.ToString();
        if (resourceKind == ShaderResourceViewKind.Buffer && !IsIntCompatible(indexType))
        {
            yield return BodyDiagnostic(FeatherDiagnostics.BufferIndexInvalid, elementAccess.ArgumentList.Arguments[0].GetLocation(), resourceName);
        }

        if (resourceKind == ShaderResourceViewKind.Texture2D && !IsTextureIndexCompatible(indexType))
        {
            yield return BodyDiagnostic(FeatherDiagnostics.TextureIndexInvalid, elementAccess.ArgumentList.Arguments[0].GetLocation(), resourceName);
        }

        if (resourceKind == ShaderResourceViewKind.Texture3D && !IsTexture3DIndexCompatible(indexType))
        {
            yield return BodyDiagnostic(FeatherDiagnostics.TextureIndexInvalid, elementAccess.ArgumentList.Arguments[0].GetLocation(), resourceName);
        }

        if (!IsSimpleAssignmentWriteTarget(elementAccess) && GetShaderResourceAccess(expressionType) == ResourceAccessModel.Write)
        {
            yield return BodyDiagnostic(FeatherDiagnostics.ResourceAccessViolation, elementAccess.GetLocation(), resourceName);
        }
    }

    private static IEnumerable<ShaderBodyDiagnosticModel> ValidateAssignment(AssignmentExpressionSyntax assignment, SemanticModel semanticModel, CancellationToken cancellationToken)
    {
        if (assignment.Left is not ElementAccessExpressionSyntax left)
        {
            yield break;
        }

        var expressionType = semanticModel.GetTypeInfo(left.Expression, cancellationToken).Type;
        var access = GetShaderResourceAccess(expressionType);
        if (access is ResourceAccessModel.Read or ResourceAccessModel.Sample)
        {
            yield return BodyDiagnostic(FeatherDiagnostics.ResourceAccessViolation, left.GetLocation(), left.Expression.ToString());
        }

        foreach (var invocation in assignment.Right.DescendantNodesAndSelf().OfType<InvocationExpressionSyntax>())
        {
            cancellationToken.ThrowIfCancellationRequested();
            var symbol = semanticModel.GetSymbolInfo(invocation, cancellationToken).Symbol as IMethodSymbol;
            if (symbol is not null && IsKnownMathIntrinsic(symbol) && !IsSupportedElementwiseMathIntrinsic(symbol))
            {
                yield return BodyDiagnostic(
                    FeatherDiagnostics.UnsupportedElementwiseIntrinsic,
                    invocation.GetLocation(),
                    GetMethodMetadataName(symbol));
            }
        }
    }

    private static IEnumerable<ShaderBodyDiagnosticModel> ValidateTextureSampleCall(
        InvocationExpressionSyntax invocation,
        IMethodSymbol symbol,
        SemanticModel semanticModel,
        CancellationToken cancellationToken)
    {
        if (symbol.Name is not ("Sample" or "SampleLevel" or "SampleGrad") || !IsSampledTexture2DType(symbol.ContainingType))
        {
            yield break;
        }

        if (invocation.Expression is MemberAccessExpressionSyntax { Expression: var receiver })
        {
            var receiverType = semanticModel.GetTypeInfo(receiver, cancellationToken).Type;
            if (GetShaderResourceAccess(receiverType) != ResourceAccessModel.Sample)
            {
                yield return BodyDiagnostic(
                    FeatherDiagnostics.ResourceAccessViolation,
                    receiver.GetLocation(),
                    receiver.ToString());
            }
        }

        if (invocation.ArgumentList.Arguments.Count > 0)
        {
            var samplerArgument = invocation.ArgumentList.Arguments[0].Expression;
            var samplerType = semanticModel.GetTypeInfo(samplerArgument, cancellationToken).Type;
            if (!IsSamplerStateType(samplerType))
            {
                yield return BodyDiagnostic(
                    FeatherDiagnostics.UnsupportedCall,
                    samplerArgument.GetLocation(),
                    "Sample requires a SamplerState argument");
            }
        }
    }

    private static IEnumerable<ShaderBodyDiagnosticModel> ValidateCallableInvocation(
        InvocationExpressionSyntax invocation,
        IMethodSymbol symbol,
        SemanticModel semanticModel,
        CancellationToken cancellationToken)
    {
        var containingShader = invocation.Ancestors()
            .OfType<StructDeclarationSyntax>()
            .Select(structSyntax => semanticModel.GetDeclaredSymbol(structSyntax, cancellationToken))
            .FirstOrDefault(type => type is not null);

        if (containingShader is null || !SymbolEqualityComparer.Default.Equals(symbol.ContainingType, containingShader))
        {
            if (!IsShaderLibraryCallable(symbol))
            {
                yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedCall, invocation.GetLocation(), symbol.Name);
                yield break;
            }

            if (!symbol.IsStatic)
            {
                yield return BodyDiagnostic(
                    FeatherDiagnostics.UnsupportedCall,
                    invocation.GetLocation(),
                    $"{symbol.Name} must be static because it belongs to a [ShaderLibrary]");
                yield break;
            }

            if (!HasSourceAvailableMethodBody(symbol))
            {
                yield return BodyDiagnostic(
                    FeatherDiagnostics.UnsupportedCall,
                    invocation.GetLocation(),
                    $"{symbol.Name} must be source-available to be imported from a [ShaderLibrary]");
                yield break;
            }
        }

        if (!SymbolEqualityComparer.Default.Equals(symbol.ContainingType, containingShader) && !IsShaderLibraryCallable(symbol))
        {
            yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedCall, invocation.GetLocation(), symbol.Name);
            yield break;
        }

        foreach (var parameter in symbol.Parameters)
        {
            if (parameter.RefKind is RefKind.Ref or RefKind.Out or RefKind.In)
            {
                yield return BodyDiagnostic(FeatherDiagnostics.UnsupportedCall, invocation.GetLocation(), symbol.Name);
                yield break;
            }
        }
    }

    private static bool HasSourceAvailableMethodBody(IMethodSymbol symbol)
        => symbol.DeclaringSyntaxReferences.Any(static reference =>
            reference.GetSyntax() is MethodDeclarationSyntax { Body: not null } or MethodDeclarationSyntax { ExpressionBody: not null });

    private static bool IsSimpleAssignmentWriteTarget(ElementAccessExpressionSyntax elementAccess)
        => elementAccess.Parent is AssignmentExpressionSyntax assignment
            && assignment.IsKind(SyntaxKind.SimpleAssignmentExpression)
            && assignment.Left == elementAccess;

    private static bool IsKnownMathIntrinsic(IMethodSymbol symbol)
    {
        var containing = symbol.ContainingType?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        return containing is "global::Feather.Math.ShaderMath" or "global::Feather.Math.Hlsl";
    }

    private static bool IsSupportedElementwiseMathIntrinsic(IMethodSymbol symbol)
        => symbol.Name switch
        {
            "Sin" or "Cos" or "Tan" or "Exp" or "Log" or "Sqrt" or "InverseSqrt"
                => ShaderSemanticFacts.HasFloatSignature(symbol, 1),
            "Abs" or "Floor" or "Ceil" or "Round" or "Fract" or "Saturate"
                => ShaderSemanticFacts.HasFloatOrMatchingFloatVectorUnarySignature(symbol),
            "Length" => symbol.Parameters.Length == 1 && symbol.ReturnType.SpecialType == SpecialType.System_Single
                && ShaderSemanticFacts.IsFeatherVectorType(symbol.Parameters[0].Type),
            "Normalize" => symbol.Parameters.Length == 1
                && ShaderSemanticFacts.IsFeatherVectorType(symbol.ReturnType)
                && ShaderSemanticFacts.IsFeatherVectorType(symbol.Parameters[0].Type),
            "Pow" => ShaderSemanticFacts.HasFloatSignature(symbol, 2),
            "Min" or "Max" => ShaderSemanticFacts.HasFloatOrMatchingFloatVectorBinarySignature(symbol),
            "Clamp" => ShaderSemanticFacts.HasFloatOrMatchingFloatVectorClampSignature(symbol),
            "Lerp" or "Mix" => ShaderSemanticFacts.HasFloatOrMatchingFloatVectorLerpSignature(symbol),
            "Smoothstep" => ShaderSemanticFacts.HasFloatSignature(symbol, 3),
            "Dot" => ShaderSemanticFacts.HasFloatVectorDotSignature(symbol),
            "Cross" => ShaderSemanticFacts.HasFloat3CrossSignature(symbol),
            "Mul" => ShaderSemanticFacts.HasMatrixMulSignature(symbol),
            "Transpose" or "Inverse" => ShaderSemanticFacts.HasMatrixTransformSignature(symbol),
            "Determinant" => ShaderSemanticFacts.HasMatrixScalarSignature(symbol),
            "Hadamard" => ShaderSemanticFacts.HasMatrixHadamardSignature(symbol),
            _ => false
        };

    private static string GetMethodMetadataName(IMethodSymbol symbol)
        => symbol.ContainingType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) + "." + symbol.Name;

    private static ShaderKind? GetShaderKind(INamedTypeSymbol symbol)
    {
        foreach (var iface in symbol.AllInterfaces)
        {
            var name = iface.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
            if (name == "global::Feather.IKernel1D")
            {
                return ShaderKind.Compute1D;
            }

            if (name == "global::Feather.IKernel2D")
            {
                return ShaderKind.Compute2D;
            }

            if (name == "global::Feather.IKernel3D")
            {
                return ShaderKind.Compute3D;
            }

            if (name.StartsWith("global::Feather.IVertexShader<", StringComparison.Ordinal))
            {
                return ShaderKind.Vertex;
            }

            if (name.StartsWith("global::Feather.IFragmentShader<", StringComparison.Ordinal))
            {
                return ShaderKind.Fragment;
            }
        }

        return null;
    }

    private static (MethodDeclarationSyntax? Syntax, IMethodSymbol? Symbol) ResolveEntryPoint(
        StructDeclarationSyntax syntax,
        INamedTypeSymbol symbol)
    {
        var methods = GetDeclaredMethods(syntax, symbol).ToArray();
        var marked = methods
            .Where(static method => HasAttribute(method.Symbol, "Feather.EntryAttribute"))
            .ToArray();
        if (marked.Length == 1)
        {
            return marked[0];
        }

        if (marked.Length > 1)
        {
            return default;
        }

        var legacy = methods
            .Where(static method => method.Syntax.Identifier.ValueText == "Execute")
            .ToArray();
        return legacy.Length == 1 ? legacy[0] : default;
    }

    private static IEnumerable<(MethodDeclarationSyntax Syntax, IMethodSymbol Symbol)> GetDeclaredMethods(
        StructDeclarationSyntax syntax,
        INamedTypeSymbol symbol)
    {
        foreach (var method in syntax.Members.OfType<MethodDeclarationSyntax>())
        {
            var methodSymbol = symbol.GetMembers(method.Identifier.ValueText)
                .OfType<IMethodSymbol>()
                .FirstOrDefault(candidate => candidate.DeclaringSyntaxReferences.Any(reference => reference.GetSyntax() == method));
            if (methodSymbol is not null)
            {
                yield return (method, methodSymbol);
            }
        }
    }

    private static bool IsCompatibleEntryPoint(ShaderModel model)
    {
        var entry = model.EntryPointSymbol;
        if (model.EntryPointSyntax is null || entry is null ||
            entry.DeclaredAccessibility != Accessibility.Public ||
            entry.IsStatic ||
            entry.IsGenericMethod ||
            entry.RefKind != RefKind.None)
        {
            return false;
        }

        if (model.EntryPointSyntax.Body is null && model.EntryPointSyntax.ExpressionBody is null)
        {
            return false;
        }

        if (entry.Parameters.Any(static parameter => parameter.RefKind != RefKind.None))
        {
            return false;
        }

        return model.Kind switch
        {
            ShaderKind.Compute1D or ShaderKind.Compute2D or ShaderKind.Compute3D =>
                entry.Parameters.Length == 0 && entry.ReturnsVoid,
            ShaderKind.Vertex =>
                entry.Parameters.Length == 0 &&
                SameType(entry.ReturnType, GetGraphicsVaryingType(model)),
            ShaderKind.Fragment =>
                entry.Parameters.Length == 1 &&
                SameType(entry.Parameters[0].Type, GetGraphicsVaryingType(model)) &&
                IsCompatibleFragmentReturnType(model, entry.ReturnType),
            _ => false
        };
    }

    private static bool IsCompatibleFragmentReturnType(ShaderModel model, ITypeSymbol returnType)
    {
        var outputType = GetFragmentOutputType(model);
        return outputType is null
            ? returnType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) == "global::Feather.Math.float4"
            : SameType(returnType, outputType);
    }

    private static bool SameType(ITypeSymbol? left, ITypeSymbol? right)
        => SymbolEqualityComparer.Default.Equals(left, right);

    private static EquatableArray<CallableMethodModel> DiscoverDeclaredCallables(
        StructDeclarationSyntax syntax,
        INamedTypeSymbol symbol,
        IMethodSymbol? entryPoint)
    {
        var results = new List<CallableMethodModel>();
        foreach (var (method, ms) in GetDeclaredMethods(syntax, symbol))
        {
            if (entryPoint is not null && SymbolEqualityComparer.Default.Equals(ms, entryPoint)) continue;
            if (!IsCallable(ms))
                continue;
            var pars = ms.Parameters.Select(p => new CallableParameterModel(
                p.Name, p.Type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat),
                p.RefKind is RefKind.Ref or RefKind.Out or RefKind.In)).ToArray();
            results.Add(new CallableMethodModel(method, ms, ms.Name, GetCallableMangledName(ms),
                ms.ReturnType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat),
                ms.IsStatic, new EquatableArray<CallableParameterModel>(pars)));
        }
        return new EquatableArray<CallableMethodModel>(results);
    }

    private static EquatableArray<CallableMethodModel> DiscoverCallables(
        StructDeclarationSyntax syntax,
        INamedTypeSymbol symbol,
        MethodDeclarationSyntax? entryPointSyntax,
        IMethodSymbol? entryPoint,
        SemanticModel semanticModel,
        CancellationToken cancellationToken)
    {
        var results = new List<CallableMethodModel>();
        var byId = new Dictionary<string, CallableMethodModel>(StringComparer.Ordinal);
        var pending = new Queue<IMethodSymbol>();
        var queued = new HashSet<string>(StringComparer.Ordinal);
        var compilation = semanticModel.Compilation;

        void AddCallable(MethodDeclarationSyntax method, IMethodSymbol methodSymbol)
        {
            var id = GetCallableId(methodSymbol.OriginalDefinition);
            if (byId.ContainsKey(id))
            {
                return;
            }

            var pars = methodSymbol.Parameters.Select(p => new CallableParameterModel(
                p.Name,
                p.Type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat),
                p.RefKind is RefKind.Ref or RefKind.Out or RefKind.In)).ToArray();
            var model = new CallableMethodModel(
                method,
                methodSymbol,
                methodSymbol.Name,
                GetCallableMangledName(methodSymbol),
                methodSymbol.ReturnType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat),
                methodSymbol.IsStatic,
                new EquatableArray<CallableParameterModel>(pars));
            byId.Add(id, model);
            results.Add(model);
        }

        void Enqueue(IMethodSymbol methodSymbol)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!IsCallable(methodSymbol))
            {
                return;
            }

            if (entryPoint is not null &&
                SymbolEqualityComparer.Default.Equals(methodSymbol.OriginalDefinition, entryPoint.OriginalDefinition))
            {
                return;
            }

            var id = GetCallableId(methodSymbol.OriginalDefinition);
            if (queued.Add(id))
            {
                pending.Enqueue(methodSymbol);
            }
        }

        void Scan(MethodDeclarationSyntax? method, SemanticModel methodSemanticModel)
        {
            if (method is null)
            {
                return;
            }

            SyntaxNode? root = method.Body;
            if (root is null)
            {
                root = method.ExpressionBody?.Expression;
            }

            if (root is null)
            {
                return;
            }

            foreach (var invocation in root.DescendantNodesAndSelf().OfType<InvocationExpressionSyntax>())
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (methodSemanticModel.GetSymbolInfo(invocation, cancellationToken).Symbol is IMethodSymbol target)
                {
                    Enqueue(target);
                }
            }
        }

        foreach (var (method, methodSymbol) in GetDeclaredMethods(syntax, symbol))
        {
            if (entryPoint is not null && SymbolEqualityComparer.Default.Equals(methodSymbol, entryPoint))
            {
                continue;
            }

            if (!IsCallable(methodSymbol))
            {
                continue;
            }

            AddCallable(method, methodSymbol);
            Scan(method, semanticModel);
        }

        Scan(entryPointSyntax, semanticModel);

        while (pending.Count > 0)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var methodSymbol = pending.Dequeue();
            var id = GetCallableId(methodSymbol.OriginalDefinition);
            if (byId.TryGetValue(id, out var existing))
            {
                Scan(existing.Syntax, compilation.GetSemanticModel(existing.Syntax.SyntaxTree));
                continue;
            }

            if (!IsShaderLibraryCallable(methodSymbol) ||
                !TryGetMethodSyntax(methodSymbol, cancellationToken, out var methodSyntax))
            {
                continue;
            }

            AddCallable(methodSyntax, methodSymbol);
            Scan(methodSyntax, compilation.GetSemanticModel(methodSyntax.SyntaxTree));
        }

        return new EquatableArray<CallableMethodModel>(results);
    }

    private static bool TryGetMethodSyntax(
        IMethodSymbol symbol,
        CancellationToken cancellationToken,
        out MethodDeclarationSyntax method)
    {
        foreach (var reference in symbol.DeclaringSyntaxReferences)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (reference.GetSyntax(cancellationToken) is MethodDeclarationSyntax syntax)
            {
                method = syntax;
                return true;
            }
        }

        method = null!;
        return false;
    }

    private static IEnumerable<ResourceModel> GetPrimaryConstructorResources(StructDeclarationSyntax syntax, INamedTypeSymbol symbol)
    {
        if (syntax.ParameterList is null)
        {
            yield break;
        }

        var constructor = symbol.InstanceConstructors.FirstOrDefault(ctor => ctor.Parameters.Length == syntax.ParameterList.Parameters.Count);
        if (constructor is null)
        {
            yield break;
        }

        for (var i = 0; i < constructor.Parameters.Length; i++)
        {
            var parameter = constructor.Parameters[i];
            var binding = ReadBinding(parameter) ?? (uint)i;
            var typeName = parameter.Type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
            var resource = ClassifyResource(binding, parameter.Name, parameter.Type, typeName);
            yield return resource;
        }
    }

    private static ResourceModel ClassifyResource(uint binding, string name, ITypeSymbol type, string typeName)
    {
        if (type is INamedTypeSymbol named && named.IsGenericType)
        {
            var genericName = named.ConstructedFrom.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
            var elementSymbol = named.TypeArguments[0];
            var elementType = elementSymbol.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
            var storageElementSupported = IsSupportedComputeStorageElementType(elementSymbol);
            var textureElementSupported = IsSupportedComputeTextureElementType(elementSymbol);
            return genericName switch
            {
                "global::Feather.Resources.ReadOnlyBuffer<T>" => new(binding, name, typeName, elementType, ResourceKindModel.Buffer, ResourceAccessModel.Read, IsComputeStorageElementSupported: storageElementSupported),
                "global::Feather.Resources.WriteOnlyBuffer<T>" => new(binding, name, typeName, elementType, ResourceKindModel.Buffer, ResourceAccessModel.Write, IsComputeStorageElementSupported: storageElementSupported),
                "global::Feather.Resources.ReadWriteBuffer<T>" => new(binding, name, typeName, elementType, ResourceKindModel.Buffer, ResourceAccessModel.ReadWrite, IsComputeStorageElementSupported: storageElementSupported),
                "global::Feather.Resources.ReadOnlyTexture2D<T>" => new(binding, name, typeName, elementType, ResourceKindModel.Texture2D, ResourceAccessModel.Read, IsComputeStorageElementSupported: textureElementSupported),
                "global::Feather.Resources.WriteOnlyTexture2D<T>" => new(binding, name, typeName, elementType, ResourceKindModel.Texture2D, ResourceAccessModel.Write, IsComputeStorageElementSupported: textureElementSupported),
                "global::Feather.Resources.ReadWriteTexture2D<T>" => new(binding, name, typeName, elementType, ResourceKindModel.Texture2D, ResourceAccessModel.ReadWrite, IsComputeStorageElementSupported: textureElementSupported),
                "global::Feather.Resources.ReadWriteNormalizedTexture2D<T>" => new(binding, name, typeName, elementType, ResourceKindModel.Texture2D, ResourceAccessModel.ReadWrite, IsComputeStorageElementSupported: textureElementSupported),
                "global::Feather.Resources.SampledTexture2D<T>" => new(binding, name, typeName, elementType, ResourceKindModel.Texture2D, ResourceAccessModel.Sample, IsComputeStorageElementSupported: textureElementSupported),
                "global::Feather.Resources.ReadOnlyTexture3D<T>" => new(binding, name, typeName, elementType, ResourceKindModel.Texture3D, ResourceAccessModel.Read, IsComputeStorageElementSupported: textureElementSupported),
                "global::Feather.Resources.WriteOnlyTexture3D<T>" => new(binding, name, typeName, elementType, ResourceKindModel.Texture3D, ResourceAccessModel.Write, IsComputeStorageElementSupported: textureElementSupported),
                "global::Feather.Resources.ReadWriteTexture3D<T>" => new(binding, name, typeName, elementType, ResourceKindModel.Texture3D, ResourceAccessModel.ReadWrite, IsComputeStorageElementSupported: textureElementSupported),
                "global::Feather.Resources.ReadWriteNormalizedTexture3D<T>" => new(binding, name, typeName, elementType, ResourceKindModel.Texture3D, ResourceAccessModel.ReadWrite, IsComputeStorageElementSupported: textureElementSupported),
                "global::Feather.Resources.Uniform<T>" => new(binding, name, typeName, elementType, ResourceKindModel.PushConstant, ResourceAccessModel.Read, IsUniformWrapper: true, IsPushConstantSupported: IsSupportedPushConstantType(named.TypeArguments[0])),
                _ => new(binding, name, typeName, typeName, ResourceKindModel.PushConstant, ResourceAccessModel.Read, IsPushConstantSupported: IsSupportedPushConstantType(type))
            };
        }

        if (typeName == "global::Feather.Resources.SamplerState")
        {
            return new(binding, name, typeName, typeName, ResourceKindModel.Sampler, ResourceAccessModel.Sample);
        }

        return new(binding, name, typeName, typeName, ResourceKindModel.PushConstant, ResourceAccessModel.Read, IsPushConstantSupported: IsSupportedPushConstantType(type));
    }

    private static ThreadGroupModel ReadThreadGroup(INamedTypeSymbol symbol)
    {
        var attribute = symbol.GetAttributes().FirstOrDefault(attr => attr.AttributeClass?.ToDisplayString() == "Feather.ThreadGroupSizeAttribute");
        if (attribute is null)
        {
            return new ThreadGroupModel(256, 1, 1);
        }

        if (attribute.ConstructorArguments.Length == 1 && attribute.ConstructorArguments[0].Type?.ToDisplayString() == "Feather.DefaultThreadGroupSizes")
        {
            return (int)attribute.ConstructorArguments[0].Value! switch
            {
                0 => new ThreadGroupModel(256, 1, 1),
                1 => new ThreadGroupModel(16, 16, 1),
                2 => new ThreadGroupModel(8, 8, 4),
                _ => new ThreadGroupModel(256, 1, 1)
            };
        }

        var values = attribute.ConstructorArguments.Select(arg => arg.Value is int value ? value : 1).ToArray();
        return new ThreadGroupModel(values.ElementAtOrDefault(0), values.ElementAtOrDefault(1) == 0 ? 1 : values.ElementAtOrDefault(1), values.ElementAtOrDefault(2) == 0 ? 1 : values.ElementAtOrDefault(2));
    }

    private static bool HasKernelBoundsCheckFalse(INamedTypeSymbol symbol)
        => symbol.GetAttributes()
            .Where(attr => attr.AttributeClass?.ToDisplayString() == "Feather.KernelAttribute")
            .SelectMany(attr => attr.NamedArguments)
            .Any(arg => arg.Key == "BoundsCheck" && arg.Value.Value is false);

    private static bool HasAttribute(ISymbol symbol, string metadataName)
        => symbol.GetAttributes().Any(attr => attr.AttributeClass?.ToDisplayString() == metadataName);

    private static ITypeSymbol? GetGraphicsVaryingType(ShaderModel model)
    {
        var prefix = model.Kind == ShaderKind.Vertex
            ? "global::Feather.IVertexShader<"
            : "global::Feather.IFragmentShader<";
        return model.Symbol.AllInterfaces
            .FirstOrDefault(iface => iface.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat).StartsWith(prefix, StringComparison.Ordinal))
            ?.TypeArguments
            .FirstOrDefault();
    }

    private static ITypeSymbol? GetFragmentOutputType(ShaderModel model)
    {
        var fragmentInterface = model.Symbol.AllInterfaces
            .Where(iface => iface.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat).StartsWith("global::Feather.IFragmentShader<", StringComparison.Ordinal))
            .OrderByDescending(static iface => iface.TypeArguments.Length)
            .FirstOrDefault();

        if (fragmentInterface is null)
        {
            return null;
        }

        return fragmentInterface.TypeArguments.Length >= 2
            ? fragmentInterface.TypeArguments[1]
            : null;
    }

    private static bool IsSupportedGraphicsVarying(ITypeSymbol type)
    {
        if (!type.IsUnmanagedType)
        {
            return false;
        }

        var typeName = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        if (typeName == "global::Feather.Math.float4")
        {
            return true;
        }

        return type is INamedTypeSymbol named
            && HasAttribute(named, "Feather.GpuStructAttribute")
            && HasPositionMember(named);
    }

    private static bool HasPositionMember(INamedTypeSymbol symbol)
        => symbol.GetMembers().Any(static member =>
        {
            var type = member switch
            {
                IFieldSymbol { IsStatic: false } field => field.Type,
                IPropertySymbol { IsStatic: false, Parameters.Length: 0 } property => property.Type,
                _ => null
            };

            return type?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) == "global::Feather.Math.float4"
                && member.GetAttributes().Any(attr => attr.AttributeClass?.ToDisplayString() == "Feather.PositionAttribute");
        });

    private static bool IsSupportedFragmentOutput(ITypeSymbol type, out string reason)
    {
        reason = "";
        if (!type.IsUnmanagedType)
        {
            reason = "output type must be unmanaged";
            return false;
        }

        var typeName = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        if (typeName == "global::Feather.Math.float4")
        {
            return true;
        }

        if (type is not INamedTypeSymbol named || !HasAttribute(named, "Feather.GpuStructAttribute"))
        {
            reason = "output must be float4 or a [GpuStruct] with [Color(n)] float4 members";
            return false;
        }

        var colors = new HashSet<uint>();
        var fieldCount = 0;
        foreach (var member in named.GetMembers())
        {
            var memberType = member switch
            {
                IFieldSymbol { IsStatic: false } field => field.Type,
                IPropertySymbol { IsStatic: false, Parameters.Length: 0 } property => property.Type,
                _ => null
            };
            if (memberType is null)
            {
                continue;
            }

            fieldCount++;
            if (memberType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) != "global::Feather.Math.float4")
            {
                reason = "MRT output members must be float4";
                return false;
            }

            var colorAttribute = member.GetAttributes()
                .FirstOrDefault(static attr => attr.AttributeClass?.ToDisplayString() == "Feather.ColorAttribute");
            if (colorAttribute is null)
            {
                reason = "MRT output members must have [Color(n)]";
                return false;
            }

            var index = colorAttribute.ConstructorArguments.Length > 0
                && colorAttribute.ConstructorArguments[0].Value is uint colorIndex
                    ? colorIndex
                    : 0u;
            if (index >= 8)
            {
                reason = "MRT color index must be less than 8";
                return false;
            }
            if (!colors.Add(index))
            {
                reason = "MRT color indices must be unique";
                return false;
            }
        }

        if (fieldCount == 0)
        {
            reason = "MRT output struct must contain at least one [Color(n)] float4 member";
            return false;
        }

        for (uint i = 0; i < fieldCount; i++)
        {
            if (!colors.Contains(i))
            {
                reason = "MRT color indices must be dense starting at 0";
                return false;
            }
        }

        return true;
    }

    private static bool IsSupportedCapturedField(ITypeSymbol type)
    {
        var kind = GetShaderResourceViewKind(type);
        if (kind != ShaderResourceViewKind.None)
        {
            return true;
        }

        var typeName = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        return typeName == "global::Feather.Resources.SamplerState"
            || IsSupportedPushConstantType(type);
    }

    private static bool IsRecursiveShaderCall(IMethodSymbol symbol, InvocationExpressionSyntax invocation, SemanticModel semanticModel, CancellationToken cancellationToken)
    {
        var containingMethod = invocation.Ancestors()
            .OfType<MethodDeclarationSyntax>()
            .Select(method => semanticModel.GetDeclaredSymbol(method, cancellationToken))
            .FirstOrDefault(method => method is not null);
        return containingMethod is not null
            && SymbolEqualityComparer.Default.Equals(symbol.OriginalDefinition, containingMethod.OriginalDefinition);
    }

    private static IEnumerable<ShaderBodyDiagnosticModel> GetCallableRecursionDiagnostics(
        ShaderModel model,
        SemanticModel semanticModel,
        CancellationToken cancellationToken)
    {
        var callables = model.Callables.Items.ToArray();
        if (callables.Length == 0)
        {
            yield break;
        }

        var callableIds = callables.ToDictionary(
            callable => GetCallableId(callable.Symbol),
            callable => callable,
            StringComparer.Ordinal);
        var adjacency = callables.ToDictionary(
            callable => GetCallableId(callable.Symbol),
            _ => new List<CallableEdge>(),
            StringComparer.Ordinal);

        foreach (var callable in callables)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var id = GetCallableId(callable.Symbol);
            var callableSemanticModel = GetSemanticModelForMethod(semanticModel, callable.Syntax);
            SyntaxNode? root = callable.Syntax.Body;
            if (root is null)
            {
                root = callable.Syntax.ExpressionBody?.Expression;
            }

            if (root is null)
            {
                continue;
            }

            foreach (var invocation in root.DescendantNodesAndSelf().OfType<InvocationExpressionSyntax>())
            {
                cancellationToken.ThrowIfCancellationRequested();

                if (callableSemanticModel.GetSymbolInfo(invocation, cancellationToken).Symbol is not IMethodSymbol target
                    || !IsCallable(target))
                {
                    continue;
                }

                var targetId = GetCallableId(target.OriginalDefinition);
                if (callableIds.ContainsKey(targetId))
                {
                    adjacency[id].Add(new CallableEdge(targetId, target.Name, invocation.GetLocation()));
                }
            }
        }

        var state = new Dictionary<string, int>(StringComparer.Ordinal);
        var stack = new List<string>();
        var reported = new HashSet<string>(StringComparer.Ordinal);
        foreach (var id in adjacency.Keys)
        {
            foreach (var diagnostic in Visit(id))
            {
                yield return diagnostic;
            }
        }

        IEnumerable<ShaderBodyDiagnosticModel> Visit(string id)
        {
            if (state.TryGetValue(id, out var currentState))
            {
                if (currentState == 1)
                {
                    var cycleStart = stack.IndexOf(id);
                    var cycleIds = cycleStart >= 0 ? stack.Skip(cycleStart).Append(id).ToArray() : [id];
                    foreach (var cycleId in cycleIds)
                    {
                        reported.Add(cycleId);
                    }
                }

                yield break;
            }

            state[id] = 1;
            stack.Add(id);

            foreach (var edge in adjacency[id])
            {
                if (state.TryGetValue(edge.TargetId, out var targetState) && targetState == 1)
                {
                    if (reported.Add(id))
                    {
                        yield return BodyDiagnostic(
                            FeatherDiagnostics.RecursiveShaderFunction,
                            edge.Location,
                            edge.TargetName);
                    }

                    continue;
                }

                foreach (var diagnostic in Visit(edge.TargetId))
                {
                    yield return diagnostic;
                }
            }

            stack.RemoveAt(stack.Count - 1);
            state[id] = 2;
        }
    }

    private static string GetCallableId(IMethodSymbol symbol)
        => GetCallableMangledName(symbol.OriginalDefinition);

    internal static string GetCallableMangledName(IMethodSymbol symbol)
    {
        var builder = new System.Text.StringBuilder();
        builder.Append(symbol.ContainingType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat));
        builder.Append('.');
        builder.Append(symbol.Name);
        foreach (var parameter in symbol.Parameters)
        {
            builder.Append('_');
            builder.Append(parameter.Type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat));
        }

        return SanitizeMangledName(builder.ToString());
    }

    private static string SanitizeMangledName(string value)
    {
        var builder = new System.Text.StringBuilder(value.Length);
        foreach (var ch in value)
        {
            builder.Append(char.IsLetterOrDigit(ch) ? ch : '_');
        }

        return builder.ToString();
    }

    private static bool IsCallable(IMethodSymbol symbol)
        => symbol.GetAttributes().Any(static a => a.AttributeClass?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)
            is "global::Feather.CallableAttribute" or "global::Feather.ShaderFunctionAttribute");

    private static bool IsShaderLibraryCallable(IMethodSymbol symbol)
        => IsCallable(symbol) && ShaderSemanticFacts.IsShaderLibraryType(symbol.ContainingType);

    private static ShaderBodyDiagnosticModel BodyDiagnostic(
        DiagnosticDescriptor descriptor,
        Location? location,
        params string[] arguments)
        => new(descriptor, location, new EquatableArray<string>(arguments));

    private static bool IsSupportedCallableType(ITypeSymbol type)
    {
        if (type.SpecialType is SpecialType.System_Void)
        {
            return true;
        }

        if (type.IsReferenceType)
        {
            return false;
        }

        if (GetShaderResourceViewKind(type) != ShaderResourceViewKind.None)
        {
            return true;
        }

        if (type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) == "global::Feather.Resources.SamplerState")
        {
            return true;
        }

        return IsSupportedComputeValueType(type);
    }

    private static bool IsAllowedShaderViewCall(IMethodSymbol symbol)
    {
        var containingDisplay = symbol.ContainingType?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) ?? string.Empty;
        var allowedPrefixes = new[]
        {
            "global::Feather.Resources.SampledTexture2D<",
            "global::Feather.Resources.ReadOnlyTexture2D<",
            "global::Feather.Resources.WriteOnlyTexture2D<",
            "global::Feather.Resources.ReadWriteTexture2D<",
            "global::Feather.Resources.ReadWriteNormalizedTexture2D<",
            "global::Feather.Resources.ReadOnlyTexture3D<",
            "global::Feather.Resources.WriteOnlyTexture3D<",
            "global::Feather.Resources.ReadWriteTexture3D<",
            "global::Feather.Resources.ReadWriteNormalizedTexture3D<"
        };

        return allowedPrefixes.Any(prefix => containingDisplay.StartsWith(prefix, StringComparison.Ordinal));
    }

    private static ShaderResourceViewKind GetShaderResourceViewKind(ITypeSymbol? type)
    {
        if (type is not INamedTypeSymbol named || !named.IsGenericType)
        {
            return ShaderResourceViewKind.None;
        }

        var genericName = named.ConstructedFrom.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        return genericName switch
        {
            "global::Feather.Resources.ReadOnlyBuffer<T>" => ShaderResourceViewKind.Buffer,
            "global::Feather.Resources.WriteOnlyBuffer<T>" => ShaderResourceViewKind.Buffer,
            "global::Feather.Resources.ReadWriteBuffer<T>" => ShaderResourceViewKind.Buffer,
            "global::Feather.Resources.ReadOnlyTexture2D<T>" => ShaderResourceViewKind.Texture2D,
            "global::Feather.Resources.WriteOnlyTexture2D<T>" => ShaderResourceViewKind.Texture2D,
            "global::Feather.Resources.ReadWriteTexture2D<T>" => ShaderResourceViewKind.Texture2D,
            "global::Feather.Resources.ReadWriteNormalizedTexture2D<T>" => ShaderResourceViewKind.Texture2D,
            "global::Feather.Resources.SampledTexture2D<T>" => ShaderResourceViewKind.Texture2D,
            "global::Feather.Resources.ReadOnlyTexture3D<T>" => ShaderResourceViewKind.Texture3D,
            "global::Feather.Resources.WriteOnlyTexture3D<T>" => ShaderResourceViewKind.Texture3D,
            "global::Feather.Resources.ReadWriteTexture3D<T>" => ShaderResourceViewKind.Texture3D,
            "global::Feather.Resources.ReadWriteNormalizedTexture3D<T>" => ShaderResourceViewKind.Texture3D,
            _ => ShaderResourceViewKind.None
        };
    }

    private static ResourceAccessModel? GetShaderResourceAccess(ITypeSymbol? type)
    {
        if (type is not INamedTypeSymbol named || !named.IsGenericType)
        {
            return null;
        }

        var genericName = named.ConstructedFrom.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        return genericName switch
        {
            "global::Feather.Resources.ReadOnlyBuffer<T>" => ResourceAccessModel.Read,
            "global::Feather.Resources.WriteOnlyBuffer<T>" => ResourceAccessModel.Write,
            "global::Feather.Resources.ReadWriteBuffer<T>" => ResourceAccessModel.ReadWrite,
            "global::Feather.Resources.ReadOnlyTexture2D<T>" => ResourceAccessModel.Read,
            "global::Feather.Resources.WriteOnlyTexture2D<T>" => ResourceAccessModel.Write,
            "global::Feather.Resources.ReadWriteTexture2D<T>" => ResourceAccessModel.ReadWrite,
            "global::Feather.Resources.ReadWriteNormalizedTexture2D<T>" => ResourceAccessModel.ReadWrite,
            "global::Feather.Resources.SampledTexture2D<T>" => ResourceAccessModel.Sample,
            "global::Feather.Resources.ReadOnlyTexture3D<T>" => ResourceAccessModel.Read,
            "global::Feather.Resources.WriteOnlyTexture3D<T>" => ResourceAccessModel.Write,
            "global::Feather.Resources.ReadWriteTexture3D<T>" => ResourceAccessModel.ReadWrite,
            "global::Feather.Resources.ReadWriteNormalizedTexture3D<T>" => ResourceAccessModel.ReadWrite,
            _ => null
        };
    }

    private static bool IsIntCompatible(ITypeSymbol? type)
        => type?.SpecialType is SpecialType.System_Int32 or SpecialType.System_UInt32 or SpecialType.System_Int16 or SpecialType.System_UInt16 or SpecialType.System_Byte or SpecialType.System_SByte;

    private static bool IsTextureIndexCompatible(ITypeSymbol? type)
    {
        var typeName = type?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        return typeName is "global::Feather.Math.int2" or "global::Feather.Math.int3";
    }

    private static bool IsTexture3DIndexCompatible(ITypeSymbol? type)
        => type?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) == "global::Feather.Math.int3";

    private static bool IsSampledTexture2DType(ITypeSymbol? type)
        => type is INamedTypeSymbol named
            && named.IsGenericType
            && named.ConstructedFrom.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) == "global::Feather.Resources.SampledTexture2D<T>";

    private static bool IsSupportedAutoDiffValueType(ITypeSymbol? type)
    {
        if (type is null)
        {
            return false;
        }

        var name = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        return type.SpecialType == SpecialType.System_Single
            || name is "global::Feather.Math.float2"
                or "global::Feather.Math.float3"
                or "global::Feather.Math.float4";
    }

    private static bool IsBufferElementArgument(
        ExpressionSyntax expression,
        SemanticModel semanticModel,
        CancellationToken cancellationToken,
        out ITypeSymbol? sourceType)
        => TryTraceAdParameterSource(expression, semanticModel, cancellationToken, out var source)
            ? (sourceType = source.ResourceType) is not null
            : (sourceType = null) is not null;

    internal static bool TryTraceAdParameterSource(
        ExpressionSyntax expression,
        SemanticModel semanticModel,
        CancellationToken cancellationToken,
        out TraceableAdSource source)
    {
        source = default;
        var current = expression;
        while (current is CastExpressionSyntax cast)
        {
            current = cast.Expression;
        }

        if (current is ParenthesizedExpressionSyntax parenthesized)
        {
            return TryTraceAdParameterSource(parenthesized.Expression, semanticModel, cancellationToken, out source);
        }

        if (current is ElementAccessExpressionSyntax elementAccess)
        {
            return TryCreateTraceableSource(elementAccess, semanticModel, cancellationToken, out source);
        }

        if (current is IdentifierNameSyntax identifier &&
            semanticModel.GetSymbolInfo(identifier, cancellationToken).Symbol is ILocalSymbol local)
        {
            return TryTraceLocalAdAlias(local, expression, semanticModel, cancellationToken, out source);
        }

        return false;
    }

    private static bool TryCreateTraceableSource(
        ElementAccessExpressionSyntax elementAccess,
        SemanticModel semanticModel,
        CancellationToken cancellationToken,
        out TraceableAdSource source)
    {
        source = default;
        var resourceType = semanticModel.GetTypeInfo(elementAccess.Expression, cancellationToken).Type;
        if (GetShaderResourceViewKind(resourceType) == ShaderResourceViewKind.None ||
            elementAccess.ArgumentList.Arguments.Count != 1)
        {
            return false;
        }

        var resourceName = elementAccess.Expression switch
        {
            IdentifierNameSyntax id => id.Identifier.ValueText,
            MemberAccessExpressionSyntax member => member.Name.Identifier.ValueText,
            _ => elementAccess.Expression.ToString()
        };
        var valueType = semanticModel.GetTypeInfo(elementAccess, cancellationToken).Type;
        var index = elementAccess.ArgumentList.Arguments[0].Expression.ToString().Trim();
        if (resourceName.Length == 0 || index.Length == 0)
        {
            return false;
        }

        source = new TraceableAdSource(
            resourceName,
            resourceType,
            valueType?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) ?? string.Empty,
            index);
        return true;
    }

    private static bool TryTraceLocalAdAlias(
        ILocalSymbol local,
        ExpressionSyntax markerArgument,
        SemanticModel semanticModel,
        CancellationToken cancellationToken,
        out TraceableAdSource source)
    {
        source = default;
        if (local.DeclaringSyntaxReferences.FirstOrDefault()?.GetSyntax(cancellationToken) is not VariableDeclaratorSyntax declarator ||
            declarator.Initializer?.Value is not { } initializer ||
            !TryTraceAdParameterSource(initializer, semanticModel, cancellationToken, out source))
        {
            return false;
        }

        var containingMethod = markerArgument.Ancestors().OfType<MethodDeclarationSyntax>().FirstOrDefault();
        if (containingMethod?.Body is null)
        {
            return false;
        }

        foreach (var assignment in containingMethod.Body.DescendantNodes().OfType<AssignmentExpressionSyntax>())
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (assignment.SpanStart <= declarator.SpanStart || assignment.SpanStart >= markerArgument.SpanStart)
            {
                continue;
            }

            if (semanticModel.GetOperation(assignment.Left, cancellationToken) is ILocalReferenceOperation assigned &&
                SymbolEqualityComparer.Default.Equals(assigned.Local, local))
            {
                return false;
            }
        }

        foreach (var prefixOrPostfix in containingMethod.Body.DescendantNodes().OfType<PostfixUnaryExpressionSyntax>()
                     .Cast<ExpressionSyntax>()
                     .Concat(containingMethod.Body.DescendantNodes().OfType<PrefixUnaryExpressionSyntax>()))
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (prefixOrPostfix.SpanStart <= declarator.SpanStart || prefixOrPostfix.SpanStart >= markerArgument.SpanStart)
            {
                continue;
            }

            if (semanticModel.GetOperation(prefixOrPostfix, cancellationToken) is IIncrementOrDecrementOperation inc &&
                inc.Target is ILocalReferenceOperation target &&
                SymbolEqualityComparer.Default.Equals(target.Local, local))
            {
                return false;
            }
        }

        return true;
    }

    private static bool IsSamplerStateType(ITypeSymbol? type)
        => type?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) == "global::Feather.Resources.SamplerState";

    private static uint? ReadBinding(IParameterSymbol parameter)
    {
        var attribute = parameter.GetAttributes().FirstOrDefault(attr => attr.AttributeClass?.ToDisplayString() == "Feather.BindingAttribute");
        if (attribute?.ConstructorArguments.Length == 1 && attribute.ConstructorArguments[0].Value is uint binding)
        {
            return binding;
        }

        return null;
    }

    private static bool IsSupportedPushConstantType(ITypeSymbol type)
    {
        if (type is INamedTypeSymbol gpuStruct
            && GpuStructLayoutRules.IsGpuStruct(gpuStruct)
            && (!GpuStructLayoutRules.IsValidGpuStructLayout(gpuStruct, requirePartial: true) || GpuStructLayoutRules.ContainsNarrowScalar(gpuStruct)))
        {
            return false;
        }

        if (type.SpecialType is SpecialType.System_Int32 or SpecialType.System_UInt32 or SpecialType.System_Single or SpecialType.System_Boolean)
        {
            return true;
        }

        var typeName = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        if (typeName is "global::Feather.Math.int2"
            or "global::Feather.Math.int3"
            or "global::Feather.Math.int4"
            or "global::Feather.Math.float2"
            or "global::Feather.Math.float3"
            or "global::Feather.Math.float4"
            or "global::Feather.Math.float2x2"
            or "global::Feather.Math.float3x3"
            or "global::Feather.Math.float4x4"
            or "global::Feather.Math.bool2"
            or "global::Feather.Math.bool3"
            or "global::Feather.Math.bool4")
        {
            return true;
        }

        return type.TypeKind == TypeKind.Enum
            || (type is INamedTypeSymbol named
                && named.IsUnmanagedType
                && GpuStructLayoutRules.IsGpuStruct(named)
                && GpuStructLayoutRules.IsValidGpuStructLayout(named, requirePartial: true));
    }

    private static bool IsSupportedComputeStorageElementType(ITypeSymbol type)
    {
        if (type.SpecialType is SpecialType.System_Byte
            or SpecialType.System_SByte
            or SpecialType.System_Int16
            or SpecialType.System_UInt16)
        {
            return false;
        }

        if (type.SpecialType is SpecialType.System_Int32
            or SpecialType.System_UInt32
            or SpecialType.System_Single
            or SpecialType.System_Boolean)
        {
            return true;
        }

        var typeName = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        if (typeName is "global::Feather.Math.int2"
            or "global::Feather.Math.int3"
            or "global::Feather.Math.int4"
            or "global::Feather.Math.float2"
            or "global::Feather.Math.float3"
            or "global::Feather.Math.float4"
            or "global::Feather.Math.float2x2"
            or "global::Feather.Math.float3x3"
            or "global::Feather.Math.float4x4"
            or "global::Feather.Math.bool2"
            or "global::Feather.Math.bool3"
            or "global::Feather.Math.bool4")
        {
            return true;
        }

        return type is INamedTypeSymbol named
            && named.IsUnmanagedType
            && GpuStructLayoutRules.IsGpuStruct(named)
            && GpuStructLayoutRules.IsValidGpuStructLayout(named, requirePartial: true)
            && !GpuStructLayoutRules.ContainsNarrowScalar(named);
    }

    private static bool IsSupportedComputeTextureElementType(ITypeSymbol type)
    {
        if (type is INamedTypeSymbol named && GpuStructLayoutRules.IsGpuStruct(named))
        {
            return named.IsUnmanagedType
                && GpuStructLayoutRules.IsValidGpuStructLayout(named, requirePartial: true);
        }

        return true;
    }

    private static bool IsSupportedComputeResourceElementType(ShaderResourceViewKind kind, ITypeSymbol type)
        => kind switch
        {
            ShaderResourceViewKind.Buffer => IsSupportedComputeStorageElementType(type),
            ShaderResourceViewKind.Texture2D or ShaderResourceViewKind.Texture3D => IsSupportedComputeTextureElementType(type),
            _ => true
        };

    private static bool IsSupportedComputeValueType(ITypeSymbol type)
    {
        if (type.SpecialType is SpecialType.System_Void)
        {
            return true;
        }

        if (type.SpecialType is SpecialType.System_Byte
            or SpecialType.System_SByte
            or SpecialType.System_Int16
            or SpecialType.System_UInt16)
        {
            return false;
        }

        if (type.SpecialType is SpecialType.System_Int32
            or SpecialType.System_UInt32
            or SpecialType.System_Single
            or SpecialType.System_Boolean)
        {
            return true;
        }

        var typeName = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        if (IsNonSquareMatrixTypeName(typeName))
        {
            return false;
        }

        if (typeName is "global::Feather.Math.int2"
            or "global::Feather.Math.int3"
            or "global::Feather.Math.int4"
            or "global::Feather.Math.float2"
            or "global::Feather.Math.float3"
            or "global::Feather.Math.float4"
            or "global::Feather.Math.float2x2"
            or "global::Feather.Math.float3x3"
            or "global::Feather.Math.float4x4"
            or "global::Feather.Math.bool2"
            or "global::Feather.Math.bool3"
            or "global::Feather.Math.bool4"
            or "global::Feather.Resources.SamplerState")
        {
            return true;
        }

        if (type is INamedTypeSymbol { IsGenericType: true, TypeArguments.Length: 1 } generic &&
            generic.ConstructedFrom.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) == "global::Feather.SharedMemory<T>")
        {
            return IsSupportedComputeStorageElementType(generic.TypeArguments[0]);
        }

        var resourceKind = GetShaderResourceViewKind(type);
        if (resourceKind != ShaderResourceViewKind.None)
        {
            return type is INamedTypeSymbol { TypeArguments.Length: 1 } resource &&
                   IsSupportedComputeResourceElementType(resourceKind, resource.TypeArguments[0]);
        }

        return type is INamedTypeSymbol named
            && named.IsUnmanagedType
            && GpuStructLayoutRules.IsGpuStruct(named)
            && GpuStructLayoutRules.IsValidGpuStructLayout(named, requirePartial: true)
            && !GpuStructLayoutRules.ContainsNarrowScalar(named);
    }

    private static bool IsNonSquareMatrixTypeName(string typeName)
        => typeName is "global::Feather.Math.float2x3"
            or "global::Feather.Math.float3x2"
            or "global::Feather.Math.float2x4"
            or "global::Feather.Math.float4x2"
            or "global::Feather.Math.float3x4"
            or "global::Feather.Math.float4x3";

    private enum ShaderResourceViewKind
    {
        None,
        Buffer,
        Texture2D,
        Texture3D
    }

    private sealed record CallableEdge(string TargetId, string TargetName, Location Location);
}
