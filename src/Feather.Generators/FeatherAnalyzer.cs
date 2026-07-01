using System.Collections.Immutable;
using Feather.Generators.Diagnostics;
using Feather.Generators.Model;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Feather.Generators;

[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class FeatherAnalyzer : DiagnosticAnalyzer
{
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics
        => ImmutableArray.Create(FeatherDiagnostics.All);

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterSyntaxNodeAction(AnalyzeStruct, Microsoft.CodeAnalysis.CSharp.SyntaxKind.StructDeclaration);
        context.RegisterSyntaxNodeAction(AnalyzeInvocation, SyntaxKind.InvocationExpression);
    }

    private static void AnalyzeInvocation(SyntaxNodeAnalysisContext context)
    {
        var syntax = (InvocationExpressionSyntax)context.Node;
        if (context.SemanticModel.GetSymbolInfo(syntax, context.CancellationToken).Symbol is not IMethodSymbol method ||
            method.ContainingType?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) != "global::Feather.AD.AD" ||
            method.Name is not ("Parameter" or "Loss"))
        {
            return;
        }

        var containingStruct = syntax.Ancestors().OfType<StructDeclarationSyntax>().FirstOrDefault();
        if (containingStruct is not null && HasShaderAttribute(containingStruct))
        {
            return;
        }

        context.ReportDiagnostic(Diagnostic.Create(
            FeatherDiagnostics.AutoDiffMarkerUnsupported,
            syntax.GetLocation(),
            method.Name,
            "markers may only be used inside Feather-generated GPU kernels"));
    }

    private static void AnalyzeStruct(SyntaxNodeAnalysisContext context)
    {
        var syntax = (StructDeclarationSyntax)context.Node;
        if (!HasShaderAttribute(syntax))
        {
            return;
        }

        if (context.SemanticModel.GetDeclaredSymbol(syntax, context.CancellationToken) is not INamedTypeSymbol symbol)
        {
            return;
        }

        var model = ShaderModelFactory.Create(syntax, symbol, context.SemanticModel, context.CancellationToken);
        if (model is null)
        {
            context.ReportDiagnostic(Diagnostic.Create(FeatherDiagnostics.ShaderInterface, syntax.Identifier.GetLocation(), syntax.Identifier.ValueText));
            return;
        }

        foreach (var diagnostic in ShaderModelFactory.Validate(model))
        {
            context.ReportDiagnostic(diagnostic);
        }
    }

    private static bool HasShaderAttribute(StructDeclarationSyntax syntax)
    {
        foreach (var list in syntax.AttributeLists)
        {
            foreach (var attribute in list.Attributes)
            {
                var name = attribute.Name.ToString();
                if (name is "Kernel" or "KernelAttribute" or "VertexShader" or "VertexShaderAttribute" or "FragmentShader" or "FragmentShaderAttribute" or "AutoDiff" or "AutoDiffAttribute")
                {
                    return true;
                }
            }
        }

        return false;
    }
}
