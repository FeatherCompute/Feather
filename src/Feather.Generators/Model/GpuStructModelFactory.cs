using Feather.Generators.Diagnostics;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Feather.Generators.Model;

internal static class GpuStructModelFactory
{
    public static GpuStructModel? Create(GeneratorAttributeSyntaxContext context, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (context.TargetNode is not TypeDeclarationSyntax syntax || context.TargetSymbol is not INamedTypeSymbol symbol)
        {
            return null;
        }

        return Create(syntax, symbol);
    }

    public static GpuStructModel Create(TypeDeclarationSyntax syntax, INamedTypeSymbol symbol)
    {
        var namespaceName = symbol.ContainingNamespace.IsGlobalNamespace ? string.Empty : symbol.ContainingNamespace.ToDisplayString();
        var fields = CreateFields(symbol);
        var alignment = fields.Count == 0 ? 1 : fields.Max(static field => field.Alignment);
        var size = fields.Count == 0 ? 0 : GpuStructLayoutRules.Align(fields.Max(static field => field.Offset + field.SizeInBytes), alignment);
        return new GpuStructModel(
            syntax,
            symbol,
            namespaceName,
            symbol.Name,
            symbol.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat),
            symbol.IsRecord,
            "global::Feather.GpuLayout.Std430",
            new EquatableArray<GpuStructFieldModel>(fields),
            size,
            alignment,
            RequiresRepacking: true);
    }

    public static IEnumerable<Diagnostic> Validate(GpuStructModel model)
    {
        foreach (var error in GpuStructLayoutRules.GetLayoutErrors(model.Symbol, requirePartial: true))
        {
            if (error.IsMatrixError)
            {
                yield return Diagnostic.Create(
                    FeatherDiagnostics.MatrixLayoutUnsupported,
                    error.Symbol.Locations.FirstOrDefault() ?? model.Syntax.Identifier.GetLocation(),
                    error.Message);
                continue;
            }

            yield return Diagnostic.Create(
                FeatherDiagnostics.StructLayoutUnsupported,
                error.Symbol.Locations.FirstOrDefault() ?? model.Syntax.Identifier.GetLocation(),
                error.StructSymbol.Name,
                error.Message);
        }
    }

    private static IReadOnlyList<GpuStructFieldModel> CreateFields(INamedTypeSymbol symbol)
    {
        var fields = new List<GpuStructFieldModel>();
        var offset = 0;

        foreach (var member in GpuStructFieldDiscovery.GetFields(symbol))
        {
            var field = TryCreateField(member, offset);
            if (field is null)
            {
                continue;
            }

            offset = field.Offset + field.SizeInBytes;
            fields.Add(field);
        }

        return fields;
    }

    private static GpuStructFieldModel? TryCreateField(GpuStructMember member, int currentOffset)
    {
        var layout = GpuStructLayoutRules.GetTypeLayout(member.Type);
        var offset = GpuStructLayoutRules.Align(currentOffset, layout.Alignment);
        return new GpuStructFieldModel(
            member.Name,
            member.Type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat),
            layout.ArrayElementTypeName,
            offset,
            layout.SizeInBytes,
            layout.Alignment,
            member.MemberAccessor,
            layout.ArrayLength,
            layout.ArrayStride,
            member.IsRecordPrimaryConstructorProperty,
            member.IsReadonlyExplicitField,
            GpuStructLayoutRules.UsesUnsupportedMatrixType(member.Type));
    }
}
