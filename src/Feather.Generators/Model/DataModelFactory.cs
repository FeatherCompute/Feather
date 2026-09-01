using System.Collections.Immutable;
using System.Security.Cryptography;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Feather.Generators.Model;

internal static class DataModelFactory
{
    private const string DataResourceAttributeName = "Feather.RenderGraph.DataResourceAttribute";
    private const string RelativePathOption = "build_metadata.Compile.FeatherProjectRelativePath";

    public static DataTypeModel? Create(
        GeneratorAttributeSyntaxContext context,
        CancellationToken cancellationToken)
    {
        if (context.TargetSymbol is not INamedTypeSymbol symbol ||
            context.TargetNode is not TypeDeclarationSyntax syntax)
        {
            return null;
        }

        var attribute = context.Attributes[0];
        var location = syntax.Identifier.GetLocation();
        var lineSpan = location.GetLineSpan();
        var resources = ImmutableArray.CreateBuilder<DataResourceModel>();
        foreach (var member in symbol.GetMembers()
                     .Where(static candidate => candidate is IFieldSymbol or IPropertySymbol)
                     .OrderBy(static candidate => candidate.Locations.FirstOrDefault()?.SourceTree?.FilePath ?? string.Empty, StringComparer.Ordinal)
                     .ThenBy(static candidate => candidate.Locations.FirstOrDefault()?.SourceSpan.Start ?? int.MaxValue)
                     .ThenBy(static candidate => candidate.Name, StringComparer.Ordinal))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var declaration = member.GetAttributes().FirstOrDefault(static candidate =>
                candidate.AttributeClass?.ToDisplayString() == DataResourceAttributeName);
            if (declaration is null)
            {
                continue;
            }

            var memberType = member switch
            {
                IFieldSymbol field => field.Type,
                IPropertySymbol property => property.Type,
                _ => null,
            };
            if (memberType is not INamedTypeSymbol { TypeArguments.Length: 1 } resourceType)
            {
                resources.Add(InvalidResource(member, declaration));
                continue;
            }

            var kind = ResourceKind(resourceType.OriginalDefinition);
            var elementType = resourceType.TypeArguments[0];
            var layout = GpuStructLayoutRules.GetTypeLayout(elementType);
            resources.Add(new DataResourceModel(
                ConstructorString(declaration) ?? string.Empty,
                member.Name,
                NamedString(declaration, "Name") ?? member.Name,
                kind,
                NamedInt32(declaration, "Access") ?? 3,
                TypeName(elementType),
                layout.SizeInBytes,
                layout.Alignment,
                ElementLayoutIdentity(elementType, layout, cancellationToken),
                NamedInt32(declaration, "Creation") ?? 0,
                NamedInt32(declaration, "Update") ?? 2,
                NamedInt32(declaration, "Lifetime") ?? 1,
                NamedInt32(declaration, "Frames") ?? 0,
                NamedInt64(declaration, "MaximumBytes") ?? 0,
                NamedInt64(declaration, "ElementCount") ?? 0,
                NamedInt32(declaration, "Width") ?? 0,
                NamedInt32(declaration, "Height") ?? 0,
                NamedInt32(declaration, "Depth") ?? 0,
                NamedString(declaration, "Format"),
                member.Locations.FirstOrDefault() ?? location));
        }

        return new DataTypeModel(
            ConstructorString(attribute) ?? string.Empty,
            TypeName(symbol),
            NamedString(attribute, "Name") ?? symbol.Name,
            NamedUInt16(attribute, "ContractMajor") ?? 1,
            NamedUInt16(attribute, "ContractMinor") ?? 0,
            symbol.ContainingType is null,
            symbol.TypeParameters.Length != 0,
            lineSpan.Path,
            SourceHash(syntax.SyntaxTree, cancellationToken),
            lineSpan.StartLinePosition.Line,
            lineSpan.StartLinePosition.Character,
            resources.ToImmutable(),
            location);
    }

    public static DataTypeModel ApplyProjectRelativePath(
        DataTypeModel model,
        AnalyzerConfigOptionsProvider optionsProvider)
    {
        if (model.Location.SourceTree is null)
        {
            return model;
        }

        var options = optionsProvider.GetOptions(model.Location.SourceTree);
        return options.TryGetValue(RelativePathOption, out var relative) &&
               !string.IsNullOrWhiteSpace(relative)
            ? model with { SourcePath = relative.Replace('\\', '/') }
            : model;
    }

    private static DataResourceModel InvalidResource(ISymbol member, AttributeData declaration)
        => new(
            ConstructorString(declaration) ?? string.Empty,
            member.Name,
            NamedString(declaration, "Name") ?? member.Name,
            "UNSUPPORTED",
            NamedInt32(declaration, "Access") ?? 3,
            string.Empty,
            0,
            0,
            string.Empty,
            NamedInt32(declaration, "Creation") ?? 0,
            NamedInt32(declaration, "Update") ?? 2,
            NamedInt32(declaration, "Lifetime") ?? 1,
            NamedInt32(declaration, "Frames") ?? 0,
            NamedInt64(declaration, "MaximumBytes") ?? 0,
            NamedInt64(declaration, "ElementCount") ?? 0,
            NamedInt32(declaration, "Width") ?? 0,
            NamedInt32(declaration, "Height") ?? 0,
            NamedInt32(declaration, "Depth") ?? 0,
            NamedString(declaration, "Format"),
            member.Locations.FirstOrDefault() ?? Location.None);

    private static string ResourceKind(INamedTypeSymbol definition)
        => definition.ToDisplayString() switch
        {
            "Feather.RenderGraph.DataBuffer<T>" => "BUFFER",
            "Feather.RenderGraph.DataTexture1D<T>" => "TEXTURE_1D",
            "Feather.RenderGraph.DataTexture2D<T>" => "TEXTURE_2D",
            "Feather.RenderGraph.DataTexture3D<T>" => "TEXTURE_3D",
            "Feather.RenderGraph.DataProbeVolume<T>" => "PROBE_VOLUME",
            "Feather.RenderGraph.DataRadianceCascade<T>" => "RADIANCE_CASCADE",
            "Feather.RenderGraph.DataCustom<T>" => "CUSTOM",
            _ => "UNSUPPORTED",
        };

    private static string ElementLayoutIdentity(
        ITypeSymbol elementType,
        GpuStructTypeLayout layout,
        CancellationToken cancellationToken)
    {
        var builder = new StringBuilder()
            .Append(TypeName(elementType)).Append('|')
            .Append(layout.SizeInBytes).Append('|')
            .Append(layout.Alignment);
        if (elementType is INamedTypeSymbol named && GpuStructLayoutRules.IsGpuStruct(named))
        {
            foreach (var field in GpuStructFieldDiscovery.GetFields(named))
            {
                cancellationToken.ThrowIfCancellationRequested();
                var fieldLayout = GpuStructLayoutRules.GetTypeLayout(field.Type);
                builder.Append('|').Append(field.Name).Append(':')
                    .Append(TypeName(field.Type)).Append(':')
                    .Append(fieldLayout.SizeInBytes).Append(':')
                    .Append(fieldLayout.Alignment);
            }
        }
        return builder.ToString();
    }

    private static string TypeName(ITypeSymbol type)
        => type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)
            .Replace("global::", string.Empty);

    private static string? ConstructorString(AttributeData attribute)
        => attribute.ConstructorArguments.Length > 0
            ? attribute.ConstructorArguments[0].Value as string
            : null;

    private static string? NamedString(AttributeData attribute, string name)
        => attribute.NamedArguments.FirstOrDefault(pair => pair.Key == name).Value.Value as string;

    private static int? NamedInt32(AttributeData attribute, string name)
        => attribute.NamedArguments.FirstOrDefault(pair => pair.Key == name).Value.Value is int value
            ? value
            : null;

    private static long? NamedInt64(AttributeData attribute, string name)
        => attribute.NamedArguments.FirstOrDefault(pair => pair.Key == name).Value.Value switch
        {
            long value => value,
            int value => value,
            _ => null,
        };

    private static ushort? NamedUInt16(AttributeData attribute, string name)
        => attribute.NamedArguments.FirstOrDefault(pair => pair.Key == name).Value.Value switch
        {
            ushort value => value,
            int value when value is >= ushort.MinValue and <= ushort.MaxValue => (ushort)value,
            _ => null,
        };

    private static string SourceHash(SyntaxTree syntaxTree, CancellationToken cancellationToken)
    {
        string content = syntaxTree.GetText(cancellationToken).ToString();
        using SHA256 sha256 = SHA256.Create();
        byte[] hash = sha256.ComputeHash(Encoding.UTF8.GetBytes(content));
        return "sha256:" + BitConverter.ToString(hash).Replace("-", string.Empty).ToLowerInvariant();
    }
}

