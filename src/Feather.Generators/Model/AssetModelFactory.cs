using System.Collections.Immutable;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Feather.Generators.Model;

internal static class AssetModelFactory
{
    private const string AssetRootName = "Feather.Assets.Asset";
    private const string AssetInputAttributeName = "Feather.Assets.AssetInputAttribute";
    private const string AssetCapabilityAttributeMetadataName = "AssetCapabilityAttribute`1";
    private const string AssetOutputAttributeMetadataName = "AssetOutputAttribute`1";
    private const string CapabilityAttributeName = "Feather.Assets.FeatherAssetCapabilityAttribute";
    private const string OutputContractAttributeName = "Feather.Assets.FeatherAssetOutputContractAttribute";
    private const string CapabilityMarkerName = "Feather.Assets.IAssetCapabilityContract";
    private const string OutputMarkerName = "Feather.Assets.IAssetOutputContract";
    private const string RelativePathOption = "build_metadata.Compile.FeatherProjectRelativePath";

    public static AssetTypeModel? CreateType(
        GeneratorAttributeSyntaxContext context,
        CancellationToken cancellationToken)
    {
        if (context.TargetSymbol is not INamedTypeSymbol symbol ||
            context.TargetNode is not ClassDeclarationSyntax syntax)
        {
            return null;
        }

        var attribute = context.Attributes[0];
        var location = syntax.Identifier.GetLocation();
        var lineSpan = location.GetLineSpan();
        var inputs = ImmutableArray.CreateBuilder<AssetInputModel>();
        foreach (var member in symbol.GetMembers()
                     .Where(static member => member is IFieldSymbol or IPropertySymbol)
                     .OrderBy(static member => member.Locations.FirstOrDefault()?.SourceTree?.FilePath ?? string.Empty, StringComparer.Ordinal)
                     .ThenBy(static member => member.Locations.FirstOrDefault()?.SourceSpan.Start ?? int.MaxValue)
                     .ThenBy(static member => member.Name, StringComparer.Ordinal))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var input = FindAttribute(member.GetAttributes(), AssetInputAttributeName);
            if (input is null)
            {
                continue;
            }

            var memberType = member switch
            {
                IFieldSymbol field => field.Type,
                IPropertySymbol property => property.Type,
                _ => null,
            };
            if (memberType is not null)
            {
                inputs.Add(CreateInput(member, memberType, input));
            }
        }

        var capabilities = ImmutableArray.CreateBuilder<AssetCapabilityUseModel>();
        var outputs = ImmutableArray.CreateBuilder<AssetOutputModel>();
        foreach (var candidate in symbol.GetAttributes())
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (IsGenericAssetAttribute(candidate, AssetCapabilityAttributeMetadataName) &&
                candidate.AttributeClass is { TypeArguments.Length: 1 } capabilityClass)
            {
                var contract = capabilityClass.TypeArguments[0];
                var declaration = FindAttribute(contract.GetAttributes(), CapabilityAttributeName);
                capabilities.Add(new AssetCapabilityUseModel(
                    TypeName(contract),
                    declaration is null ? null : ConstructorString(declaration),
                    NamedUInt16(candidate, "MinimumMajor") ?? 1,
                    NamedUInt16(candidate, "MinimumMinor") ?? 0,
                    NamedBoolean(candidate, "Required", defaultValue: true),
                    candidate.ApplicationSyntaxReference?.GetSyntax(cancellationToken).GetLocation() ?? location));
            }
            else if (IsGenericAssetAttribute(candidate, AssetOutputAttributeMetadataName) &&
                     candidate.AttributeClass is { TypeArguments.Length: 1 } outputClass)
            {
                var contract = outputClass.TypeArguments[0];
                var declaration = FindAttribute(contract.GetAttributes(), OutputContractAttributeName);
                outputs.Add(new AssetOutputModel(
                    ConstructorString(candidate) ?? string.Empty,
                    TypeName(contract),
                    declaration is null ? null : ConstructorString(declaration),
                    declaration is null ? (ushort)0 : NamedUInt16(declaration, "ContractMajor") ?? 1,
                    declaration is null ? (ushort)0 : NamedUInt16(declaration, "ContractMinor") ?? 0,
                    NamedString(candidate, "Symbol") ?? string.Empty,
                    NamedString(candidate, "Name") ?? NamedString(candidate, "Symbol") ?? contract.Name,
                    NamedBoolean(candidate, "Required", defaultValue: true),
                    NamedBoolean(candidate, "GraphOutput", defaultValue: true),
                    NamedInt64(candidate, "PassDirections") ?? 1,
                    candidate.ApplicationSyntaxReference?.GetSyntax(cancellationToken).GetLocation() ?? location));
            }
        }

        var directBase = symbol.BaseType;
        string? baseTypeName = null;
        string? baseTypeGuid = null;
        if (directBase is not null && TypeName(directBase) != AssetRootName && DerivesFromAsset(directBase))
        {
            baseTypeName = TypeName(directBase);
            var baseAttribute = FindAttribute(
                directBase.GetAttributes(),
                "Feather.Assets.FeatherAssetTypeAttribute");
            baseTypeGuid = baseAttribute is null ? null : ConstructorString(baseAttribute);
        }

        return new AssetTypeModel(
            ConstructorString(attribute) ?? string.Empty,
            symbol.ContainingNamespace.IsGlobalNamespace ? string.Empty : symbol.ContainingNamespace.ToDisplayString(),
            symbol.Name,
            TypeName(symbol),
            NamedString(attribute, "Name") ?? symbol.Name,
            NamedString(attribute, "Description"),
            NamedUInt16(attribute, "ContractMajor") ?? 1,
            NamedUInt16(attribute, "ContractMinor") ?? 0,
            NamedInt32(attribute, "PayloadSchemaVersion") ?? 1,
            NamedBoolean(attribute, "Abstract", defaultValue: false),
            symbol.IsAbstract,
            symbol.ContainingType is null,
            symbol.TypeParameters.Length != 0,
            syntax.Modifiers.Any(SyntaxKind.PartialKeyword),
            DerivesFromAsset(symbol),
            baseTypeName,
            baseTypeGuid,
            lineSpan.Path,
            SourceHash(syntax.SyntaxTree, cancellationToken),
            lineSpan.StartLinePosition.Line,
            lineSpan.StartLinePosition.Character,
            inputs.ToImmutable(),
            capabilities.ToImmutable(),
            outputs.ToImmutable(),
            location);
    }

    public static AssetContractModel? CreateCapability(
        GeneratorAttributeSyntaxContext context,
        CancellationToken cancellationToken)
        => CreateContract(context, cancellationToken, AssetContractKind.Capability, CapabilityMarkerName);

    public static AssetContractModel? CreateOutputContract(
        GeneratorAttributeSyntaxContext context,
        CancellationToken cancellationToken)
        => CreateContract(context, cancellationToken, AssetContractKind.Output, OutputMarkerName);

    public static AssetProviderModel? CreateProvider(
        GeneratorAttributeSyntaxContext context,
        CancellationToken cancellationToken)
    {
        if (context.TargetSymbol is not INamedTypeSymbol symbol ||
            context.TargetNode is not ClassDeclarationSyntax syntax)
        {
            return null;
        }

        var attribute = context.Attributes[0];
        var location = syntax.Identifier.GetLocation();
        var lineSpan = location.GetLineSpan();
        var primaryOperations = ImmutableArray.CreateBuilder<string>();
        var assetTypes = ImmutableArray.CreateBuilder<string>();
        var outputs = ImmutableArray.CreateBuilder<string>();
        foreach (var contract in symbol.AllInterfaces)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var implementedOperation = ProviderOperation(contract.OriginalDefinition);
            if (implementedOperation is null)
            {
                continue;
            }

            primaryOperations.Add(implementedOperation);
            foreach (var typeArgument in contract.TypeArguments)
            {
                if (DerivesFromAsset(typeArgument))
                {
                    assetTypes.Add(TypeName(typeArgument));
                }
                else if (Implements(typeArgument, OutputMarkerName))
                {
                    outputs.Add(TypeName(typeArgument));
                }
            }
        }

        var operationValue = attribute.ConstructorArguments.Length > 1 &&
                             attribute.ConstructorArguments[1].Value is int declaredOperation
            ? declaredOperation
            : -1;
        return new AssetProviderModel(
            ConstructorString(attribute) ?? string.Empty,
            TypeName(symbol),
            NamedString(attribute, "Name") ?? symbol.Name,
            ProviderOperation(operationValue),
            NamedUInt16(attribute, "ContractMajor") ?? 1,
            NamedUInt16(attribute, "ContractMinor") ?? 0,
            EnumName(attribute, "Owner") ?? "ASSET_SERVICE",
            EnumName(attribute, "Determinism") ?? "DETERMINISTIC",
            primaryOperations.ToImmutable(),
            assetTypes.Distinct(StringComparer.Ordinal).OrderBy(static value => value, StringComparer.Ordinal).ToImmutableArray(),
            outputs.Distinct(StringComparer.Ordinal).OrderBy(static value => value, StringComparer.Ordinal).ToImmutableArray(),
            lineSpan.Path,
            SourceHash(syntax.SyntaxTree, cancellationToken),
            lineSpan.StartLinePosition.Line,
            lineSpan.StartLinePosition.Character,
            location);
    }

    public static AssetTypeModel ApplyProjectRelativePath(
        AssetTypeModel model,
        AnalyzerConfigOptionsProvider optionsProvider)
        => model with { SourcePath = RelativePath(model.Location, model.SourcePath, optionsProvider) };

    public static AssetContractModel ApplyProjectRelativePath(
        AssetContractModel model,
        AnalyzerConfigOptionsProvider optionsProvider)
        => model with { SourcePath = RelativePath(model.Location, model.SourcePath, optionsProvider) };

    public static AssetProviderModel ApplyProjectRelativePath(
        AssetProviderModel model,
        AnalyzerConfigOptionsProvider optionsProvider)
        => model with { SourcePath = RelativePath(model.Location, model.SourcePath, optionsProvider) };

    private static AssetContractModel? CreateContract(
        GeneratorAttributeSyntaxContext context,
        CancellationToken cancellationToken,
        AssetContractKind kind,
        string markerName)
    {
        if (context.TargetSymbol is not INamedTypeSymbol symbol || context.TargetNode is not TypeDeclarationSyntax syntax)
        {
            return null;
        }

        cancellationToken.ThrowIfCancellationRequested();
        var attribute = context.Attributes[0];
        var location = syntax.Identifier.GetLocation();
        var lineSpan = location.GetLineSpan();
        return new AssetContractModel(
            kind,
            ConstructorString(attribute) ?? string.Empty,
            TypeName(symbol),
            NamedString(attribute, "Name") ?? symbol.Name,
            NamedUInt16(attribute, "ContractMajor") ?? 1,
            NamedUInt16(attribute, "ContractMinor") ?? 0,
            symbol.ContainingType is null,
            symbol.TypeParameters.Length != 0,
            Implements(symbol, markerName),
            lineSpan.Path,
            SourceHash(syntax.SyntaxTree, cancellationToken),
            lineSpan.StartLinePosition.Line,
            lineSpan.StartLinePosition.Character,
            location);
    }

    private static string SourceHash(SyntaxTree syntaxTree, CancellationToken cancellationToken)
    {
        string content = syntaxTree.GetText(cancellationToken).ToString();
        using SHA256 sha256 = SHA256.Create();
        byte[] hash = sha256.ComputeHash(Encoding.UTF8.GetBytes(content));
        return "sha256:" + BitConverter.ToString(hash).Replace("-", string.Empty).ToLowerInvariant();
    }

    private static AssetInputModel CreateInput(ISymbol member, ITypeSymbol type, AttributeData attribute)
    {
        var valueKind = ValueKind(
            type,
            out var referencedAssetType,
            out var referencedAssetTypeGuid);
        var immutable = member switch
        {
            IPropertySymbol property => property.SetMethod is null || property.SetMethod.IsInitOnly,
            IFieldSymbol field => field.IsReadOnly,
            _ => false,
        };
        return new AssetInputModel(
            ConstructorString(attribute) ?? string.Empty,
            member.Name,
            NamedString(attribute, "Name") ?? member.Name,
            NamedString(attribute, "Group"),
            NamedInt32(attribute, "Order") ?? 0,
            NamedBoolean(attribute, "Required", defaultValue: true),
            NamedInt64(attribute, "Role") ?? 1,
            EnumName(attribute, "ChangeImpact") ?? "REEVALUATE_OUTPUTS",
            valueKind,
            TypeName(type),
            referencedAssetType,
            referencedAssetTypeGuid,
            FiniteDouble(attribute, "Min"),
            FiniteDouble(attribute, "Max"),
            FiniteDouble(attribute, "Step"),
            NonNegative(attribute, "MinItems"),
            NonNegative(attribute, "MaxItems"),
            NonNegative(attribute, "MaxLength"),
            immutable,
            member.Locations.FirstOrDefault() ?? Location.None);
    }

    private static string ValueKind(
        ITypeSymbol type,
        out string? referencedAssetType,
        out string? referencedAssetTypeGuid)
    {
        referencedAssetType = null;
        referencedAssetTypeGuid = null;
        if (type is INamedTypeSymbol { IsGenericType: true } named &&
            named.OriginalDefinition.ToDisplayString(SymbolDisplayFormat.CSharpErrorMessageFormat) ==
            "Feather.Assets.AssetRef<TAsset>" &&
            named.TypeArguments.Length == 1 &&
            DerivesFromAsset(named.TypeArguments[0]))
        {
            ITypeSymbol assetType = named.TypeArguments[0];
            referencedAssetType = TypeName(assetType);
            AttributeData? assetTypeAttribute = FindAttribute(
                assetType.GetAttributes(),
                "Feather.Assets.FeatherAssetTypeAttribute");
            referencedAssetTypeGuid = assetTypeAttribute is null
                ? null
                : ConstructorString(assetTypeAttribute);
            return "ASSET_REFERENCE";
        }

        return TypeName(type) switch
        {
            "bool" => "BOOLEAN",
            "int" or "uint" => "INTEGER",
            "float" or "double" => "FLOAT",
            "string" => "TEXT",
            "Feather.Math.float2" => "VECTOR2",
            "Feather.Math.float3" => "VECTOR3",
            "Feather.Math.float4" => "VECTOR4",
            _ => "UNSUPPORTED",
        };
    }

    private static bool DerivesFromAsset(ITypeSymbol type)
    {
        for (var current = type as INamedTypeSymbol; current is not null; current = current.BaseType)
        {
            if (TypeName(current) == AssetRootName)
            {
                return true;
            }
        }

        return false;
    }

    private static bool Implements(ITypeSymbol type, string interfaceName)
        => type is INamedTypeSymbol named && named.AllInterfaces.Any(item => TypeName(item) == interfaceName) ||
           TypeName(type) == interfaceName;

    private static string? ProviderOperation(INamedTypeSymbol definition)
        => definition.ToDisplayString(SymbolDisplayFormat.CSharpErrorMessageFormat) switch
        {
            "Feather.Assets.IAssetCreator<TAsset>" => "CREATE",
            "Feather.Assets.IAssetImporter<TAsset>" => "IMPORT",
            "Feather.Assets.IAssetTransformer<TSource, TDestination>" => "TRANSFORM",
            "Feather.Assets.IAssetBuilder<TAsset>" => "BUILD",
            "Feather.Assets.IAssetPreviewProvider<TAsset>" => "PREVIEW",
            "Feather.Assets.IAssetRuntimeAdapter<TOutput>" => "RUNTIME_ADAPTER",
            _ => null,
        };

    private static string ProviderOperation(int value)
        => value switch
        {
            0 => "CREATE",
            1 => "IMPORT",
            2 => "TRANSFORM",
            3 => "BUILD",
            4 => "PREVIEW",
            5 => "RUNTIME_ADAPTER",
            _ => "INVALID",
        };

    private static string RelativePath(
        Location location,
        string fallback,
        AnalyzerConfigOptionsProvider optionsProvider)
    {
        var sourceTree = location.SourceTree;
        if (sourceTree is not null &&
            optionsProvider.GetOptions(sourceTree).TryGetValue(RelativePathOption, out var relativePath) &&
            !string.IsNullOrWhiteSpace(relativePath))
        {
            return relativePath.Replace('\\', '/');
        }

        return fallback.Replace('\\', '/');
    }

    private static bool IsGenericAssetAttribute(AttributeData attribute, string metadataName)
        => attribute.AttributeClass?.OriginalDefinition.MetadataName == metadataName &&
           attribute.AttributeClass.ContainingNamespace.ToDisplayString() == "Feather.Assets";

    private static AttributeData? FindAttribute(ImmutableArray<AttributeData> attributes, string typeName)
        => attributes.FirstOrDefault(attribute => TypeName(attribute.AttributeClass) == typeName);

    private static string TypeName(ITypeSymbol? type)
        => type?.ToDisplayString(SymbolDisplayFormat.CSharpErrorMessageFormat) ?? string.Empty;

    private static string? ConstructorString(AttributeData attribute)
        => attribute.ConstructorArguments.Length > 0 ? attribute.ConstructorArguments[0].Value as string : null;

    private static string? NamedString(AttributeData attribute, string name)
        => NamedArgument(attribute, name, out var value) ? value.Value as string : null;

    private static int? NamedInt32(AttributeData attribute, string name)
        => NamedArgument(attribute, name, out var value) && value.Value is int result ? result : null;

    private static long? NamedInt64(AttributeData attribute, string name)
        => NamedArgument(attribute, name, out var value) && value.Value is not null
            ? Convert.ToInt64(value.Value, CultureInfo.InvariantCulture)
            : null;

    private static ushort? NamedUInt16(AttributeData attribute, string name)
        => NamedArgument(attribute, name, out var value) && value.Value is ushort result ? result : null;

    private static bool NamedBoolean(AttributeData attribute, string name, bool defaultValue)
        => NamedArgument(attribute, name, out var value) && value.Value is bool result ? result : defaultValue;

    private static double? FiniteDouble(AttributeData attribute, string name)
    {
        if (!NamedArgument(attribute, name, out var value) || value.Value is null)
        {
            return null;
        }

        var result = Convert.ToDouble(value.Value, CultureInfo.InvariantCulture);
        return !double.IsNaN(result) && !double.IsInfinity(result) ? result : null;
    }

    private static int? NonNegative(AttributeData attribute, string name)
        => NamedInt32(attribute, name) is >= 0 and var value ? value : null;

    private static string? EnumName(AttributeData attribute, string name)
    {
        if (!NamedArgument(attribute, name, out var value) ||
            value.Type is not INamedTypeSymbol enumType ||
            value.Value is null)
        {
            return null;
        }

        var member = enumType.GetMembers().OfType<IFieldSymbol>()
            .FirstOrDefault(field => field.HasConstantValue && Equals(field.ConstantValue, value.Value));
        return member?.Name switch
        {
            "MetadataOnly" => "METADATA_ONLY",
            "PreviewOnly" => "PREVIEW_ONLY",
            "ReevaluateOutputs" => "REEVALUATE_OUTPUTS",
            "RuntimeCandidate" => "RUNTIME_CANDIDATE",
            "AssetService" => "ASSET_SERVICE",
            "IsolatedWorker" => "ISOLATED_WORKER",
            "RenderHost" => "RENDER_HOST",
            "Deterministic" => "DETERMINISTIC",
            "Seeded" => "SEEDED",
            "EnvironmentDependent" => "ENVIRONMENT_DEPENDENT",
            "NonDeterministic" => "NON_DETERMINISTIC",
            _ => member?.Name.ToUpperInvariant(),
        };
    }

    private static bool NamedArgument(AttributeData attribute, string name, out TypedConstant value)
    {
        foreach (var argument in attribute.NamedArguments)
        {
            if (argument.Key == name)
            {
                value = argument.Value;
                return true;
            }
        }

        value = default;
        return false;
    }
}
