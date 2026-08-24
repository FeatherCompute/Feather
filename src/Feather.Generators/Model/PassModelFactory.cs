using System.Collections.Immutable;
using System.Globalization;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Feather.Generators.Model;

internal static class PassModelFactory
{
    private const string InputAttributeName = "Feather.RenderGraph.InputAttribute";
    private const string OutputAttributeName = "Feather.RenderGraph.OutputAttribute";
    private const string ParameterAttributeName = "Feather.RenderGraph.ParameterAttribute";
    private const string FeatherEnumAttributeName = "Feather.RenderGraph.FeatherEnumAttribute";
    private const string FeatherEnumMemberAttributeName = "Feather.RenderGraph.FeatherEnumMemberAttribute";
    private const string FlagsAttributeName = "System.FlagsAttribute";
    private const string RenderPassInterfaceName = "Feather.RenderGraph.IRenderPass";
    private const string AssetTypeAttributeName = "Feather.Assets.FeatherAssetTypeAttribute";
    private const string AssetCapabilityAttributeName = "Feather.Assets.FeatherAssetCapabilityAttribute";
    private const string AssetOutputContractAttributeName = "Feather.Assets.FeatherAssetOutputContractAttribute";
    private const string AssetCapabilityAttributeMetadataName = "AssetCapabilityAttribute`1";
    private const string AssetOutputAttributeMetadataName = "AssetOutputAttribute`1";
    private const string AssetProductBindingAttributeName = "Feather.RenderGraph.AssetProductBindingAttribute";
    private const string RelativePathOption = "build_metadata.Compile.FeatherProjectRelativePath";

    public static PassModel? Create(GeneratorAttributeSyntaxContext context, CancellationToken cancellationToken)
    {
        if (context.TargetSymbol is not INamedTypeSymbol symbol ||
            context.TargetNode is not ClassDeclarationSyntax syntax)
        {
            return null;
        }

        var passAttribute = context.Attributes[0];
        var guid = GetConstructorString(passAttribute);
        if (guid is null)
        {
            return null;
        }

        var sourceLocation = syntax.Identifier.GetLocation();
        var lineSpan = sourceLocation.GetLineSpan();
        var inputs = ImmutableArray.CreateBuilder<PassSocketModel>();
        var outputs = ImmutableArray.CreateBuilder<PassSocketModel>();
        var parameters = ImmutableArray.CreateBuilder<PassParameterModel>();

        foreach (var member in symbol.GetMembers()
                     .Where(static member => member is IFieldSymbol or IPropertySymbol)
                     .OrderBy(
                         static member => member.Locations.FirstOrDefault()?.SourceTree?.FilePath ?? string.Empty,
                         StringComparer.Ordinal)
                     .ThenBy(static member => member.Locations.FirstOrDefault()?.SourceSpan.Start ?? int.MaxValue)
                     .ThenBy(static member => member.Name, StringComparer.Ordinal))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var memberType = member switch
            {
                IFieldSymbol field => field.Type,
                IPropertySymbol property => property.Type,
                _ => null
            };
            if (memberType is null)
            {
                continue;
            }

            var attributes = member.GetAttributes();
            var input = FindAttribute(attributes, InputAttributeName);
            var output = FindAttribute(attributes, OutputAttributeName);
            var parameter = FindAttribute(attributes, ParameterAttributeName);
            if (input is not null)
            {
                inputs.Add(CreateSocket(member, memberType, input, "Read"));
            }
            if (output is not null)
            {
                outputs.Add(CreateSocket(member, memberType, output, "Write"));
            }
            if (parameter is not null)
            {
                parameters.Add(CreateParameter(
                    context.SemanticModel.Compilation,
                    member,
                    memberType,
                    parameter,
                    cancellationToken));
            }
        }

        var displayName = GetNamedString(passAttribute, "Name") ?? symbol.Name;
        var category = GetNamedString(passAttribute, "Category") ?? "Uncategorized";
        var version = GetNamedInt32(passAttribute, "Version") ?? 1;
        var implementsRenderPass = symbol.AllInterfaces.Any(static item =>
            item.ToDisplayString(SymbolDisplayFormat.CSharpErrorMessageFormat) == RenderPassInterfaceName);

        return new PassModel(
            guid,
            symbol.ToDisplayString(SymbolDisplayFormat.CSharpErrorMessageFormat),
            displayName,
            category,
            version,
            lineSpan.Path,
            lineSpan.StartLinePosition.Line,
            lineSpan.StartLinePosition.Character,
            implementsRenderPass,
            inputs.ToImmutable(),
            outputs.ToImmutable(),
            parameters.ToImmutable(),
            sourceLocation);
    }

    public static PassModel ApplyProjectRelativePath(
        PassModel model,
        AnalyzerConfigOptionsProvider optionsProvider)
    {
        var sourceTree = model.Location.SourceTree;
        if (sourceTree is not null &&
            optionsProvider.GetOptions(sourceTree).TryGetValue(RelativePathOption, out var relativePath) &&
            !string.IsNullOrWhiteSpace(relativePath))
        {
            return model with { SourcePath = relativePath.Replace('\\', '/') };
        }

        return model;
    }

    private static PassSocketModel CreateSocket(
        ISymbol member,
        ITypeSymbol memberType,
        AttributeData attribute,
        string access)
    {
        var contract = CreateSocketContract(member, memberType);
        return new PassSocketModel(
            GetConstructorString(attribute) ?? string.Empty,
            GetNamedString(attribute, "Name") ?? member.Name,
            ResourceKind(memberType),
            BufferElementType(memberType),
            GetTextureFormat(attribute),
            access,
            contract,
            member.Locations.FirstOrDefault() ?? Location.None);
    }

    private static PassSocketContractModel CreateSocketContract(ISymbol member, ITypeSymbol memberType)
    {
        if (memberType is INamedTypeSymbol { IsGenericType: true } named &&
            named.ConstructedFrom.ToDisplayString(SymbolDisplayFormat.CSharpErrorMessageFormat) ==
            "Feather.Assets.AssetRef<TAsset>")
        {
            var assetType = named.TypeArguments[0];
            var typeAttribute = FindAttribute(assetType.GetAttributes(), AssetTypeAttributeName);
            return new PassSocketContractModel(
                "ASSET_REFERENCE",
                typeAttribute is null ? null : GetConstructorString(typeAttribute),
                typeAttribute is null ? (ushort)0 : GetNamedUInt16(typeAttribute, "ContractMajor") ?? 1,
                typeAttribute is null ? (ushort)0 : GetNamedUInt16(typeAttribute, "ContractMinor") ?? 0,
                AssetCapabilities(assetType),
                null,
                null,
                0,
                0,
                AdapterRequired: false);
        }

        if (memberType is INamedTypeSymbol { IsGenericType: true } product &&
            product.ConstructedFrom.ToDisplayString(SymbolDisplayFormat.CSharpErrorMessageFormat) ==
            "Feather.RenderGraph.AssetOutputHandle<TOutput>")
        {
            var outputType = product.TypeArguments[0];
            var binding = FindAttribute(member.GetAttributes(), AssetProductBindingAttributeName);
            var assetType = binding?.ConstructorArguments.Length > 0 &&
                            binding.ConstructorArguments[0].Value is ITypeSymbol boundType
                ? boundType
                : null;
            var slotGuid = binding?.ConstructorArguments.Length > 1
                ? binding.ConstructorArguments[1].Value as string
                : null;
            var typeAttribute = assetType is null
                ? null
                : FindAttribute(assetType.GetAttributes(), AssetTypeAttributeName);
            var outputContract = FindAttribute(outputType.GetAttributes(), AssetOutputContractAttributeName);
            var outputSlot = assetType is null
                ? null
                : FindAssetOutput(assetType, outputType, slotGuid);
            return new PassSocketContractModel(
                "ASSET_PRODUCT",
                typeAttribute is null ? null : GetConstructorString(typeAttribute),
                typeAttribute is null ? (ushort)0 : GetNamedUInt16(typeAttribute, "ContractMajor") ?? 1,
                typeAttribute is null ? (ushort)0 : GetNamedUInt16(typeAttribute, "ContractMinor") ?? 0,
                assetType is null ? [] : AssetCapabilities(assetType),
                outputSlot is null ? null : GetConstructorString(outputSlot),
                outputContract is null ? null : GetConstructorString(outputContract),
                outputContract is null ? (ushort)0 : GetNamedUInt16(outputContract, "ContractMajor") ?? 1,
                outputContract is null ? (ushort)0 : GetNamedUInt16(outputContract, "ContractMinor") ?? 0,
                AdapterRequired: true);
        }

        return new PassSocketContractModel(
            "GPU_RESOURCE",
            null,
            0,
            0,
            [],
            null,
            null,
            0,
            0,
            AdapterRequired: false);
    }

    private static ImmutableArray<PassSocketCapabilityModel> AssetCapabilities(ITypeSymbol assetType)
    {
        var result = ImmutableArray.CreateBuilder<PassSocketCapabilityModel>();
        foreach (var attribute in assetType.GetAttributes())
        {
            if (!IsGenericAssetAttribute(attribute, AssetCapabilityAttributeMetadataName) ||
                attribute.AttributeClass is not { TypeArguments.Length: 1 } capabilityClass)
            {
                continue;
            }

            var contract = FindAttribute(
                capabilityClass.TypeArguments[0].GetAttributes(),
                AssetCapabilityAttributeName);
            result.Add(new PassSocketCapabilityModel(
                contract is null ? string.Empty : GetConstructorString(contract) ?? string.Empty,
                GetNamedUInt16(attribute, "MinimumMajor") ?? 1,
                GetNamedUInt16(attribute, "MinimumMinor") ?? 0,
                !GetNamedArgument(attribute, "Required", out var required) || required.Value is not false));
        }

        return result.OrderBy(static capability => capability.CapabilityId, StringComparer.Ordinal)
            .ToImmutableArray();
    }

    private static AttributeData? FindAssetOutput(
        ITypeSymbol assetType,
        ITypeSymbol outputType,
        string? slotGuid)
        => assetType.GetAttributes().FirstOrDefault(attribute =>
            IsGenericAssetAttribute(attribute, AssetOutputAttributeMetadataName) &&
            attribute.AttributeClass is { TypeArguments.Length: 1 } outputClass &&
            SymbolEqualityComparer.Default.Equals(outputClass.TypeArguments[0], outputType) &&
            GetConstructorString(attribute) == slotGuid);

    private static bool IsGenericAssetAttribute(AttributeData attribute, string metadataName)
        => attribute.AttributeClass?.OriginalDefinition.MetadataName == metadataName &&
           attribute.AttributeClass.ContainingNamespace.ToDisplayString() == "Feather.Assets";

    private static PassParameterModel CreateParameter(
        Compilation compilation,
        ISymbol member,
        ITypeSymbol memberType,
        AttributeData attribute,
        CancellationToken cancellationToken)
    {
        var enumContract = memberType is INamedTypeSymbol { TypeKind: TypeKind.Enum } enumType
            ? CreateEnumContract(compilation, member, enumType, attribute, cancellationToken)
            : null;
        var defaultValue = enumContract is null
            ? GetNamedArgument(attribute, "DefaultValue", out var explicitDefault)
                ? JsonConstant(explicitDefault)
                : GetInitializerConstant(compilation, member, cancellationToken) ?? DefaultValue(memberType)
            : null;
        var runtimeAbi = RuntimeAbi(memberType, enumContract);

        return new PassParameterModel(
            GetConstructorString(attribute) ?? string.Empty,
            GetNamedString(attribute, "Name") ?? member.Name,
            ParameterType(memberType),
            LogicalType(memberType, enumContract),
            enumContract?.TypeGuid,
            defaultValue,
            GetNamedDouble(attribute, "Min"),
            GetNamedDouble(attribute, "Max"),
            GetNamedDouble(attribute, "Step"),
            GetNamedString(attribute, "Unit"),
            GetNamedString(attribute, "Description"),
            GetNamedString(attribute, "Group"),
            GetNamedInt32(attribute, "Order") ?? 0,
            GetNamedString(attribute, "EditorHint"),
            ParameterMutability(attribute),
            ParameterBindings(attribute),
            ParameterRedaction(attribute),
            runtimeAbi,
            enumContract,
            member.Locations.FirstOrDefault() ?? Location.None);
    }

    private static PassEnumModel CreateEnumContract(
        Compilation compilation,
        ISymbol member,
        INamedTypeSymbol enumType,
        AttributeData parameterAttribute,
        CancellationToken cancellationToken)
    {
        var contract = FindAttribute(enumType.GetAttributes(), FeatherEnumAttributeName);
        bool isFlags = FindAttribute(enumType.GetAttributes(), FlagsAttributeName) is not null;
        var members = ImmutableArray.CreateBuilder<PassEnumMemberModel>();
        foreach (var field in enumType.GetMembers().OfType<IFieldSymbol>()
                     .Where(static field => field.HasConstantValue)
                     .OrderBy(static field => field.Locations.FirstOrDefault()?.SourceSpan.Start ?? int.MaxValue)
                     .ThenBy(static field => field.Name, StringComparer.Ordinal))
        {
            var metadata = FindAttribute(field.GetAttributes(), FeatherEnumMemberAttributeName);
            members.Add(new PassEnumMemberModel(
                metadata is null ? string.Empty : GetConstructorString(metadata) ?? string.Empty,
                field.Name,
                EnumValue(field.ConstantValue),
                metadata is null ? field.Name : GetNamedString(metadata, "Name") ?? field.Name,
                metadata is null ? null : GetNamedString(metadata, "Description"),
                metadata is null ? 0 : GetNamedInt32(metadata, "Order") ?? 0,
                metadata is not null && GetNamedBoolean(metadata, "Deprecated"),
                metadata is null ? null : GetNamedString(metadata, "ReplacementMemberGuid"),
                field.Locations.FirstOrDefault() ?? Location.None));
        }

        object? defaultValue = null;
        if (GetNamedArgument(parameterAttribute, "DefaultValue", out var explicitDefault))
        {
            defaultValue = explicitDefault.Value;
        }
        else if (!TryGetInitializerConstantValue(compilation, member, cancellationToken, out defaultValue))
        {
            defaultValue = 0;
        }
        long rawDefault = EnumValue(defaultValue);
        long allowedMask = 0;
        foreach (var enumMember in members)
        {
            allowedMask |= enumMember.NumericValue;
        }

        return new PassEnumModel(
            contract is null ? string.Empty : GetConstructorString(contract) ?? string.Empty,
            enumType.ToDisplayString(SymbolDisplayFormat.CSharpErrorMessageFormat),
            UnderlyingScalar(enumType.EnumUnderlyingType),
            isFlags,
            contract is not null && GetNamedBoolean(contract, "AllowUnknownNumeric"),
            contract is not null && GetNamedBoolean(contract, "AllowUnknownBits"),
            rawDefault,
            allowedMask,
            members.ToImmutable(),
            enumType.Locations.FirstOrDefault() ?? Location.None);
    }

    private static AttributeData? FindAttribute(ImmutableArray<AttributeData> attributes, string name)
        => attributes.FirstOrDefault(attribute =>
            attribute.AttributeClass?.ToDisplayString(SymbolDisplayFormat.CSharpErrorMessageFormat) == name);

    private static string? GetConstructorString(AttributeData attribute)
        => attribute.ConstructorArguments.Length > 0
            ? attribute.ConstructorArguments[0].Value as string
            : null;

    private static string? GetNamedString(AttributeData attribute, string name)
        => GetNamedArgument(attribute, name, out var value) ? value.Value as string : null;

    private static int? GetNamedInt32(AttributeData attribute, string name)
        => GetNamedArgument(attribute, name, out var value) && value.Value is int result
            ? result
            : null;

    private static ushort? GetNamedUInt16(AttributeData attribute, string name)
        => GetNamedArgument(attribute, name, out var value) && value.Value is ushort result
            ? result
            : null;

    private static double? GetNamedDouble(AttributeData attribute, string name)
    {
        if (!GetNamedArgument(attribute, name, out var value) || value.Value is null)
        {
            return null;
        }

        var result = Convert.ToDouble(value.Value, CultureInfo.InvariantCulture);
        return double.IsNaN(result) || double.IsInfinity(result) ? null : result;
    }

    private static bool GetNamedBoolean(AttributeData attribute, string name)
        => GetNamedArgument(attribute, name, out var value) && value.Value is true;

    private static string ParameterMutability(AttributeData attribute)
        => GetNamedEnumMemberName(attribute, "Mutability") switch
        {
            "Specialization" => "SPECIALIZATION",
            "ResourceShape" => "RESOURCE_SHAPE",
            "CompileTime" => "COMPILE_TIME",
            _ => "DYNAMIC",
        };

    private static ImmutableArray<string> ParameterBindings(AttributeData attribute)
    {
        long value = GetNamedArgument(attribute, "Bindings", out var bindingValue) &&
                     bindingValue.Value is not null
            ? Convert.ToInt64(bindingValue.Value, CultureInfo.InvariantCulture)
            : 1L;
        var result = ImmutableArray.CreateBuilder<string>();
        if ((value & (1L << 0)) != 0) result.Add("INSTANCE");
        if ((value & (1L << 1)) != 0) result.Add("GRAPH_VALUE");
        if ((value & (1L << 2)) != 0) result.Add("RUNTIME_PROPERTY");
        if ((value & (1L << 3)) != 0) result.Add("TIMELINE");
        if ((value & (1L << 4)) != 0) result.Add("PUBLIC");
        return result.ToImmutable();
    }

    private static string ParameterRedaction(AttributeData attribute)
        => GetNamedEnumMemberName(attribute, "Redaction") switch
        {
            "MetadataOnly" => "METADATA_ONLY",
            "Secret" => "SECRET",
            _ => "PUBLIC",
        };

    private static string? GetNamedEnumMemberName(AttributeData attribute, string name)
    {
        if (!GetNamedArgument(attribute, name, out var value) ||
            value.Type is not INamedTypeSymbol enumType ||
            value.Value is null)
        {
            return null;
        }
        return enumType.GetMembers().OfType<IFieldSymbol>()
            .FirstOrDefault(field => field.HasConstantValue && Equals(field.ConstantValue, value.Value))
            ?.Name;
    }

    private static bool GetNamedArgument(
        AttributeData attribute,
        string name,
        out TypedConstant value)
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

    private static string ResourceKind(ITypeSymbol type)
    {
        if (IsTypedBufferHandle(type))
        {
            return "Buffer";
        }

        return type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) switch
        {
            "global::Feather.RenderGraph.BufferHandle" => "Buffer",
            "global::Feather.RenderGraph.CameraHandle" => "Camera",
            "global::Feather.RenderGraph.LightTableHandle" => "LightTable",
            "global::Feather.RenderGraph.MaterialTableHandle" => "MaterialTable",
            "global::Feather.RenderGraph.SceneGeometryHandle" => "SceneGeometry",
            "global::Feather.RenderGraph.SceneObjectHandle" => "SceneObject",
            "global::Feather.RenderGraph.TextureHandle" => "Texture2D",
            "global::Feather.RenderGraph.TextureTableHandle" => "TextureTable",
            "global::Feather.RenderGraph.TimeHandle" => "Time",
            _ => "Value"
        };
    }

    private static string? BufferElementType(ITypeSymbol type)
    {
        if (type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) ==
            "global::Feather.RenderGraph.BufferHandle")
        {
            return "Unknown";
        }

        return type is INamedTypeSymbol { IsGenericType: true } named && IsTypedBufferHandle(named)
            ? CanonicalElementType(named.TypeArguments[0])
            : null;
    }

    private static bool IsTypedBufferHandle(ITypeSymbol type)
        => type is INamedTypeSymbol { IsGenericType: true } named &&
           named.ConstructedFrom.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) ==
           "global::Feather.RenderGraph.BufferHandle<T>";

    private static string CanonicalElementType(ITypeSymbol type)
    {
        var shaderType = ShaderTypeFactory.FromTypeSymbol(type);
        if (shaderType is not null)
        {
            return TrimGlobalPrefix(shaderType.CSharpTypeName);
        }

        return TrimGlobalPrefix(type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat));
    }

    private static string TrimGlobalPrefix(string typeName)
        => typeName.StartsWith("global::", StringComparison.Ordinal)
            ? typeName.Substring("global::".Length)
            : typeName;

    private static string GetTextureFormat(AttributeData attribute)
    {
        if (!GetNamedArgument(attribute, "Format", out var value) ||
            value.Type is not INamedTypeSymbol enumType ||
            value.Value is null)
        {
            return "Unknown";
        }

        var enumName = enumType.GetMembers()
            .OfType<IFieldSymbol>()
            .FirstOrDefault(field => field.HasConstantValue && Equals(field.ConstantValue, value.Value))
            ?.Name;
        return enumName switch
        {
            "Rg8" => "RG8",
            "Rgba8" => "RGBA8",
            "Bgra8" => "BGRA8",
            "Rg16Float" => "RG16Float",
            "Rgba16Float" => "RGBA16Float",
            "Rg32Float" => "RG32Float",
            "Rgba32Float" => "RGBA32Float",
            null => "Unknown",
            _ => enumName
        };
    }

    private static string ParameterType(ITypeSymbol type)
    {
        return type.SpecialType switch
        {
            SpecialType.System_Boolean => "bool",
            SpecialType.System_Byte => "byte",
            SpecialType.System_Char => "char",
            SpecialType.System_Decimal => "decimal",
            SpecialType.System_Double => "double",
            SpecialType.System_Int16 => "short",
            SpecialType.System_Int32 => "int",
            SpecialType.System_Int64 => "long",
            SpecialType.System_SByte => "sbyte",
            SpecialType.System_Single => "float",
            SpecialType.System_String => "string",
            SpecialType.System_UInt16 => "ushort",
            SpecialType.System_UInt32 => "uint",
            SpecialType.System_UInt64 => "ulong",
            // Vector parameters are named without their namespace, matching how the shader side and
            // the Blender UI refer to them. A fully-qualified name would not be recognised as a
            // vector, so the parameter would fall back to a scalar control.
            _ => type.ToDisplayString(SymbolDisplayFormat.CSharpErrorMessageFormat) switch
            {
                "Feather.Math.float2" => "float2",
                "Feather.Math.float3" => "float3",
                "Feather.Math.float4" => "float4",
                var other => other
            }
        };
    }

    private static string LogicalType(ITypeSymbol type, PassEnumModel? enumContract)
    {
        if (enumContract is not null) return enumContract.IsFlags ? "FLAGS" : "ENUM";
        return type.SpecialType switch
        {
            SpecialType.System_Boolean => "BOOL",
            SpecialType.System_Byte or SpecialType.System_UInt16 or SpecialType.System_UInt32 or
                SpecialType.System_UInt64 or SpecialType.System_Char => "UINT",
            SpecialType.System_SByte or SpecialType.System_Int16 or SpecialType.System_Int32 or
                SpecialType.System_Int64 => "INT",
            SpecialType.System_Single or SpecialType.System_Double or SpecialType.System_Decimal => "FLOAT",
            SpecialType.System_String => "STRING",
            _ => type.ToDisplayString(SymbolDisplayFormat.CSharpErrorMessageFormat) switch
            {
                "Feather.Math.float2" or "Feather.Math.float3" or "Feather.Math.float4" => "VECTOR",
                _ => "HOST_VALUE",
            },
        };
    }

    private static PassParameterRuntimeAbiModel RuntimeAbi(ITypeSymbol type, PassEnumModel? enumContract)
    {
        if (enumContract is not null)
        {
            var (size, alignment) = ScalarLayout(enumContract.UnderlyingScalar);
            return new PassParameterRuntimeAbiModel(
                "PASS_INSTANCE", enumContract.UnderlyingScalar, 0, size, alignment,
                "CLR_ENUM_UNDERLYING_LE");
        }

        return type.SpecialType switch
        {
            SpecialType.System_Boolean => Abi("BOOL32", 4, 4),
            SpecialType.System_SByte => Abi("I8", 1, 1),
            SpecialType.System_Byte => Abi("U8", 1, 1),
            SpecialType.System_Int16 => Abi("I16", 2, 2),
            SpecialType.System_UInt16 or SpecialType.System_Char => Abi("U16", 2, 2),
            SpecialType.System_Int32 => Abi("I32", 4, 4),
            SpecialType.System_UInt32 => Abi("U32", 4, 4),
            SpecialType.System_Int64 => Abi("I64", 8, 8),
            SpecialType.System_UInt64 => Abi("U64", 8, 8),
            SpecialType.System_Single => Abi("F32", 4, 4),
            SpecialType.System_Double => Abi("F64", 8, 8),
            SpecialType.System_Decimal => Abi("DECIMAL128", 16, 16, "CLR_VALUE"),
            SpecialType.System_String => Abi("UTF8", 0, 1, "UTF8_BOUNDED"),
            _ => type.ToDisplayString(SymbolDisplayFormat.CSharpErrorMessageFormat) switch
            {
                "Feather.Math.float2" => Abi("F32X2", 8, 8),
                "Feather.Math.float3" => Abi("F32X3", 12, 16),
                "Feather.Math.float4" => Abi("F32X4", 16, 16),
                _ => Abi("OPAQUE", 0, 1, "CLR_VALUE"),
            },
        };
    }

    private static PassParameterRuntimeAbiModel Abi(
        string scalarKind,
        int size,
        int alignment,
        string packing = "FEATHER_SCALAR_LE")
        => new("PASS_INSTANCE", scalarKind, 0, size, alignment, packing);

    private static (int Size, int Alignment) ScalarLayout(string scalarKind)
        => scalarKind switch
        {
            "I8" or "U8" => (1, 1),
            "I16" or "U16" => (2, 2),
            "I32" or "U32" => (4, 4),
            "I64" or "U64" => (8, 8),
            _ => (0, 1),
        };

    private static string UnderlyingScalar(ITypeSymbol? type)
        => type?.SpecialType switch
        {
            SpecialType.System_SByte => "I8",
            SpecialType.System_Byte => "U8",
            SpecialType.System_Int16 => "I16",
            SpecialType.System_UInt16 => "U16",
            SpecialType.System_Int32 => "I32",
            SpecialType.System_UInt32 => "U32",
            SpecialType.System_Int64 => "I64",
            SpecialType.System_UInt64 => "U64",
            _ => "UNSUPPORTED",
        };

    private static long EnumValue(object? value)
        => value switch
        {
            null => 0,
            sbyte item => item,
            byte item => item,
            short item => item,
            ushort item => item,
            int item => item,
            uint item => item,
            long item => item,
            ulong item => unchecked((long)item),
            _ => Convert.ToInt64(value, CultureInfo.InvariantCulture),
        };

    private static string? GetInitializerConstant(
        Compilation compilation,
        ISymbol member,
        CancellationToken cancellationToken)
    {
        foreach (var syntaxReference in member.DeclaringSyntaxReferences)
        {
            var syntax = syntaxReference.GetSyntax(cancellationToken);
            ExpressionSyntax? expression = syntax switch
            {
                PropertyDeclarationSyntax property => property.Initializer?.Value,
                VariableDeclaratorSyntax field => field.Initializer?.Value,
                _ => null
            };
            if (expression is null)
            {
                continue;
            }

            var semanticModel = compilation.GetSemanticModel(expression.SyntaxTree);
            var constant = semanticModel.GetConstantValue(expression, cancellationToken);
            if (constant.HasValue)
            {
                return JsonValue(constant.Value);
            }

            // A vector default is written `new float3(x, y, z)`, which is not a compile-time
            // constant, so read its arguments from the syntax instead. Without this the manifest
            // records no default and the host rejects the parameter as an unconvertible value.
            if (VectorInitializer(semanticModel, expression, cancellationToken) is { } vector)
            {
                return vector;
            }
        }

        return null;
    }

    private static bool TryGetInitializerConstantValue(
        Compilation compilation,
        ISymbol member,
        CancellationToken cancellationToken,
        out object? value)
    {
        foreach (var syntaxReference in member.DeclaringSyntaxReferences)
        {
            var syntax = syntaxReference.GetSyntax(cancellationToken);
            ExpressionSyntax? expression = syntax switch
            {
                PropertyDeclarationSyntax property => property.Initializer?.Value,
                VariableDeclaratorSyntax field => field.Initializer?.Value,
                _ => null,
            };
            if (expression is null) continue;
            var constant = compilation.GetSemanticModel(expression.SyntaxTree)
                .GetConstantValue(expression, cancellationToken);
            if (!constant.HasValue) continue;
            value = constant.Value;
            return true;
        }
        value = null;
        return false;
    }

    private static string? VectorInitializer(
        SemanticModel semanticModel,
        ExpressionSyntax expression,
        CancellationToken cancellationToken)
    {
        if (expression is not BaseObjectCreationExpressionSyntax creation)
        {
            return null;
        }

        var type = semanticModel.GetTypeInfo(expression, cancellationToken).Type;
        var expected = type?.ToDisplayString(SymbolDisplayFormat.CSharpErrorMessageFormat) switch
        {
            "Feather.Math.float2" => 2,
            "Feather.Math.float3" => 3,
            "Feather.Math.float4" => 4,
            _ => 0
        };
        if (expected == 0)
        {
            return null;
        }

        var arguments = creation.ArgumentList?.Arguments ?? default;
        var components = new List<string>(expected);
        foreach (var argument in arguments)
        {
            var value = semanticModel.GetConstantValue(argument.Expression, cancellationToken);
            if (!value.HasValue || value.Value is null)
            {
                return null;
            }
            components.Add(JsonValue(value.Value) ?? "0");
        }

        // A single scalar argument fills every component, matching the vector constructors.
        if (components.Count == 1)
        {
            var repeated = components[0];
            while (components.Count < expected)
            {
                components.Add(repeated);
            }
        }
        if (components.Count != expected)
        {
            return null;
        }
        return "[" + string.Join(", ", components) + "]";
    }

    private static string? DefaultValue(ITypeSymbol type)
    {
        return type.SpecialType switch
        {
            SpecialType.System_Boolean => "false",
            SpecialType.System_Byte or
            SpecialType.System_Decimal or
            SpecialType.System_Double or
            SpecialType.System_Int16 or
            SpecialType.System_Int32 or
            SpecialType.System_Int64 or
            SpecialType.System_SByte or
            SpecialType.System_Single or
            SpecialType.System_UInt16 or
            SpecialType.System_UInt32 or
            SpecialType.System_UInt64 => "0",
            SpecialType.System_Char => JsonString("\0"),
            SpecialType.System_String => "null",
            _ when type.IsReferenceType || type.NullableAnnotation == NullableAnnotation.Annotated => "null",
            _ => null
        };
    }

    private static string? JsonConstant(TypedConstant constant)
        => constant.IsNull ? "null" : JsonValue(constant.Value);

    private static string? JsonValue(object? value)
    {
        return value switch
        {
            null => "null",
            bool boolean => boolean ? "true" : "false",
            char character => JsonString(character.ToString()),
            string text => JsonString(text),
            float number when !float.IsNaN(number) && !float.IsInfinity(number)
                => number.ToString("R", CultureInfo.InvariantCulture),
            double number when !double.IsNaN(number) && !double.IsInfinity(number)
                => number.ToString("R", CultureInfo.InvariantCulture),
            IFormattable formattable => formattable.ToString(null, CultureInfo.InvariantCulture),
            _ => null
        };
    }

    private static string JsonString(string value)
    {
        var builder = new StringBuilder(value.Length + 2);
        builder.Append('"');
        foreach (var character in value)
        {
            switch (character)
            {
                case '"':
                    builder.Append("\\\"");
                    break;
                case '\\':
                    builder.Append("\\\\");
                    break;
                case '\b':
                    builder.Append("\\b");
                    break;
                case '\f':
                    builder.Append("\\f");
                    break;
                case '\n':
                    builder.Append("\\n");
                    break;
                case '\r':
                    builder.Append("\\r");
                    break;
                case '\t':
                    builder.Append("\\t");
                    break;
                default:
                    if (character < ' ')
                    {
                        builder.Append("\\u")
                            .Append(((int)character).ToString("x4", CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        builder.Append(character);
                    }
                    break;
            }
        }
        return builder.Append('"').ToString();
    }
}
