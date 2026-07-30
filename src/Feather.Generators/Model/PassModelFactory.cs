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
    private const string RenderPassInterfaceName = "Feather.RenderGraph.IRenderPass";
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
        return new PassSocketModel(
            GetConstructorString(attribute) ?? string.Empty,
            GetNamedString(attribute, "Name") ?? member.Name,
            ResourceKind(memberType),
            BufferElementType(memberType),
            GetTextureFormat(attribute),
            access,
            member.Locations.FirstOrDefault() ?? Location.None);
    }

    private static PassParameterModel CreateParameter(
        Compilation compilation,
        ISymbol member,
        ITypeSymbol memberType,
        AttributeData attribute,
        CancellationToken cancellationToken)
    {
        var defaultValue = GetNamedArgument(attribute, "DefaultValue", out var explicitDefault)
            ? JsonConstant(explicitDefault)
            : GetInitializerConstant(compilation, member, cancellationToken) ?? DefaultValue(memberType);

        return new PassParameterModel(
            GetConstructorString(attribute) ?? string.Empty,
            GetNamedString(attribute, "Name") ?? member.Name,
            ParameterType(memberType),
            defaultValue,
            GetNamedDouble(attribute, "Min"),
            GetNamedDouble(attribute, "Max"),
            member.Locations.FirstOrDefault() ?? Location.None);
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

    private static double? GetNamedDouble(AttributeData attribute, string name)
    {
        if (!GetNamedArgument(attribute, name, out var value) || value.Value is null)
        {
            return null;
        }

        var result = Convert.ToDouble(value.Value, CultureInfo.InvariantCulture);
        return double.IsNaN(result) || double.IsInfinity(result) ? null : result;
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
