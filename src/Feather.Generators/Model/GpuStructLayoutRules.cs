using Microsoft.CodeAnalysis;

namespace Feather.Generators.Model;

internal static class GpuStructLayoutRules
{
    public static bool IsGpuStruct(ITypeSymbol? type)
        => type is INamedTypeSymbol named
            && named.GetAttributes().Any(static attr =>
                attr.AttributeClass?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) == "global::Feather.GpuStructAttribute"
                || attr.AttributeClass?.ToDisplayString() == "Feather.GpuStructAttribute");

    public static bool IsValidGpuStructLayout(INamedTypeSymbol symbol, bool requirePartial = false)
        => !GetLayoutErrors(symbol, requirePartial).Any();

    public static IEnumerable<GpuStructLayoutError> GetLayoutErrors(INamedTypeSymbol symbol, bool requirePartial)
        => GetLayoutErrors(symbol, requirePartial, new HashSet<INamedTypeSymbol>(SymbolEqualityComparer.Default));

    public static bool ContainsNarrowScalar(ITypeSymbol type)
        => ContainsNarrowScalar(type, new HashSet<ITypeSymbol>(SymbolEqualityComparer.Default));

    public static GpuStructTypeLayout GetTypeLayout(ITypeSymbol type)
    {
        if (type is IArrayTypeSymbol)
        {
            return GpuStructTypeLayout.Unsupported;
        }

        if (TryGetGpuArrayLayout(type, out var arrayLayout))
        {
            return arrayLayout;
        }

        if (type.SpecialType is SpecialType.System_Boolean)
        {
            return new GpuStructTypeLayout(4, 4);
        }

        if (type.SpecialType is SpecialType.System_Byte or SpecialType.System_SByte)
        {
            return new GpuStructTypeLayout(1, 1);
        }

        if (type.SpecialType is SpecialType.System_Int16 or SpecialType.System_UInt16)
        {
            return new GpuStructTypeLayout(2, 2);
        }

        if (type.SpecialType is SpecialType.System_Int32 or SpecialType.System_UInt32 or SpecialType.System_Single)
        {
            return new GpuStructTypeLayout(4, 4);
        }

        var typeName = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        return typeName switch
        {
            "global::Feather.Math.int2" or "global::Feather.Math.float2" or "global::Feather.Math.bool2" => new GpuStructTypeLayout(8, 8),
            "global::Feather.Math.int3" or "global::Feather.Math.float3" or "global::Feather.Math.bool3" => new GpuStructTypeLayout(12, 16),
            "global::Feather.Math.int4" or "global::Feather.Math.float4" or "global::Feather.Math.bool4" => new GpuStructTypeLayout(16, 16),
            "global::Feather.Math.float2x2" => new GpuStructTypeLayout(32, 16),
            "global::Feather.Math.float3x3" => new GpuStructTypeLayout(48, 16),
            "global::Feather.Math.float4x4" => new GpuStructTypeLayout(64, 16),
            _ => GetNestedStructLayout(type)
        };
    }

    public static bool TryGetGpuArrayLength(INamedTypeSymbol type, out int length)
    {
        length = 0;
        var generic = type.ConstructedFrom.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        const string prefix = "global::Feather.GpuArray";
        const string suffix = "<T>";
        if (!generic.StartsWith(prefix, StringComparison.Ordinal) ||
            !generic.EndsWith(suffix, StringComparison.Ordinal))
        {
            return false;
        }

        var digits = generic.Substring(prefix.Length, generic.Length - prefix.Length - suffix.Length);
        if (!int.TryParse(digits, out length) || length <= 0)
        {
            length = 0;
            return false;
        }

        return true;
    }

    public static bool UsesUnsupportedMatrixType(ITypeSymbol type)
    {
        var typeName = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        return typeName.StartsWith("global::System.Numerics.Matrix", StringComparison.Ordinal);
    }

    public static int Align(int value, int alignment)
    {
        if (alignment <= 1)
        {
            return value;
        }

        var remainder = value % alignment;
        return remainder == 0 ? value : checked(value + alignment - remainder);
    }

    private static IEnumerable<GpuStructLayoutError> GetLayoutErrors(
        INamedTypeSymbol symbol,
        bool requirePartial,
        HashSet<INamedTypeSymbol> visiting)
    {
        if (requirePartial && !IsPartial(symbol))
        {
            yield return new GpuStructLayoutError(
                symbol,
                symbol,
                "the struct must be partial so the generator can emit IGpuStruct<T>",
                IsMatrixError: false);
        }

        if (!symbol.IsUnmanagedType)
        {
            yield return new GpuStructLayoutError(symbol, symbol, "the struct must be unmanaged", IsMatrixError: false);
        }

        foreach (var property in GpuStructFieldDiscovery.GetUnsupportedProperties(symbol))
        {
            yield return new GpuStructLayoutError(
                property.Symbol,
                symbol,
                $"property '{property.Name}' is not a record primary-constructor storage property",
                IsMatrixError: false);
        }

        if (!visiting.Add(symbol))
        {
            yield break;
        }

        foreach (var field in GpuStructFieldDiscovery.GetFields(symbol))
        {
            var layout = GetTypeLayout(field.Type);
            if (UsesUnsupportedMatrixType(field.Type))
            {
                yield return new GpuStructLayoutError(field.Symbol, symbol, field.Name, IsMatrixError: true);
            }

            if (layout.SizeInBytes <= 0 || layout.Alignment <= 0)
            {
                yield return new GpuStructLayoutError(
                    field.Symbol,
                    symbol,
                    $"field '{field.Name}' uses an unsupported GPU type",
                    IsMatrixError: false);
            }

            if (field.IsReadonlyExplicitField)
            {
                yield return new GpuStructLayoutError(
                    field.Symbol,
                    symbol,
                    $"field '{field.Name}' is readonly and cannot be written by generated unpack",
                    IsMatrixError: false);
            }

            foreach (var nested in GetNestedGpuStructs(field.Type))
            {
                foreach (var nestedError in GetLayoutErrors(nested, requirePartial: false, visiting))
                {
                    yield return nestedError;
                }
            }
        }

        visiting.Remove(symbol);
    }

    private static bool IsPartial(INamedTypeSymbol symbol)
        => symbol.DeclaringSyntaxReferences
            .Select(static reference => reference.GetSyntax())
            .OfType<Microsoft.CodeAnalysis.CSharp.Syntax.TypeDeclarationSyntax>()
            .Any(static syntax => syntax.Modifiers.Any(Microsoft.CodeAnalysis.CSharp.SyntaxKind.PartialKeyword));

    private static IEnumerable<INamedTypeSymbol> GetNestedGpuStructs(ITypeSymbol type)
    {
        if (type is INamedTypeSymbol { IsGenericType: true, TypeArguments.Length: 1 } generic &&
            TryGetGpuArrayLength(generic, out _))
        {
            if (generic.TypeArguments[0] is INamedTypeSymbol arrayElement && IsGpuStruct(arrayElement))
            {
                yield return arrayElement;
            }

            yield break;
        }

        if (type is INamedTypeSymbol named && IsGpuStruct(named))
        {
            yield return named;
        }
    }

    private static bool TryGetGpuArrayLayout(ITypeSymbol type, out GpuStructTypeLayout layout)
    {
        layout = GpuStructTypeLayout.Unsupported;
        if (type is not INamedTypeSymbol { IsGenericType: true } named ||
            !TryGetGpuArrayLength(named, out var length))
        {
            return false;
        }

        var elementType = named.TypeArguments[0];
        if (elementType is IArrayTypeSymbol || IsGpuArrayType(elementType))
        {
            return true;
        }

        var elementLayout = GetTypeLayout(elementType);
        if (length <= 0 || elementLayout.SizeInBytes <= 0 || elementLayout.Alignment <= 0)
        {
            return true;
        }

        var stride = Align(elementLayout.SizeInBytes, elementLayout.Alignment);
        layout = new GpuStructTypeLayout(
            checked(stride * length),
            elementLayout.Alignment,
            length,
            stride,
            elementType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat));
        return true;
    }

    private static bool IsGpuArrayType(ITypeSymbol type)
        => type is INamedTypeSymbol { IsGenericType: true } named && TryGetGpuArrayLength(named, out _);

    private static GpuStructTypeLayout GetNestedStructLayout(ITypeSymbol type)
    {
        if (type is not INamedTypeSymbol named || !IsGpuStruct(named) || !IsValidGpuStructLayout(named, requirePartial: true))
        {
            return GpuStructTypeLayout.Unsupported;
        }

        var nestedFields = GpuStructFieldDiscovery.GetFields(named)
            .Select(static field => GetTypeLayout(field.Type))
            .ToArray();
        var alignment = nestedFields.Length == 0 ? 1 : nestedFields.Max(static field => field.Alignment);
        var offset = 0;
        foreach (var field in nestedFields)
        {
            offset = Align(offset, field.Alignment) + field.SizeInBytes;
        }

        var size = nestedFields.Length == 0 ? 0 : Align(offset, alignment);
        return new GpuStructTypeLayout(size, alignment);
    }

    private static bool ContainsNarrowScalar(ITypeSymbol type, HashSet<ITypeSymbol> visiting)
    {
        if (type.SpecialType is SpecialType.System_Byte
            or SpecialType.System_SByte
            or SpecialType.System_Int16
            or SpecialType.System_UInt16)
        {
            return true;
        }

        if (type is not INamedTypeSymbol named)
        {
            return false;
        }

        if (named.IsGenericType && TryGetGpuArrayLength(named, out _))
        {
            return ContainsNarrowScalar(named.TypeArguments[0], visiting);
        }

        if (!IsGpuStruct(named))
        {
            return false;
        }

        if (!visiting.Add(named))
        {
            return false;
        }

        foreach (var field in GpuStructFieldDiscovery.GetFields(named))
        {
            if (ContainsNarrowScalar(field.Type, visiting))
            {
                visiting.Remove(named);
                return true;
            }
        }

        visiting.Remove(named);
        return false;
    }
}

internal readonly record struct GpuStructLayoutError(
    ISymbol Symbol,
    INamedTypeSymbol StructSymbol,
    string Message,
    bool IsMatrixError);

internal readonly record struct GpuStructTypeLayout(
    int SizeInBytes,
    int Alignment,
    int ArrayLength = 0,
    int ArrayStride = 0,
    string? ArrayElementTypeName = null)
{
    public static GpuStructTypeLayout Unsupported => new(0, 0);
}
