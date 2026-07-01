using System.Globalization;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Operations;

namespace Feather.Generators.Model;

internal static class ShaderSemanticFacts
{
    public static string GetMethodMetadataName(IMethodSymbol method)
        => method.ContainingType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) + "." + method.Name;

    public static string GetTypeName(ITypeSymbol? type)
        => type?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) ?? string.Empty;

    public static bool TryFormatNumericLiteral(IOperation operation, out string literal)
    {
        literal = string.Empty;
        if (!TryUnwrapConversion(operation, out var unwrapped))
        {
            return false;
        }

        if (unwrapped is IUnaryOperation { OperatorKind: UnaryOperatorKind.Minus } unary
            && TryFormatNumericLiteral(unary.Operand, out var operand))
        {
            literal = "-" + operand;
            return true;
        }

        if (!unwrapped.ConstantValue.HasValue || unwrapped.ConstantValue.Value is null)
        {
            return false;
        }

        literal = unwrapped.ConstantValue.Value switch
        {
            float value => value.ToString("R", CultureInfo.InvariantCulture),
            double value => value.ToString("R", CultureInfo.InvariantCulture),
            int value => value.ToString(CultureInfo.InvariantCulture),
            uint value => value.ToString(CultureInfo.InvariantCulture),
            short value => value.ToString(CultureInfo.InvariantCulture),
            ushort value => value.ToString(CultureInfo.InvariantCulture),
            byte value => value.ToString(CultureInfo.InvariantCulture),
            sbyte value => value.ToString(CultureInfo.InvariantCulture),
            _ => string.Empty
        };
        return literal.Length > 0;
    }

    public static bool TryUnwrapConversion(IOperation operation, out IOperation unwrapped)
    {
        unwrapped = operation;
        while (unwrapped is IConversionOperation conversion)
        {
            unwrapped = conversion.Operand;
        }

        return true;
    }

    public static bool IsFloatType(ITypeSymbol? type)
        => type?.SpecialType == SpecialType.System_Single;

    public static bool IsFloatVectorType(ITypeSymbol? type, int componentCount)
    {
        var typeName = type?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        return typeName == "global::Feather.Math.float" + componentCount.ToString(CultureInfo.InvariantCulture);
    }

    public static bool IsFloatVectorType(ITypeSymbol? type)
        => IsFloatVectorType(type, 2) || IsFloatVectorType(type, 3) || IsFloatVectorType(type, 4);

    public static bool IsSquareFloatMatrixType(ITypeSymbol? type)
    {
        var typeName = type?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        return typeName is "global::Feather.Math.float2x2"
            or "global::Feather.Math.float3x3"
            or "global::Feather.Math.float4x4";
    }

    public static bool HasFloatSignature(IMethodSymbol method, int parameterCount)
        => IsFloatType(method.ReturnType)
            && method.Parameters.Length == parameterCount
            && method.Parameters.All(static parameter => IsFloatType(parameter.Type));

    public static bool HasFloatOrMatchingFloatVectorUnarySignature(IMethodSymbol method)
        => method.Parameters.Length == 1
            && IsSameType(method.ReturnType, method.Parameters[0].Type)
            && (IsFloatType(method.ReturnType) || IsFloatVectorType(method.ReturnType));

    public static bool HasFloatOrMatchingFloatVectorBinarySignature(IMethodSymbol method)
    {
        if (HasFloatSignature(method, 2))
        {
            return true;
        }

        return method.Parameters.Length == 2
            && IsFloatVectorType(method.ReturnType)
            && IsSameType(method.ReturnType, method.Parameters[0].Type)
            && IsSameType(method.ReturnType, method.Parameters[1].Type);
    }

    public static bool HasFloatOrMatchingFloatVectorClampSignature(IMethodSymbol method)
    {
        if (HasFloatSignature(method, 3))
        {
            return true;
        }

        if (method.Parameters.Length != 3 ||
            !IsFloatVectorType(method.ReturnType) ||
            !IsSameType(method.ReturnType, method.Parameters[0].Type))
        {
            return false;
        }

        return (IsFloatType(method.Parameters[1].Type) && IsFloatType(method.Parameters[2].Type)) ||
            (IsSameType(method.ReturnType, method.Parameters[1].Type) &&
             IsSameType(method.ReturnType, method.Parameters[2].Type));
    }

    public static bool HasFloatOrMatchingFloatVectorLerpSignature(IMethodSymbol method)
    {
        if (HasFloatSignature(method, 3))
        {
            return true;
        }

        return method.Parameters.Length == 3
            && IsFloatVectorType(method.ReturnType)
            && IsSameType(method.ReturnType, method.Parameters[0].Type)
            && IsSameType(method.ReturnType, method.Parameters[1].Type)
            && IsFloatType(method.Parameters[2].Type);
    }

    public static bool HasFloatVectorDotSignature(IMethodSymbol method)
        => IsFloatType(method.ReturnType)
            && method.Parameters.Length == 2
            && ((IsFloatVectorType(method.Parameters[0].Type, 2) && IsFloatVectorType(method.Parameters[1].Type, 2))
                || (IsFloatVectorType(method.Parameters[0].Type, 3) && IsFloatVectorType(method.Parameters[1].Type, 3))
                || (IsFloatVectorType(method.Parameters[0].Type, 4) && IsFloatVectorType(method.Parameters[1].Type, 4)));

    public static bool HasFloat3CrossSignature(IMethodSymbol method)
        => IsFloatVectorType(method.ReturnType, 3)
            && method.Parameters.Length == 2
            && IsFloatVectorType(method.Parameters[0].Type, 3)
            && IsFloatVectorType(method.Parameters[1].Type, 3);

    public static bool HasMatrixTransformSignature(IMethodSymbol method)
        => method.Parameters.Length == 1
            && IsSquareFloatMatrixType(method.ReturnType)
            && IsSameType(method.ReturnType, method.Parameters[0].Type);

    public static bool HasMatrixScalarSignature(IMethodSymbol method)
        => IsFloatType(method.ReturnType)
            && method.Parameters.Length == 1
            && IsSquareFloatMatrixType(method.Parameters[0].Type);

    public static bool HasMatrixHadamardSignature(IMethodSymbol method)
        => method.Parameters.Length == 2
            && IsSquareFloatMatrixType(method.ReturnType)
            && IsSameType(method.ReturnType, method.Parameters[0].Type)
            && IsSameType(method.ReturnType, method.Parameters[1].Type);

    public static bool HasMatrixMulSignature(IMethodSymbol method)
    {
        if (method.Parameters.Length != 2)
        {
            return false;
        }

        if (IsSquareFloatMatrixType(method.ReturnType) &&
            IsSameType(method.ReturnType, method.Parameters[0].Type) &&
            IsSameType(method.ReturnType, method.Parameters[1].Type))
        {
            return true;
        }

        return IsFloatVectorType(method.ReturnType)
            && IsSquareFloatMatrixType(method.Parameters[0].Type)
            && IsSameType(method.ReturnType, method.Parameters[1].Type);
    }

    private static bool IsSameType(ITypeSymbol? left, ITypeSymbol? right)
        => SymbolEqualityComparer.Default.Equals(left, right);

    public static bool IsUniformResourceType(ITypeSymbol? type)
        => type is INamedTypeSymbol named
            && named.IsGenericType
            && named.ConstructedFrom.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) == "global::Feather.Resources.Uniform<T>";

    public static bool IsSupportedScalarPushConstantType(ITypeSymbol type)
        => type.SpecialType is SpecialType.System_Single or SpecialType.System_Int32 or SpecialType.System_UInt32;

    public static bool IsShaderResourceType(ITypeSymbol? type)
    {
        if (type is not INamedTypeSymbol named || !named.IsGenericType)
        {
            return false;
        }

        var genericName = named.ConstructedFrom.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        return genericName is "global::Feather.Resources.ReadOnlyBuffer<T>"
            or "global::Feather.Resources.WriteOnlyBuffer<T>"
            or "global::Feather.Resources.ReadWriteBuffer<T>"
            or "global::Feather.Resources.ReadOnlyTexture2D<T>"
            or "global::Feather.Resources.WriteOnlyTexture2D<T>"
            or "global::Feather.Resources.ReadWriteTexture2D<T>"
            or "global::Feather.Resources.ReadWriteNormalizedTexture2D<T>"
            or "global::Feather.Resources.SampledTexture2D<T>"
            or "global::Feather.Resources.ReadOnlyTexture3D<T>"
            or "global::Feather.Resources.WriteOnlyTexture3D<T>"
            or "global::Feather.Resources.ReadWriteTexture3D<T>"
            or "global::Feather.Resources.ReadWriteNormalizedTexture3D<T>";
    }

    public static bool IsFeatherVectorType(ITypeSymbol? type)
    {
        if (type is null) return false;
        var name = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        return name is "global::Feather.Math.float2" or "global::Feather.Math.float3" or "global::Feather.Math.float4"
            or "global::Feather.Math.int2" or "global::Feather.Math.int3" or "global::Feather.Math.int4"
            or "global::Feather.Math.bool2" or "global::Feather.Math.bool3" or "global::Feather.Math.bool4";
    }

    public static bool IsFeatherMatrixType(ITypeSymbol? type)
    {
        if (type is null) return false;
        var name = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        return name is "global::Feather.Math.float2x2" or "global::Feather.Math.float3x3" or "global::Feather.Math.float4x4"
            or "global::Feather.Math.float2x3" or "global::Feather.Math.float3x2"
            or "global::Feather.Math.float2x4" or "global::Feather.Math.float4x2"
            or "global::Feather.Math.float3x4" or "global::Feather.Math.float4x3";
    }

    public static bool IsCallableMethod(IMethodSymbol method)
        => method.GetAttributes().Any(static a =>
            a.AttributeClass?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) is "global::Feather.CallableAttribute"
                or "global::Feather.ShaderFunctionAttribute");

    public static bool IsGpuStructType(ITypeSymbol? type)
        => type is not null
            && type.GetAttributes().Any(static a =>
                a.AttributeClass?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) == "global::Feather.GpuStructAttribute");

    /// <summary>Returns the canonical field index (0-based) for a known GpuStruct member.</summary>
    public static bool TryGetGpuStructFieldIndex(IPropertyReferenceOperation propRef, out int fieldIndex, out string fieldTypeName)
        => GpuStructFieldDiscovery.TryGetFieldIndex(propRef.Property, out fieldIndex, out fieldTypeName);

    /// <summary>Returns the canonical field index (0-based) for a known GpuStruct member.</summary>
    public static bool TryGetGpuStructFieldIndex(IFieldReferenceOperation fieldRef, out int fieldIndex, out string fieldTypeName)
        => GpuStructFieldDiscovery.TryGetFieldIndex(fieldRef.Field, out fieldIndex, out fieldTypeName);

    public static bool IsBuiltinField(IFieldSymbol field, out ShaderBuiltinKind kind)
    {
        kind = default;
        if (field.ContainingType?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) != "global::Feather.ThreadIds")
            return false;
        if (!field.IsStatic) return false;
        kind = field.Name switch { "X" => ShaderBuiltinKind.ThreadIndexX, "Y" => ShaderBuiltinKind.ThreadIndexY, "Z" => ShaderBuiltinKind.ThreadIndexZ, _ => default };
        return kind != default;
    }

    public static bool TryGetBuiltinKind(IPropertyReferenceOperation prop, out LoweredShaderBuiltinKind kind)
    {
        kind = default;
        var containingType = prop.Property.ContainingType;
        if (containingType is null) return false;
        var typeName = containingType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        if (typeName == "global::Feather.ThreadIds")
            kind = prop.Property.Name switch { "X" => LoweredShaderBuiltinKind.ThreadIndexX, "Y" => LoweredShaderBuiltinKind.ThreadIndexY, "Z" => LoweredShaderBuiltinKind.ThreadIndexZ, _ => default };
        else if (typeName == "global::Feather.LocalIds")
            kind = prop.Property.Name switch { "X" => LoweredShaderBuiltinKind.LocalIndexX, "Y" => LoweredShaderBuiltinKind.LocalIndexY, "Z" => LoweredShaderBuiltinKind.LocalIndexZ, _ => default };
        else if (typeName == "global::Feather.GroupIds")
            kind = prop.Property.Name switch { "X" => LoweredShaderBuiltinKind.GroupIdX, "Y" => LoweredShaderBuiltinKind.GroupIdY, "Z" => LoweredShaderBuiltinKind.GroupIdZ, _ => default };
        else if (typeName == "global::Feather.DispatchSize")
            kind = prop.Property.Name switch { "X" => LoweredShaderBuiltinKind.DispatchSizeX, "Y" => LoweredShaderBuiltinKind.DispatchSizeY, "Z" => LoweredShaderBuiltinKind.DispatchSizeZ, _ => default };
        else if (typeName == "global::Feather.GroupSize")
            kind = prop.Property.Name switch { "X" => LoweredShaderBuiltinKind.GroupSizeX, "Y" => LoweredShaderBuiltinKind.GroupSizeY, "Z" => LoweredShaderBuiltinKind.GroupSizeZ, _ => default };
        else if (typeName == "global::Feather.Graphics.VertexIds")
            kind = prop.Property.Name switch { "Index" => LoweredShaderBuiltinKind.VertexIndex, "Instance" => LoweredShaderBuiltinKind.InstanceIndex, _ => default };
        else if (typeName == "global::Feather.Graphics.FragmentIds")
            kind = prop.Property.Name switch { "Coord" => LoweredShaderBuiltinKind.FragmentCoordX, _ => default };
        else return false;
        return kind != default;
    }
}
