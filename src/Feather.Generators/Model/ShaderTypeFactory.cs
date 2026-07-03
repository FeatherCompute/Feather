using Microsoft.CodeAnalysis;

namespace Feather.Generators.Model;

/// <summary>Converts Roslyn ITypeSymbol to ShaderType, providing cached singletons for common types.</summary>
internal static class ShaderTypeFactory
{
    public static readonly ShaderPrimitiveType Bool = new(ShaderPrimitiveKind.Bool) { CSharpTypeName = "bool" };
    public static readonly ShaderPrimitiveType Int = new(ShaderPrimitiveKind.Int) { CSharpTypeName = "int" };
    public static readonly ShaderPrimitiveType UInt = new(ShaderPrimitiveKind.UInt) { CSharpTypeName = "uint" };
    public static readonly ShaderPrimitiveType Float = new(ShaderPrimitiveKind.Float) { CSharpTypeName = "float" };
    public static readonly ShaderVoidType Void = new() { CSharpTypeName = "void" };

    public static readonly ShaderVectorType Float2 = new(Float, 2) { CSharpTypeName = "float2" };
    public static readonly ShaderVectorType Float3 = new(Float, 3) { CSharpTypeName = "float3" };
    public static readonly ShaderVectorType Float4 = new(Float, 4) { CSharpTypeName = "float4" };
    public static readonly ShaderVectorType Int2 = new(Int, 2) { CSharpTypeName = "int2" };
    public static readonly ShaderVectorType Int3 = new(Int, 3) { CSharpTypeName = "int3" };
    public static readonly ShaderVectorType Int4 = new(Int, 4) { CSharpTypeName = "int4" };
    public static readonly ShaderVectorType Bool2 = new(Bool, 2) { CSharpTypeName = "bool2" };
    public static readonly ShaderVectorType Bool3 = new(Bool, 3) { CSharpTypeName = "bool3" };
    public static readonly ShaderVectorType Bool4 = new(Bool, 4) { CSharpTypeName = "bool4" };

    public static readonly ShaderMatrixType Float2x2 = new(Float, 2, 2) { CSharpTypeName = "float2x2" };
    public static readonly ShaderMatrixType Float3x3 = new(Float, 3, 3) { CSharpTypeName = "float3x3" };
    public static readonly ShaderMatrixType Float4x4 = new(Float, 4, 4) { CSharpTypeName = "float4x4" };
    public static readonly ShaderMatrixType Float2x3 = new(Float, 2, 3) { CSharpTypeName = "float2x3" };
    public static readonly ShaderMatrixType Float3x2 = new(Float, 3, 2) { CSharpTypeName = "float3x2" };
    public static readonly ShaderMatrixType Float2x4 = new(Float, 2, 4) { CSharpTypeName = "float2x4" };
    public static readonly ShaderMatrixType Float4x2 = new(Float, 4, 2) { CSharpTypeName = "float4x2" };
    public static readonly ShaderMatrixType Float3x4 = new(Float, 3, 4) { CSharpTypeName = "float3x4" };
    public static readonly ShaderMatrixType Float4x3 = new(Float, 4, 3) { CSharpTypeName = "float4x3" };

    public static ShaderType? FromTypeSymbol(ITypeSymbol? type)
    {
        if (type is null) return null;
        if (type.SpecialType == SpecialType.System_Single) return Float;
        if (type.SpecialType == SpecialType.System_Int32) return Int;
        if (type.SpecialType == SpecialType.System_UInt32) return UInt;
        if (type.SpecialType == SpecialType.System_Int16) return new ShaderPrimitiveType(ShaderPrimitiveKind.Int) { CSharpTypeName = "short" };
        if (type.SpecialType == SpecialType.System_UInt16) return new ShaderPrimitiveType(ShaderPrimitiveKind.UInt) { CSharpTypeName = "ushort" };
        if (type.SpecialType == SpecialType.System_SByte) return new ShaderPrimitiveType(ShaderPrimitiveKind.Int) { CSharpTypeName = "sbyte" };
        if (type.SpecialType == SpecialType.System_Byte) return new ShaderPrimitiveType(ShaderPrimitiveKind.UInt) { CSharpTypeName = "byte" };
        if (type.SpecialType == SpecialType.System_Boolean) return Bool;
        if (type.SpecialType == SpecialType.System_Void) return Void;

        var name = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        if (name == "global::Feather.Resources.SamplerState")
        {
            return new ShaderResourceWrapperType(ShaderResourceKind.Sampler, Void, ShaderResourceAccess.Sample)
            {
                CSharpTypeName = name
            };
        }

        if (type is IArrayTypeSymbol array)
        {
            var elementType = FromTypeSymbol(array.ElementType);
            return elementType is null ? null : new ShaderArrayType(elementType, null)
            {
                CSharpTypeName = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)
            };
        }

        if (type is INamedTypeSymbol { IsGenericType: true } named)
        {
            var elementType = FromTypeSymbol(named.TypeArguments[0]);
            if (elementType is not null)
            {
                var genericName = named.ConstructedFrom.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
                if (TryGetGpuArrayLength(named, out var length))
                {
                    if (length <= 0 || IsGpuArrayType(named.TypeArguments[0]))
                    {
                        return null;
                    }

                    return new ShaderArrayType(elementType, length)
                    {
                        CSharpTypeName = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)
                    };
                }

                var resource = genericName switch
                {
                    "global::Feather.Resources.ReadOnlyBuffer<T>" => new ShaderResourceWrapperType(ShaderResourceKind.Buffer, elementType, ShaderResourceAccess.Read),
                    "global::Feather.Resources.WriteOnlyBuffer<T>" => new ShaderResourceWrapperType(ShaderResourceKind.Buffer, elementType, ShaderResourceAccess.Write),
                    "global::Feather.Resources.ReadWriteBuffer<T>" => new ShaderResourceWrapperType(ShaderResourceKind.Buffer, elementType, ShaderResourceAccess.ReadWrite),
                    "global::Feather.Resources.ReadOnlyTexture2D<T>" => new ShaderResourceWrapperType(ShaderResourceKind.Texture2D, elementType, ShaderResourceAccess.Read),
                    "global::Feather.Resources.WriteOnlyTexture2D<T>" => new ShaderResourceWrapperType(ShaderResourceKind.Texture2D, elementType, ShaderResourceAccess.Write),
                    "global::Feather.Resources.ReadWriteTexture2D<T>" => new ShaderResourceWrapperType(ShaderResourceKind.Texture2D, elementType, ShaderResourceAccess.ReadWrite),
                    "global::Feather.Resources.ReadWriteNormalizedTexture2D<T>" => new ShaderResourceWrapperType(ShaderResourceKind.Texture2D, elementType, ShaderResourceAccess.ReadWrite),
                    "global::Feather.Resources.SampledTexture2D<T>" => new ShaderResourceWrapperType(ShaderResourceKind.Texture2D, elementType, ShaderResourceAccess.Sample),
                    "global::Feather.Resources.ReadOnlyTexture3D<T>" => new ShaderResourceWrapperType(ShaderResourceKind.Texture3D, elementType, ShaderResourceAccess.Read),
                    "global::Feather.Resources.WriteOnlyTexture3D<T>" => new ShaderResourceWrapperType(ShaderResourceKind.Texture3D, elementType, ShaderResourceAccess.Write),
                    "global::Feather.Resources.ReadWriteTexture3D<T>" => new ShaderResourceWrapperType(ShaderResourceKind.Texture3D, elementType, ShaderResourceAccess.ReadWrite),
                    "global::Feather.Resources.ReadWriteNormalizedTexture3D<T>" => new ShaderResourceWrapperType(ShaderResourceKind.Texture3D, elementType, ShaderResourceAccess.ReadWrite),
                    "global::Feather.Resources.Uniform<T>" => new ShaderResourceWrapperType(ShaderResourceKind.Buffer, elementType, ShaderResourceAccess.Read),
                    _ => null
                };

                if (resource is not null)
                {
                    return resource with { CSharpTypeName = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) };
                }

                if (genericName == "global::Feather.SharedMemory<T>")
                {
                    return new ShaderArrayType(elementType, null)
                    {
                        CSharpTypeName = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)
                    };
                }
            }
        }

        return name switch
        {
            "global::Feather.Math.float2" => Float2,
            "global::Feather.Math.float3" => Float3,
            "global::Feather.Math.float4" => Float4,
            "global::Feather.Math.int2" => Int2,
            "global::Feather.Math.int3" => Int3,
            "global::Feather.Math.int4" => Int4,
            "global::Feather.Math.bool2" => Bool2,
            "global::Feather.Math.bool3" => Bool3,
            "global::Feather.Math.bool4" => Bool4,
            "global::Feather.Math.float2x2" => Float2x2,
            "global::Feather.Math.float3x3" => Float3x3,
            "global::Feather.Math.float4x4" => Float4x4,
            "global::Feather.Math.float2x3" => Float2x3,
            "global::Feather.Math.float3x2" => Float3x2,
            "global::Feather.Math.float2x4" => Float2x4,
            "global::Feather.Math.float4x2" => Float4x2,
            "global::Feather.Math.float3x4" => Float3x4,
            "global::Feather.Math.float4x3" => Float4x3,
            _ => FromGpuStruct(type)
        };
    }

    public static ShaderType? FromTypeName(string? typeName)
    {
        if (typeName is not { Length: > 0 }) return null;
        var n = typeName.StartsWith("global::", StringComparison.Ordinal)
            ? typeName.Substring("global::".Length) : typeName;
        return n switch
        {
            "float" or "System.Single" => Float,
            "int" or "System.Int32" => Int,
            "uint" or "System.UInt32" => UInt,
            "bool" or "System.Boolean" => Bool,
            "Feather.Math.float2" => Float2, "Feather.Math.float3" => Float3, "Feather.Math.float4" => Float4,
            "Feather.Math.int2" => Int2, "Feather.Math.int3" => Int3, "Feather.Math.int4" => Int4,
            "Feather.Math.bool2" => Bool2, "Feather.Math.bool3" => Bool3, "Feather.Math.bool4" => Bool4,
            "Feather.Math.float2x2" => Float2x2, "Feather.Math.float3x3" => Float3x3, "Feather.Math.float4x4" => Float4x4,
            "Feather.Math.float2x3" => Float2x3, "Feather.Math.float3x2" => Float3x2,
            "Feather.Math.float2x4" => Float2x4, "Feather.Math.float4x2" => Float4x2,
            "Feather.Math.float3x4" => Float3x4, "Feather.Math.float4x3" => Float4x3,
            _ => null
        };
    }

    public static string ToGlslTypeName(ShaderType type) => type switch
    {
        ShaderPrimitiveType p => p.Kind switch
        {
            ShaderPrimitiveKind.Bool => "bool", ShaderPrimitiveKind.Int => "int",
            ShaderPrimitiveKind.UInt => "uint", _ => "float"
        },
        ShaderVectorType { ElementType.Kind: ShaderPrimitiveKind.Bool } v => "bvec" + v.ComponentCount.ToString(),
        ShaderVectorType { ElementType.Kind: ShaderPrimitiveKind.Int } v => "ivec" + v.ComponentCount.ToString(),
        ShaderVectorType { ElementType.Kind: ShaderPrimitiveKind.UInt } v => "uvec" + v.ComponentCount.ToString(),
        ShaderVectorType v => "vec" + v.ComponentCount.ToString(),
        ShaderMatrixType m => ToGlslTypeName(m.ElementType) + m.Rows + "x" + m.Columns,
        ShaderVoidType => "void", ShaderStructType s => s.Name, _ => "float"
    };

    private static ShaderType? FromGpuStruct(ITypeSymbol type)
    {
        if (type is not INamedTypeSymbol named
            || !GpuStructLayoutRules.IsGpuStruct(named)
            || !GpuStructLayoutRules.IsValidGpuStructLayout(named, requirePartial: true))
        {
            return null;
        }

        var fields = new List<ShaderStructField>();
        var offset = 0;
        foreach (var field in GpuStructFieldDiscovery.GetFields(named))
        {
            var fieldType = FromTypeSymbol(field.Type);
            var layout = GetLayout(field.Type, fieldType);
            if (fieldType is null || layout.SizeInBytes <= 0 || layout.Alignment <= 0)
            {
                return null;
            }

            offset = Align(offset, layout.Alignment);
            var flags = ShaderStructFieldFlags.None;
            foreach (var attr in field.Symbol.GetAttributes())
            {
                var attributeName = attr.AttributeClass?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
                if (attributeName == "global::Feather.PositionAttribute")
                {
                    flags |= ShaderStructFieldFlags.Position;
                }
                else if (attributeName == "global::Feather.ColorAttribute")
                {
                    var index = attr.ConstructorArguments.Length > 0 && attr.ConstructorArguments[0].Value is uint colorIndex
                        ? colorIndex
                        : 0u;
                    flags |= ShaderStructFieldFlags.Color |
                        (ShaderStructFieldFlags)(index << (int)ShaderStructFieldFlags.ColorIndexShift);
                }
            }
            fields.Add(new ShaderStructField(field.Name, fieldType, offset, layout.SizeInBytes, flags));
            offset += layout.SizeInBytes;
        }

        var alignment = fields.Count == 0 ? 1 : fields.Max(static field => GetShaderTypeAlignment(field.Type));
        var size = fields.Count == 0 ? 0 : Align(fields.Max(static field => field.Offset + field.SizeInBytes), alignment);
        return new ShaderStructType(named.Name, named.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat),
            new EquatableArray<ShaderStructField>(fields), size, alignment)
        {
            CSharpTypeName = named.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)
        };
    }

    private static TypeLayout GetLayout(ITypeSymbol sourceType, ShaderType? shaderType)
    {
        if (TryGetGpuArrayLayout(sourceType, shaderType, out var arrayLayout))
        {
            return arrayLayout;
        }

        if (sourceType.SpecialType is SpecialType.System_Boolean)
        {
            return new TypeLayout(4, 4);
        }

        if (sourceType.SpecialType is SpecialType.System_Byte or SpecialType.System_SByte)
        {
            return new TypeLayout(1, 1);
        }

        if (sourceType.SpecialType is SpecialType.System_Int16 or SpecialType.System_UInt16)
        {
            return new TypeLayout(2, 2);
        }

        if (shaderType is null)
        {
            return new TypeLayout(0, 0);
        }

        return shaderType switch
        {
            ShaderPrimitiveType => new TypeLayout(4, 4),
            ShaderVectorType { ComponentCount: 2 } => new TypeLayout(8, 8),
            ShaderVectorType { ComponentCount: 3 } => new TypeLayout(12, 16),
            ShaderVectorType { ComponentCount: 4 } => new TypeLayout(16, 16),
            ShaderMatrixType { Rows: 2, Columns: 2 } => new TypeLayout(16, 8),
            ShaderMatrixType { Rows: 3, Columns: 3 } => new TypeLayout(48, 16),
            ShaderMatrixType { Rows: 4, Columns: 4 } => new TypeLayout(64, 16),
            ShaderStructType s => new TypeLayout(s.SizeInBytes, s.Alignment),
            ShaderArrayType array => new TypeLayout(
                checked(GetShaderTypeArrayStride(array.ElementType) * (array.Length ?? 0)),
                GetShaderTypeAlignment(array.ElementType)),
            _ => new TypeLayout(0, 0)
        };
    }

    private static bool TryGetGpuArrayLayout(ITypeSymbol sourceType, ShaderType? shaderType, out TypeLayout layout)
    {
        layout = default;
        if (sourceType is not INamedTypeSymbol { IsGenericType: true } named ||
            !TryGetGpuArrayLength(named, out _))
        {
            return false;
        }

        if (shaderType is not ShaderArrayType { Length: > 0 } array ||
            IsGpuArrayType(named.TypeArguments[0]))
        {
            layout = new TypeLayout(0, 0);
            return true;
        }

        var elementLayout = GetLayout(named.TypeArguments[0], array.ElementType);
        if (elementLayout.SizeInBytes <= 0 || elementLayout.Alignment <= 0)
        {
            layout = new TypeLayout(0, 0);
            return true;
        }

        var stride = Align(elementLayout.SizeInBytes, elementLayout.Alignment);
        layout = new TypeLayout(checked(stride * array.Length.Value), elementLayout.Alignment);
        return true;
    }

    private static int GetShaderTypeAlignment(ShaderType type)
        => type switch
        {
            ShaderPrimitiveType { CSharpTypeName: "byte" or "sbyte" } => 1,
            ShaderPrimitiveType { CSharpTypeName: "short" or "ushort" } => 2,
            ShaderPrimitiveType => 4,
            ShaderVectorType { ComponentCount: 2 } => 8,
            ShaderVectorType { ComponentCount: 3 or 4 } => 16,
            ShaderMatrixType { Rows: 2, Columns: 2 } => 8,
            ShaderMatrixType => 16,
            ShaderStructType s => s.Alignment,
            ShaderArrayType array => GetShaderTypeAlignment(array.ElementType),
            _ => 0
        };

    private static int GetShaderTypeArrayStride(ShaderType type)
    {
        var size = type switch
        {
            ShaderPrimitiveType { CSharpTypeName: "byte" or "sbyte" } => 1,
            ShaderPrimitiveType { CSharpTypeName: "short" or "ushort" } => 2,
            ShaderPrimitiveType => 4,
            ShaderVectorType { ComponentCount: 2 } => 8,
            ShaderVectorType { ComponentCount: 3 } => 12,
            ShaderVectorType { ComponentCount: 4 } => 16,
            ShaderMatrixType { Rows: 2, Columns: 2 } => 16,
            ShaderMatrixType { Rows: 3, Columns: 3 } => 48,
            ShaderMatrixType { Rows: 4, Columns: 4 } => 64,
            ShaderStructType s => s.SizeInBytes,
            ShaderArrayType array => checked(GetShaderTypeArrayStride(array.ElementType) * (array.Length ?? 0)),
            _ => 0
        };
        var alignment = GetShaderTypeAlignment(type);
        return size <= 0 || alignment <= 0 ? 0 : Align(size, alignment);
    }

    private static bool IsGpuArrayType(ITypeSymbol type)
        => type is INamedTypeSymbol { IsGenericType: true } named && TryGetGpuArrayLength(named, out _);

    private static bool TryGetGpuArrayLength(INamedTypeSymbol type, out int length)
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

    private static int Align(int value, int alignment)
    {
        if (alignment <= 1)
        {
            return value;
        }

        var remainder = value % alignment;
        return remainder == 0 ? value : checked(value + alignment - remainder);
    }

    private readonly record struct TypeLayout(int SizeInBytes, int Alignment);
}
