using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Feather.Generators.Model;

internal sealed record GpuStructModel(
    TypeDeclarationSyntax Syntax,
    INamedTypeSymbol Symbol,
    string Namespace,
    string Name,
    string FullyQualifiedMetadataName,
    bool IsRecord,
    string LayoutName,
    EquatableArray<GpuStructFieldModel> Fields,
    int SizeInBytes,
    int Alignment,
    bool RequiresRepacking);

internal sealed record GpuStructFieldModel(
    string Name,
    string TypeName,
    string? ArrayElementTypeName,
    int Offset,
    int SizeInBytes,
    int Alignment,
    string MemberAccessor,
    int ArrayLength,
    int ArrayStride,
    bool IsConstructorParameter,
    bool IsReadonlyExplicitField,
    bool UsesUnsupportedMatrixType = false);
