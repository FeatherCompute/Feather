using System.Collections.Immutable;
using Microsoft.CodeAnalysis;

namespace Feather.Generators.Model;

internal sealed record PassSocketModel(
    string Guid,
    string Name,
    string ResourceKind,
    string? ElementType,
    string Format,
    string Access,
    PassSocketContractModel Contract,
    Location Location);

internal sealed record PassSocketContractModel(
    string Kind,
    string? RequiredAssetTypeId,
    ushort RequiredAssetTypeMajor,
    ushort RequiredAssetTypeMinor,
    ImmutableArray<PassSocketCapabilityModel> RequiredCapabilities,
    string? ProductSlotId,
    string? OutputContractId,
    ushort OutputContractMajor,
    ushort OutputContractMinor,
    bool AdapterRequired);

internal sealed record PassSocketCapabilityModel(
    string CapabilityId,
    ushort MinimumMajor,
    ushort MinimumMinor,
    bool Required);

internal sealed record PassParameterModel(
    string Guid,
    string Name,
    string Type,
    string LogicalType,
    string? TypeIdentity,
    string? DefaultValueJson,
    double? Min,
    double? Max,
    double? Step,
    string? Unit,
    string? Description,
    string? Group,
    int Order,
    string? EditorHint,
    string Mutability,
    ImmutableArray<string> Bindings,
    string Redaction,
    PassParameterRuntimeAbiModel RuntimeAbi,
    PassEnumModel? Enum,
    Location Location);

internal sealed record PassParameterRuntimeAbiModel(
    string StorageClass,
    string ScalarKind,
    int Offset,
    int Size,
    int Alignment,
    string Packing);

internal sealed record PassEnumMemberModel(
    string Guid,
    string SymbolName,
    long NumericValue,
    string DisplayName,
    string? Description,
    int Order,
    bool Deprecated,
    string? ReplacementMemberGuid,
    Location Location);

internal sealed record PassEnumModel(
    string TypeGuid,
    string TypeName,
    string UnderlyingScalar,
    bool IsFlags,
    bool AllowUnknownNumeric,
    bool AllowUnknownBits,
    long DefaultRawValue,
    long AllowedMask,
    ImmutableArray<PassEnumMemberModel> Members,
    Location Location);

internal sealed record PassModel(
    string Guid,
    string TypeName,
    string DisplayName,
    string Category,
    int Version,
    string SourcePath,
    int SourceLine,
    int SourceColumn,
    bool ImplementsRenderPass,
    ImmutableArray<PassSocketModel> Inputs,
    ImmutableArray<PassSocketModel> Outputs,
    ImmutableArray<PassParameterModel> Parameters,
    Location Location);
