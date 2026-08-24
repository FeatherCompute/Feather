using System.Collections.Immutable;
using Microsoft.CodeAnalysis;

namespace Feather.Generators.Model;

internal sealed record AssetInputModel(
    string Guid,
    string SymbolName,
    string Name,
    string? Group,
    int Order,
    bool Required,
    long Role,
    string ChangeImpact,
    string ValueKind,
    string TypeName,
    string? ReferencedAssetTypeName,
    double? Minimum,
    double? Maximum,
    double? Step,
    int? MinimumItems,
    int? MaximumItems,
    int? MaximumLength,
    bool IsImmutable,
    Location Location);

internal sealed record AssetCapabilityUseModel(
    string ContractTypeName,
    string? Guid,
    ushort MinimumMajor,
    ushort MinimumMinor,
    bool Required,
    Location Location);

internal sealed record AssetOutputModel(
    string Guid,
    string ContractTypeName,
    string? ContractGuid,
    ushort ContractMajor,
    ushort ContractMinor,
    string Symbol,
    string Name,
    bool Required,
    bool GraphOutput,
    long PassDirections,
    Location Location);

internal sealed record AssetTypeModel(
    string Guid,
    string Namespace,
    string Name,
    string TypeName,
    string DisplayName,
    string? Description,
    ushort ContractMajor,
    ushort ContractMinor,
    int PayloadSchemaVersion,
    bool DeclaredAbstract,
    bool IsAbstract,
    bool IsTopLevel,
    bool IsGeneric,
    bool IsPartial,
    bool DerivesAsset,
    string? BaseTypeName,
    string? BaseTypeGuid,
    string SourcePath,
    int SourceLine,
    int SourceColumn,
    ImmutableArray<AssetInputModel> Inputs,
    ImmutableArray<AssetCapabilityUseModel> Capabilities,
    ImmutableArray<AssetOutputModel> Outputs,
    Location Location);

internal enum AssetContractKind
{
    Capability,
    Output,
}

internal sealed record AssetContractModel(
    AssetContractKind Kind,
    string Guid,
    string TypeName,
    string Name,
    ushort ContractMajor,
    ushort ContractMinor,
    bool IsTopLevel,
    bool IsGeneric,
    bool ImplementsMarker,
    string SourcePath,
    int SourceLine,
    int SourceColumn,
    Location Location);

internal sealed record AssetProviderModel(
    string Guid,
    string TypeName,
    string Name,
    string Operation,
    ushort ContractMajor,
    ushort ContractMinor,
    string Owner,
    string Determinism,
    ImmutableArray<string> PrimaryOperations,
    ImmutableArray<string> AssetTypeNames,
    ImmutableArray<string> OutputContractTypeNames,
    string SourcePath,
    int SourceLine,
    int SourceColumn,
    Location Location);
