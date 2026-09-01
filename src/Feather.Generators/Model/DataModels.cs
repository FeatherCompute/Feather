using System.Collections.Immutable;
using Microsoft.CodeAnalysis;

namespace Feather.Generators.Model;

internal sealed record DataTypeModel(
    string Guid,
    string TypeName,
    string DisplayName,
    ushort ContractMajor,
    ushort ContractMinor,
    bool IsTopLevel,
    bool IsGeneric,
    string SourcePath,
    string SourceHash,
    int SourceLine,
    int SourceColumn,
    ImmutableArray<DataResourceModel> Resources,
    Location Location);

internal sealed record DataResourceModel(
    string Guid,
    string Symbol,
    string Name,
    string Kind,
    int Access,
    string ElementType,
    int ElementSizeBytes,
    int ElementAlignmentBytes,
    string ElementLayoutIdentity,
    int Creation,
    int Update,
    int Lifetime,
    int Frames,
    long MaximumBytes,
    long ElementCount,
    int Width,
    int Height,
    int Depth,
    string? Format,
    Location Location);

