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
    Location Location);

internal sealed record PassParameterModel(
    string Guid,
    string Name,
    string Type,
    string? DefaultValueJson,
    double? Min,
    double? Max,
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
