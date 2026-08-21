using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using Feather.Generators.Model;
using Microsoft.CodeAnalysis;

namespace Feather.Generators.IR;

/// <summary>
/// Emits the generator-owned half of Feather's source provenance contract. The payload is kept
/// beside the generated FEIR instead of changing the FEIR binary ABI: Studio can bind the source
/// path to an exact document version/hash while ordinary Feather runtimes remain unaffected.
/// Character offsets and lengths are Roslyn <see cref="Microsoft.CodeAnalysis.Text.TextSpan"/>
/// values and therefore use UTF-16 code units, matching LSP and Monaco.
/// </summary>
internal static class FeatherIrSourceMapWriter
{
    public const int SchemaVersion = 1;

    public static string ToBase64(ShaderModel model, FeatherIrEmission emission)
        => Convert.ToBase64String(Encoding.UTF8.GetBytes(WriteJson(model, emission)));

    private static string WriteJson(ShaderModel model, FeatherIrEmission emission)
    {
        var entryPoint = model.EntryPointSyntax;
        string entrySymbol = SymbolIdentity(model.EntryPointSymbol);
        string typeSymbol = SymbolIdentity(model.Symbol);
        string sourcePath = entryPoint?.SyntaxTree.FilePath ?? model.Syntax.SyntaxTree.FilePath;
        var span = entryPoint?.Span ?? model.Syntax.Span;

        var json = new StringBuilder(512 + emission.Instructions.Count * 64);
        json.Append('{');
        json.Append("\"schemaVersion\":").Append(SchemaVersion).Append(',');
        json.Append("\"kind\":\"Feather.FeirSourceMap\",");
        json.Append("\"feirSha256\":");
        AppendString(json, Sha256(emission.Module));
        json.Append(',');
        json.Append("\"sourceType\":");
        AppendString(json, NormalizeTypeName(model.FullyQualifiedMetadataName));
        json.Append(',');
        json.Append("\"sourceTypeIdentity\":");
        AppendString(json, typeSymbol);
        json.Append(',');
        json.Append("\"stage\":");
        AppendString(json, Stage(model.Kind));
        json.Append(',');
        json.Append("\"sourcePath\":");
        AppendString(json, sourcePath);
        json.Append(',');
        json.Append("\"entryPoint\":{");
        json.Append("\"symbolIdentity\":");
        AppendString(json, entrySymbol);
        json.Append(',');
        json.Append("\"span\":{");
        json.Append("\"start\":").Append(span.Start).Append(',');
        json.Append("\"length\":").Append(span.Length);
        json.Append("}},");
        json.Append("\"instructions\":[");
        for (int index = 0; index < emission.Instructions.Count; index++)
        {
            if (index > 0) json.Append(',');
            var instruction = emission.Instructions[index];
            json.Append('{');
            json.Append("\"instructionIndex\":").Append(instruction.InstructionIndex).Append(',');
            json.Append("\"span\":{");
            json.Append("\"start\":").Append(instruction.SyntaxStart).Append(',');
            json.Append("\"length\":").Append(instruction.SyntaxLength);
            json.Append("}}");
        }
        json.Append("]}");
        return json.ToString();
    }

    private static string SymbolIdentity(ISymbol? symbol)
    {
        if (symbol is null) return string.Empty;
        return DocumentationCommentId.CreateDeclarationId(symbol)
            ?? symbol.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
    }

    private static string Stage(ShaderKind kind) => kind switch
    {
        ShaderKind.Compute1D or ShaderKind.Compute2D or ShaderKind.Compute3D => "COMPUTE",
        ShaderKind.Vertex => "VERTEX",
        ShaderKind.Fragment => "FRAGMENT",
        _ => "UNKNOWN",
    };

    private static string NormalizeTypeName(string value)
        => value.StartsWith("global::", StringComparison.Ordinal)
            ? value.Substring("global::".Length)
            : value;

    private static string Sha256(ReadOnlySpan<byte> bytes)
    {
        using var hash = SHA256.Create();
        return BitConverter.ToString(hash.ComputeHash(bytes.ToArray()))
            .Replace("-", string.Empty)
            .ToLowerInvariant();
    }

    private static void AppendString(StringBuilder builder, string value)
    {
        builder.Append('"');
        foreach (char character in value)
        {
            switch (character)
            {
                case '"': builder.Append("\\\""); break;
                case '\\': builder.Append("\\\\"); break;
                case '\b': builder.Append("\\b"); break;
                case '\f': builder.Append("\\f"); break;
                case '\n': builder.Append("\\n"); break;
                case '\r': builder.Append("\\r"); break;
                case '\t': builder.Append("\\t"); break;
                default:
                    if (character < ' ')
                    {
                        builder.Append("\\u");
                        builder.Append(((int)character).ToString("x4", CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        builder.Append(character);
                    }
                    break;
            }
        }
        builder.Append('"');
    }
}

internal sealed record FeatherIrEmission(
    byte[] Module,
    IReadOnlyList<FeatherIrInstructionOrigin> Instructions);

internal readonly record struct FeatherIrInstructionOrigin(
    uint InstructionIndex,
    int SyntaxStart,
    int SyntaxLength);
