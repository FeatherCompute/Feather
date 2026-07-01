using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Operations;

namespace Feather.Generators.Model;

internal static class ShaderResourceElementLowerer
{
    public static string FormatPayload(LoweredResourceElement element)
        => string.Join("|", "RESOURCE1", element.ResourceName, element.IndexName);

    public static bool TryLower(IOperation operation, out LoweredResourceElement element)
    {
        element = default;
        // Roslyn represents shader indexers as property references; lowering through IOperation keeps aliases,
        // generated backing fields, and trivia differences out of the resource binding contract.
        if (!ShaderSemanticFacts.TryUnwrapConversion(operation, out var unwrapped)
            || unwrapped is not IPropertyReferenceOperation property
            || !property.Property.IsIndexer
            || property.Arguments.Length != 1
            || !TryGetResourceName(property.Instance, out var resourceName)
            || !TryGetIndexName(property.Arguments[0].Value, out var indexName, out var indexSymbol))
        {
            return false;
        }

        element = new LoweredResourceElement(resourceName, indexName, indexSymbol);
        return true;
    }

    private static bool TryGetResourceName(IOperation? operation, out string resourceName)
    {
        resourceName = string.Empty;
        if (operation is null || !ShaderSemanticFacts.TryUnwrapConversion(operation, out var unwrapped))
        {
            return false;
        }

        resourceName = unwrapped switch
        {
            IParameterReferenceOperation parameter => parameter.Parameter.Name,
            IFieldReferenceOperation field => field.Field.Name,
            ILocalReferenceOperation local => local.Local.Name,
            _ => string.Empty
        };
        return resourceName.Length > 0 && ShaderSemanticFacts.IsShaderResourceType(unwrapped.Type);
    }

    private static bool TryGetIndexName(IOperation operation, out string indexName, out ISymbol? indexSymbol)
    {
        indexName = string.Empty;
        indexSymbol = null;
        if (!ShaderSemanticFacts.TryUnwrapConversion(operation, out var unwrapped))
        {
            return false;
        }

        switch (unwrapped)
        {
            case ILocalReferenceOperation local:
                indexName = local.Local.Name;
                indexSymbol = local.Local;
                return true;
            case IParameterReferenceOperation parameter:
                indexName = parameter.Parameter.Name;
                indexSymbol = parameter.Parameter;
                return true;
            case ILiteralOperation literal:
                indexName = literal.Syntax.ToString().Trim();
                indexSymbol = null;
                return indexName.Length > 0;
            case IPropertyReferenceOperation property:
                indexName = property.Syntax.ToString().Trim();
                indexSymbol = property.Property;
                return indexName.Length > 0;
            case IFieldReferenceOperation field:
                indexName = field.Field.Name;
                indexSymbol = field.Field;
                return true;
            default:
                return false;
        }
    }
}

internal readonly record struct LoweredResourceElement(string ResourceName, string IndexName, ISymbol? IndexSymbol);
