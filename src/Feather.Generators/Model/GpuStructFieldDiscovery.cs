using Microsoft.CodeAnalysis;

namespace Feather.Generators.Model;

internal static class GpuStructFieldDiscovery
{
    public static IReadOnlyList<GpuStructMember> GetFields(INamedTypeSymbol symbol)
    {
        var result = new List<GpuStructMember>();
        var includedProperties = new HashSet<IPropertySymbol>(SymbolEqualityComparer.Default);
        var recordProperties = GetRecordPrimaryConstructorProperties(symbol);

        foreach (var property in recordProperties)
        {
            result.Add(new GpuStructMember(property.Name, property.Type, property.Name, property, IsRecordPrimaryConstructorProperty: true));
            includedProperties.Add(property);
        }

        foreach (var member in symbol.GetMembers())
        {
            switch (member)
            {
                case IFieldSymbol { IsStatic: false, IsConst: false, IsImplicitlyDeclared: false } field:
                    result.Add(new GpuStructMember(field.Name, field.Type, field.Name, field, IsRecordPrimaryConstructorProperty: false));
                    break;
                case IPropertySymbol { IsStatic: false, IsIndexer: false } property
                    when IsRecordPrimaryConstructorProperty(property) && includedProperties.Add(property):
                    result.Add(new GpuStructMember(property.Name, property.Type, property.Name, property, IsRecordPrimaryConstructorProperty: true));
                    break;
            }
        }

        return result;
    }

    public static IReadOnlyList<GpuStructMember> GetUnsupportedProperties(INamedTypeSymbol symbol)
        => symbol.GetMembers()
            .OfType<IPropertySymbol>()
            .Where(static property => !property.IsStatic
                && !property.IsIndexer
                && !IsRecordPrimaryConstructorProperty(property))
            .Select(static property => new GpuStructMember(property.Name, property.Type, property.Name, property, IsRecordPrimaryConstructorProperty: false))
            .ToArray();

    public static bool TryGetFieldIndex(ISymbol symbol, out int fieldIndex, out string fieldTypeName)
    {
        fieldIndex = -1;
        fieldTypeName = string.Empty;

        var containingType = symbol.ContainingType;
        if (containingType is null || !ShaderSemanticFacts.IsGpuStructType(containingType))
        {
            return false;
        }

        var fields = GetFields(containingType);
        for (var i = 0; i < fields.Count; i++)
        {
            if (SymbolEqualityComparer.Default.Equals(fields[i].Symbol, symbol))
            {
                fieldIndex = i;
                fieldTypeName = ShaderSemanticFacts.GetTypeName(fields[i].Type);
                return true;
            }
        }

        return false;
    }

    private static IReadOnlyList<IPropertySymbol> GetRecordPrimaryConstructorProperties(INamedTypeSymbol symbol)
    {
        var properties = new List<IPropertySymbol>();
        var constructor = symbol.InstanceConstructors
            .Where(static ctor => !ctor.IsImplicitlyDeclared && ctor.Parameters.Length > 0)
            .OrderBy(static ctor => ctor.Locations.FirstOrDefault()?.SourceSpan.Start ?? int.MaxValue)
            .FirstOrDefault();
        if (constructor is null)
        {
            return properties;
        }

        foreach (var parameter in constructor.Parameters)
        {
            var property = symbol.GetMembers(parameter.Name)
                .OfType<IPropertySymbol>()
                .FirstOrDefault(property => IsRecordPrimaryConstructorProperty(property, requireConstructorParameter: false));
            if (property is not null)
            {
                properties.Add(property);
            }
        }

        return properties;
    }

    private static bool IsRecordPrimaryConstructorProperty(IPropertySymbol property)
        => IsRecordPrimaryConstructorProperty(property, requireConstructorParameter: true);

    private static bool IsRecordPrimaryConstructorProperty(IPropertySymbol property, bool requireConstructorParameter)
        => property is { IsStatic: false, IsIndexer: false }
            && property.ContainingType.IsRecord
            && HasAssociatedBackingField(property)
            && (!requireConstructorParameter || IsPrimaryConstructorParameterName(property.ContainingType, property.Name));

    private static bool IsPrimaryConstructorParameterName(INamedTypeSymbol symbol, string name)
        => symbol.InstanceConstructors
            .Where(static ctor => !ctor.IsImplicitlyDeclared && ctor.Parameters.Length > 0)
            .OrderBy(static ctor => ctor.Locations.FirstOrDefault()?.SourceSpan.Start ?? int.MaxValue)
            .FirstOrDefault()
            ?.Parameters
            .Any(parameter => parameter.Name == name) == true;

    private static bool HasAssociatedBackingField(IPropertySymbol property)
        => property.ContainingType.GetMembers()
            .OfType<IFieldSymbol>()
            .Any(field => field.IsImplicitlyDeclared
                && SymbolEqualityComparer.Default.Equals(field.AssociatedSymbol, property));
}

internal readonly record struct GpuStructMember(
    string Name,
    ITypeSymbol Type,
    string MemberAccessor,
    ISymbol Symbol,
    bool IsRecordPrimaryConstructorProperty)
{
    public bool IsReadonlyExplicitField => Symbol is IFieldSymbol { IsReadOnly: true };
}
