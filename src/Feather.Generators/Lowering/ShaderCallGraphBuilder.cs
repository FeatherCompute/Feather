using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Operations;

namespace Feather.Generators.Model;

/// <summary>
/// Discovers [Callable] methods, builds a call graph, detects recursion,
/// and topologically sorts callables for emission.
/// </summary>
internal static class ShaderCallGraphBuilder
{
    public sealed record DiscoveredCallable(
        MethodDeclarationSyntax Syntax,
        IMethodSymbol Symbol,
        string Name,
        string MangledName,
        ITypeSymbol ReturnType,
        EquatableArray<CallableParam> Parameters);

    public sealed record CallableParam(string Name, ITypeSymbol Type, bool IsRef);

    public sealed record CallGraph(
        EquatableArray<DiscoveredCallable> Callables,
        EquatableArray<string> SortedOrder,
        EquatableArray<string> RecursionErrors);

    /// <summary>
    /// Discovers and validates all [Callable] / [ShaderFunction] methods in the given struct.
    /// Returns a topologically sorted call graph, or diagnostics for recursion.
    /// </summary>
    public static CallGraph Build(StructDeclarationSyntax structSyntax, INamedTypeSymbol structSymbol,
        SemanticModel semanticModel, CancellationToken ct)
    {
        var callables = new List<DiscoveredCallable>();
        var adjacency = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        var callableSet = new HashSet<string>(StringComparer.Ordinal);

        // Discover
        foreach (var method in structSyntax.Members.OfType<MethodDeclarationSyntax>())
        {
            var sym = structSymbol.GetMembers(method.Identifier.ValueText)
                .OfType<IMethodSymbol>()
                .FirstOrDefault(candidate => candidate.DeclaringSyntaxReferences.Any(reference => reference.GetSyntax() == method));
            if (sym is null) continue;
            if (IsEntry(sym)) continue;
            if (!IsCallable(sym)) continue;

            ct.ThrowIfCancellationRequested();
            var name = sym.Name;
            var mangled = Mangle(name, sym);
            var pars = sym.Parameters.Select(p =>
                new CallableParam(p.Name, p.Type,
                    p.RefKind is RefKind.Ref or RefKind.Out or RefKind.In)).ToArray();

            callables.Add(new DiscoveredCallable(method, sym, name, mangled, sym.ReturnType,
                new EquatableArray<CallableParam>(pars)));
            callableSet.Add(name);
            adjacency[name] = new List<string>();
        }

        // Build call edges: for each callable body, find callable invocations
        foreach (var c in callables)
        {
            SyntaxNode? root = c.Syntax.Body;
            if (root is null)
            {
                root = c.Syntax.ExpressionBody?.Expression;
            }

            if (root is null) continue;
            foreach (var inv in root.DescendantNodesAndSelf().OfType<InvocationExpressionSyntax>())
            {
                ct.ThrowIfCancellationRequested();
                var op = semanticModel.GetOperation(inv, ct) as IInvocationOperation;
                var target = op?.TargetMethod;
                if (target is null) continue;
                if (!IsCallable(target)) continue;
                if (callableSet.Contains(target.Name))
                    adjacency[c.Name].Add(target.Name);
            }
        }

        // Detect recursion and topologically sort
        var sorted = new List<string>();
        var visiting = new HashSet<string>(StringComparer.Ordinal);
        var visited = new HashSet<string>(StringComparer.Ordinal);
        var recursion = new List<string>();

        foreach (var name in callables.Select(c => c.Name))
            if (!visited.Contains(name))
                TopoSort(name, adjacency, visiting, visited, sorted, recursion);

        sorted.Reverse();
        return new CallGraph(
            new EquatableArray<DiscoveredCallable>(callables),
            new EquatableArray<string>(sorted),
            new EquatableArray<string>(recursion));
    }

    private static void TopoSort(string node, Dictionary<string, List<string>> adj,
        HashSet<string> visiting, HashSet<string> visited,
        List<string> sorted, List<string> recursion)
    {
        if (visiting.Contains(node)) { recursion.Add(node); return; }
        if (visited.Contains(node)) return;
        visiting.Add(node);
        foreach (var neighbor in adj[node])
            TopoSort(neighbor, adj, visiting, visited, sorted, recursion);
        visiting.Remove(node);
        visited.Add(node);
        sorted.Add(node);
    }

    public static string Mangle(string name, IMethodSymbol sym)
    {
        var sb = new System.Text.StringBuilder();
        sb.Append(name);
        foreach (var p in sym.Parameters)
        {
            sb.Append('_');
            sb.Append(p.Type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)
                .Replace("global::", "").Replace('.', '_').Replace('<', '_').Replace('>', '_'));
        }
        return sb.ToString();
    }

    private static bool IsCallable(IMethodSymbol m)
        => m.GetAttributes().Any(a =>
            a.AttributeClass?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)
                is "global::Feather.CallableAttribute" or "global::Feather.ShaderFunctionAttribute");

    private static bool IsEntry(IMethodSymbol m)
        => m.GetAttributes().Any(a =>
            a.AttributeClass?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)
                is "global::Feather.EntryAttribute");
}
