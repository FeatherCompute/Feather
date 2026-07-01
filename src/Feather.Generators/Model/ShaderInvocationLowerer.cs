using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Operations;

namespace Feather.Generators.Model;

internal static class ShaderInvocationLowerer
{
    public static bool TryLowerKnownInvocation(IInvocationOperation invocation, out LoweredShaderInstructionKind kind)
    {
        kind = default;
        var containingType = invocation.TargetMethod.ContainingType?.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        var methodName = invocation.TargetMethod.Name;
        kind = containingType switch
        {
            "global::Feather.GpuBarrier" => methodName switch
            {
                "Workgroup" => LoweredShaderInstructionKind.WorkgroupBarrier,
                "Memory" => LoweredShaderInstructionKind.MemoryBarrier,
                "Full" => LoweredShaderInstructionKind.FullBarrier,
                _ => default
            },
            "global::Feather.GpuAtomic" => methodName switch
            {
                "Add" => LoweredShaderInstructionKind.AtomicAdd,
                "Sub" => LoweredShaderInstructionKind.AtomicSub,
                "Min" => LoweredShaderInstructionKind.AtomicMin,
                "Max" => LoweredShaderInstructionKind.AtomicMax,
                "And" => LoweredShaderInstructionKind.AtomicAnd,
                "Or" => LoweredShaderInstructionKind.AtomicOr,
                "Xor" => LoweredShaderInstructionKind.AtomicXor,
                "Exchange" => LoweredShaderInstructionKind.AtomicExchange,
                "CompareExchange" => LoweredShaderInstructionKind.AtomicCompareExchange,
                _ => default
            },
            "global::Feather.AD.AD" => methodName switch
            {
                "Parameter" => LoweredShaderInstructionKind.AdParameter,
                "Loss" => LoweredShaderInstructionKind.AdLoss,
                _ => LoweredShaderInstructionKind.KnownSymbolInvocation
            },
            "global::Feather.Math.ShaderMath" or "global::Feather.Math.Hlsl" => LoweredShaderInstructionKind.KnownSymbolInvocation,
            _ => default
        };

        return kind != default;
    }
}
