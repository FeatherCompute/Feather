using Feather.Math;

namespace Feather.RayTracing;

/// <summary>
/// A ray for hardware-accelerated traversal: origin, direction, and the
/// parametric range [TMin, TMax] over which hits are accepted.
/// </summary>
public readonly record struct Ray(float3 Origin, float3 Direction, float TMin, float TMax)
{
    /// <summary>
    /// Creates a ray over the default [0, +inf) range.
    /// </summary>
    public Ray(float3 origin, float3 direction)
        : this(origin, direction, 0.0f, float.PositiveInfinity)
    {
    }
}

/// <summary>
/// The closest committed surface hit along a traced ray.
/// </summary>
public readonly record struct SurfaceHit(uint Inst, uint Prim, float2 Bary, float T);
