using Feather.Math;

namespace Feather.RayTracing;

/// <summary>
/// A ray for hardware-accelerated traversal: origin, direction, and the
/// parametric range [TMin, TMax] over which hits are accepted.
/// </summary>
[Feather.GpuStruct]
public partial struct Ray
{
    public float3 Origin;
    public float3 Direction;
    public float TMin;
    public float TMax;

    public Ray(float3 origin, float3 direction, float tMin, float tMax)
    {
        Origin = origin;
        Direction = direction;
        TMin = tMin;
        TMax = tMax;
    }

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
[Feather.GpuStruct]
public partial struct SurfaceHit
{
    public uint Inst;
    public uint Prim;
    public float2 Bary;
    public float T;

    public SurfaceHit(uint inst, uint prim, float2 bary, float t)
    {
        Inst = inst;
        Prim = prim;
        Bary = bary;
        T = t;
    }
}
