// Injected into consuming projects as SOURCE by FeatherCompute.targets. See FeatherCamera.cs for
// why a shared shader helper cannot ship as a compiled assembly.

using Feather.Math;

namespace Feather.Shaders;

/// <summary>
/// The scalar operations a procedural shader reaches for that <see cref="ShaderMath"/> does not
/// carry.
/// </summary>
/// <remarks>
/// Written from the primitives the generator already lowers rather than added as new intrinsics,
/// which keeps a new effect from needing a compiler change to build. Trigonometric and exponential
/// intrinsics only accept floats, so the vector forms are unrolled by hand.
/// </remarks>
[ShaderLibrary]
public static class FeatherMathEx
{
    /// <summary>The non-negative remainder of <paramref name="x"/> divided by <paramref name="m"/>.</summary>
    /// <remarks>
    /// Floored rather than truncated, matching GLSL's <c>mod</c>: the result keeps the sign of the
    /// divisor, so tiling a pattern across a negative coordinate does not mirror at the origin.
    /// </remarks>
    [Callable]
    public static float Mod(float x, float m)
    {
        return x - (m * ShaderMath.Floor(x / m));
    }

    /// <summary>Rounds toward zero.</summary>
    [Callable]
    public static float Trunc(float x)
    {
        var magnitude = ShaderMath.Floor(ShaderMath.Abs(x));
        return x < 0.0f ? -magnitude : magnitude;
    }

    /// <summary>Returns -1, 0, or 1 according to the sign of <paramref name="x"/>.</summary>
    [Callable]
    public static float Sign(float x)
    {
        if (x > 0.0f)
        {
            return 1.0f;
        }

        return x < 0.0f ? -1.0f : 0.0f;
    }

    /// <summary>Zero below <paramref name="edge"/>, one at or above it.</summary>
    [Callable]
    public static float Step(float edge, float x)
    {
        return x < edge ? 0.0f : 1.0f;
    }

    /// <summary>Two raised to <paramref name="x"/>.</summary>
    [Callable]
    public static float Exp2(float x)
    {
        return ShaderMath.Exp(x * 0.693147181f);
    }

    /// <summary>The base-two logarithm of <paramref name="x"/>.</summary>
    [Callable]
    public static float Log2(float x)
    {
        return ShaderMath.Log(x) * 1.442695041f;
    }

    /// <summary>The distance between two points.</summary>
    [Callable]
    public static float Distance(float3 a, float3 b)
    {
        return ShaderMath.Length(a - b);
    }
}

/// <summary>
/// Value noise, and the hash it is built on.
/// </summary>
/// <remarks>
/// Deliberately hash-based rather than texture-based: an effect that samples a noise texture needs
/// that texture wired into the graph, whereas this stays a pure function of position and so keeps a
/// render reproducible frame to frame. That reproducibility is what the acceptance checks rely on.
/// </remarks>
[ShaderLibrary]
public static class FeatherNoise
{
    /// <summary>A scalar pseudo-random value in [0, 1) from a two-dimensional seed.</summary>
    /// <remarks>
    /// The sine-fract construction is the standard trick. It is not a good hash in the statistical
    /// sense, but it is stable across drivers, which matters more here.
    /// </remarks>
    [Callable]
    public static float Hash(float2 position)
    {
        var dotted = ShaderMath.Dot(position, new float2(127.1f, 311.7f));
        return ShaderMath.Fract(ShaderMath.Sin(dotted) * 43758.5453123f);
    }

    /// <summary>
    /// Value noise in [-1, 1], smooth enough to differentiate.
    /// </summary>
    /// <remarks>
    /// Corners are interpolated with the smoothstep polynomial rather than linearly, so the
    /// gradient is continuous across cell boundaries. Linear interpolation leaves visible creases
    /// once the result is used to displace a surface.
    /// </remarks>
    [Callable]
    public static float Value(float2 position)
    {
        var cell = ShaderMath.Floor(position);
        var offset = position - cell;
        var weight = offset * offset * (new float2(3.0f, 3.0f) - (offset * 2.0f));

        var corner00 = Hash(cell);
        var corner10 = Hash(cell + new float2(1.0f, 0.0f));
        var corner01 = Hash(cell + new float2(0.0f, 1.0f));
        var corner11 = Hash(cell + new float2(1.0f, 1.0f));

        var bottom = ShaderMath.Mix(corner00, corner10, weight.X);
        var top = ShaderMath.Mix(corner01, corner11, weight.X);
        return (ShaderMath.Mix(bottom, top, weight.Y) * 2.0f) - 1.0f;
    }
}
