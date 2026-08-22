namespace Feather.Math;

/// <summary>
/// Provides CPU-equivalent shader math functions for Feather vector, matrix, and scalar types.
/// </summary>
public static class ShaderMath
{
    /// <summary>
    /// Computes the dot product of two vectors.
    /// </summary>
    public static float Dot(float2 a, float2 b) => (a.X * b.X) + (a.Y * b.Y);

    /// <summary>
    /// Computes the dot product of two vectors.
    /// </summary>
    public static float Dot(float3 a, float3 b) => (a.X * b.X) + (a.Y * b.Y) + (a.Z * b.Z);

    /// <summary>
    /// Computes the dot product of two vectors.
    /// </summary>
    public static float Dot(float4 a, float4 b) => (a.X * b.X) + (a.Y * b.Y) + (a.Z * b.Z) + (a.W * b.W);

    /// <summary>
    /// Computes the cross product of two three-component vectors.
    /// </summary>
    /// <summary>Reflects <paramref name="incident"/> about <paramref name="normal"/>.</summary>
    /// <remarks>The normal is assumed to be unit length, matching GLSL's builtin.</remarks>
    public static float2 Reflect(float2 incident, float2 normal)
    {
        var scale = 2.0f * Dot(incident, normal);
        return new float2(incident.X - scale * normal.X, incident.Y - scale * normal.Y);
    }

    /// <inheritdoc cref="Reflect(float2, float2)"/>
    public static float3 Reflect(float3 incident, float3 normal)
    {
        var scale = 2.0f * Dot(incident, normal);
        return new float3(
            incident.X - scale * normal.X,
            incident.Y - scale * normal.Y,
            incident.Z - scale * normal.Z);
    }

    /// <inheritdoc cref="Reflect(float2, float2)"/>
    public static float4 Reflect(float4 incident, float4 normal)
    {
        var scale = 2.0f * Dot(incident, normal);
        return new float4(
            incident.X - scale * normal.X,
            incident.Y - scale * normal.Y,
            incident.Z - scale * normal.Z,
            incident.W - scale * normal.W);
    }

    public static float3 Cross(float3 a, float3 b)
        => new(
            (a.Y * b.Z) - (a.Z * b.Y),
            (a.Z * b.X) - (a.X * b.Z),
            (a.X * b.Y) - (a.Y * b.X));

    /// <summary>
    /// Computes the squared Euclidean length of a vector.
    /// </summary>
    public static float LengthSquared(float2 v) => Dot(v, v);

    /// <summary>
    /// Computes the squared Euclidean length of a vector.
    /// </summary>
    public static float LengthSquared(float3 v) => Dot(v, v);

    /// <summary>
    /// Computes the squared Euclidean length of a vector.
    /// </summary>
    public static float LengthSquared(float4 v) => Dot(v, v);

    /// <summary>
    /// Computes the Euclidean length of a vector.
    /// </summary>
    public static float Length(float2 v) => MathF.Sqrt(LengthSquared(v));

    /// <summary>
    /// Computes the Euclidean length of a vector.
    /// </summary>
    public static float Length(float3 v) => MathF.Sqrt(LengthSquared(v));

    /// <summary>
    /// Computes the Euclidean length of a vector.
    /// </summary>
    public static float Length(float4 v) => MathF.Sqrt(LengthSquared(v));

    /// <summary>
    /// Returns a normalized vector, or zero when the input length is zero.
    /// </summary>
    public static float2 Normalize(float2 v)
    {
        var length = Length(v);
        return length == 0 ? float2.Zero : v / length;
    }

    /// <summary>
    /// Returns a normalized vector, or zero when the input length is zero.
    /// </summary>
    public static float3 Normalize(float3 v)
    {
        var length = Length(v);
        return length == 0 ? float3.Zero : v / length;
    }

    /// <summary>
    /// Returns a normalized vector, or zero when the input length is zero.
    /// </summary>
    public static float4 Normalize(float4 v)
    {
        var length = Length(v);
        return length == 0 ? float4.Zero : v / length;
    }

    /// <summary>
    /// Computes the sine of a scalar.
    /// </summary>
    public static float Sin(float x) => MathF.Sin(x);

    /// <summary>Computes the sine of each component.</summary>
    public static float2 Sin(float2 x) => new(Sin(x.X), Sin(x.Y));

    /// <summary>Computes the sine of each component.</summary>
    public static float3 Sin(float3 x) => new(Sin(x.X), Sin(x.Y), Sin(x.Z));

    /// <summary>Computes the sine of each component.</summary>
    public static float4 Sin(float4 x) => new(Sin(x.X), Sin(x.Y), Sin(x.Z), Sin(x.W));

    /// <summary>
    /// Computes the cosine of a scalar.
    /// </summary>
    public static float Cos(float x) => MathF.Cos(x);

    /// <summary>Computes the cosine of each component.</summary>
    public static float2 Cos(float2 x) => new(Cos(x.X), Cos(x.Y));

    /// <summary>Computes the cosine of each component.</summary>
    public static float3 Cos(float3 x) => new(Cos(x.X), Cos(x.Y), Cos(x.Z));

    /// <summary>Computes the cosine of each component.</summary>
    public static float4 Cos(float4 x) => new(Cos(x.X), Cos(x.Y), Cos(x.Z), Cos(x.W));

    /// <summary>
    /// Computes the tangent of a scalar.
    /// </summary>
    public static float Tan(float x) => MathF.Tan(x);

    /// <summary>Computes the tangent of each component.</summary>
    public static float2 Tan(float2 x) => new(Tan(x.X), Tan(x.Y));

    /// <summary>Computes the tangent of each component.</summary>
    public static float3 Tan(float3 x) => new(Tan(x.X), Tan(x.Y), Tan(x.Z));

    /// <summary>Computes the tangent of each component.</summary>
    public static float4 Tan(float4 x) => new(Tan(x.X), Tan(x.Y), Tan(x.Z), Tan(x.W));

    /// <summary>
    /// Computes the hyperbolic sine of a scalar.
    /// </summary>
    public static float Sinh(float x) => MathF.Sinh(x);

    /// <summary>Computes the hyperbolic sine of each component.</summary>
    public static float2 Sinh(float2 x) => new(Sinh(x.X), Sinh(x.Y));

    /// <summary>Computes the hyperbolic sine of each component.</summary>
    public static float3 Sinh(float3 x) => new(Sinh(x.X), Sinh(x.Y), Sinh(x.Z));

    /// <summary>Computes the hyperbolic sine of each component.</summary>
    public static float4 Sinh(float4 x) => new(Sinh(x.X), Sinh(x.Y), Sinh(x.Z), Sinh(x.W));

    /// <summary>
    /// Computes the hyperbolic cosine of a scalar.
    /// </summary>
    public static float Cosh(float x) => MathF.Cosh(x);

    /// <summary>Computes the hyperbolic cosine of each component.</summary>
    public static float2 Cosh(float2 x) => new(Cosh(x.X), Cosh(x.Y));

    /// <summary>Computes the hyperbolic cosine of each component.</summary>
    public static float3 Cosh(float3 x) => new(Cosh(x.X), Cosh(x.Y), Cosh(x.Z));

    /// <summary>Computes the hyperbolic cosine of each component.</summary>
    public static float4 Cosh(float4 x) => new(Cosh(x.X), Cosh(x.Y), Cosh(x.Z), Cosh(x.W));

    /// <summary>
    /// Computes the hyperbolic tangent of a scalar.
    /// </summary>
    public static float Tanh(float x) => MathF.Tanh(x);

    /// <summary>Computes the hyperbolic tangent of each component.</summary>
    public static float2 Tanh(float2 x) => new(Tanh(x.X), Tanh(x.Y));

    /// <summary>Computes the hyperbolic tangent of each component.</summary>
    public static float3 Tanh(float3 x) => new(Tanh(x.X), Tanh(x.Y), Tanh(x.Z));

    /// <summary>Computes the hyperbolic tangent of each component.</summary>
    public static float4 Tanh(float4 x) => new(Tanh(x.X), Tanh(x.Y), Tanh(x.Z), Tanh(x.W));

    /// <summary>
    /// Computes the base-e exponential of a scalar.
    /// </summary>
    public static float Exp(float x) => MathF.Exp(x);

    /// <summary>Computes the base-e exponential of each component.</summary>
    public static float2 Exp(float2 x) => new(Exp(x.X), Exp(x.Y));

    /// <summary>Computes the base-e exponential of each component.</summary>
    public static float3 Exp(float3 x) => new(Exp(x.X), Exp(x.Y), Exp(x.Z));

    /// <summary>Computes the base-e exponential of each component.</summary>
    public static float4 Exp(float4 x) => new(Exp(x.X), Exp(x.Y), Exp(x.Z), Exp(x.W));

    /// <summary>
    /// Computes the natural logarithm of a scalar.
    /// </summary>
    public static float Log(float x) => MathF.Log(x);

    /// <summary>Computes the natural logarithm of each component.</summary>
    public static float2 Log(float2 x) => new(Log(x.X), Log(x.Y));

    /// <summary>Computes the natural logarithm of each component.</summary>
    public static float3 Log(float3 x) => new(Log(x.X), Log(x.Y), Log(x.Z));

    /// <summary>Computes the natural logarithm of each component.</summary>
    public static float4 Log(float4 x) => new(Log(x.X), Log(x.Y), Log(x.Z), Log(x.W));

    /// <summary>
    /// Raises a scalar to a power.
    /// </summary>
    public static float Pow(float x, float y) => MathF.Pow(x, y);

    /// <summary>Raises each component of <paramref name="x"/> to the corresponding power.</summary>
    public static float2 Pow(float2 x, float2 y) => new(Pow(x.X, y.X), Pow(x.Y, y.Y));

    /// <summary>Raises each component of <paramref name="x"/> to the corresponding power.</summary>
    public static float3 Pow(float3 x, float3 y) => new(Pow(x.X, y.X), Pow(x.Y, y.Y), Pow(x.Z, y.Z));

    /// <summary>Raises each component of <paramref name="x"/> to the corresponding power.</summary>
    public static float4 Pow(float4 x, float4 y) => new(Pow(x.X, y.X), Pow(x.Y, y.Y), Pow(x.Z, y.Z), Pow(x.W, y.W));

    /// <summary>
    /// Computes the square root of a scalar.
    /// </summary>
    public static float Sqrt(float x) => MathF.Sqrt(x);

    /// <summary>Computes the square root of each component.</summary>
    public static float2 Sqrt(float2 x) => new(Sqrt(x.X), Sqrt(x.Y));

    /// <summary>Computes the square root of each component.</summary>
    public static float3 Sqrt(float3 x) => new(Sqrt(x.X), Sqrt(x.Y), Sqrt(x.Z));

    /// <summary>Computes the square root of each component.</summary>
    public static float4 Sqrt(float4 x) => new(Sqrt(x.X), Sqrt(x.Y), Sqrt(x.Z), Sqrt(x.W));

    /// <summary>
    /// Computes the reciprocal square root of a scalar.
    /// </summary>
    public static float InverseSqrt(float x) => 1.0f / MathF.Sqrt(x);

    /// <summary>Computes the reciprocal square root of each component.</summary>
    public static float2 InverseSqrt(float2 x) => new(InverseSqrt(x.X), InverseSqrt(x.Y));

    /// <summary>Computes the reciprocal square root of each component.</summary>
    public static float3 InverseSqrt(float3 x) => new(InverseSqrt(x.X), InverseSqrt(x.Y), InverseSqrt(x.Z));

    /// <summary>Computes the reciprocal square root of each component.</summary>
    public static float4 InverseSqrt(float4 x) => new(InverseSqrt(x.X), InverseSqrt(x.Y), InverseSqrt(x.Z), InverseSqrt(x.W));

    /// <summary>
    /// Computes the absolute value of a scalar.
    /// </summary>
    public static float Abs(float x) => MathF.Abs(x);

    /// <summary>
    /// Computes the absolute value of each component.
    /// </summary>
    public static float2 Abs(float2 x) => new(Abs(x.X), Abs(x.Y));

    /// <summary>
    /// Computes the absolute value of each component.
    /// </summary>
    public static float3 Abs(float3 x) => new(Abs(x.X), Abs(x.Y), Abs(x.Z));

    /// <summary>
    /// Computes the absolute value of each component.
    /// </summary>
    public static float4 Abs(float4 x) => new(Abs(x.X), Abs(x.Y), Abs(x.Z), Abs(x.W));

    /// <summary>
    /// Returns the largest integer less than or equal to the scalar.
    /// </summary>
    public static float Floor(float x) => MathF.Floor(x);

    /// <summary>
    /// Returns the largest integer less than or equal to each component.
    /// </summary>
    public static float2 Floor(float2 x) => new(Floor(x.X), Floor(x.Y));

    /// <summary>
    /// Returns the largest integer less than or equal to each component.
    /// </summary>
    public static float3 Floor(float3 x) => new(Floor(x.X), Floor(x.Y), Floor(x.Z));

    /// <summary>
    /// Returns the largest integer less than or equal to each component.
    /// </summary>
    public static float4 Floor(float4 x) => new(Floor(x.X), Floor(x.Y), Floor(x.Z), Floor(x.W));

    /// <summary>
    /// Returns the smallest integer greater than or equal to the scalar.
    /// </summary>
    public static float Ceil(float x) => MathF.Ceiling(x);

    /// <summary>
    /// Returns the smallest integer greater than or equal to each component.
    /// </summary>
    public static float2 Ceil(float2 x) => new(Ceil(x.X), Ceil(x.Y));

    /// <summary>
    /// Returns the smallest integer greater than or equal to each component.
    /// </summary>
    public static float3 Ceil(float3 x) => new(Ceil(x.X), Ceil(x.Y), Ceil(x.Z));

    /// <summary>
    /// Returns the smallest integer greater than or equal to each component.
    /// </summary>
    public static float4 Ceil(float4 x) => new(Ceil(x.X), Ceil(x.Y), Ceil(x.Z), Ceil(x.W));

    /// <summary>
    /// Rounds a scalar to the nearest integer.
    /// </summary>
    public static float Round(float x) => MathF.Round(x);

    /// <summary>
    /// Rounds each component to the nearest integer.
    /// </summary>
    public static float2 Round(float2 x) => new(Round(x.X), Round(x.Y));

    /// <summary>
    /// Rounds each component to the nearest integer.
    /// </summary>
    public static float3 Round(float3 x) => new(Round(x.X), Round(x.Y), Round(x.Z));

    /// <summary>
    /// Rounds each component to the nearest integer.
    /// </summary>
    public static float4 Round(float4 x) => new(Round(x.X), Round(x.Y), Round(x.Z), Round(x.W));

    /// <summary>
    /// Returns the fractional part of a scalar.
    /// </summary>
    public static float Fract(float x) => x - Floor(x);

    /// <summary>
    /// Returns the fractional part of each component.
    /// </summary>
    public static float2 Fract(float2 x) => x - Floor(x);

    /// <summary>
    /// Returns the fractional part of each component.
    /// </summary>
    public static float3 Fract(float3 x) => x - Floor(x);

    /// <summary>
    /// Returns the fractional part of each component.
    /// </summary>
    public static float4 Fract(float4 x) => x - Floor(x);

    /// <summary>
    /// Returns the fragment-stage partial derivative in screen-space X.
    /// </summary>
    public static float Ddx(float x) => ShaderRuntimeMarker<float>.Value;

    /// <summary>
    /// Returns the fragment-stage partial derivative in screen-space X.
    /// </summary>
    public static float2 Ddx(float2 x) => ShaderRuntimeMarker<float2>.Value;

    /// <summary>
    /// Returns the fragment-stage partial derivative in screen-space X.
    /// </summary>
    public static float3 Ddx(float3 x) => ShaderRuntimeMarker<float3>.Value;

    /// <summary>
    /// Returns the fragment-stage partial derivative in screen-space X.
    /// </summary>
    public static float4 Ddx(float4 x) => ShaderRuntimeMarker<float4>.Value;

    /// <summary>
    /// Returns the fragment-stage partial derivative in screen-space Y.
    /// </summary>
    public static float Ddy(float x) => ShaderRuntimeMarker<float>.Value;

    /// <summary>
    /// Returns the fragment-stage partial derivative in screen-space Y.
    /// </summary>
    public static float2 Ddy(float2 x) => ShaderRuntimeMarker<float2>.Value;

    /// <summary>
    /// Returns the fragment-stage partial derivative in screen-space Y.
    /// </summary>
    public static float3 Ddy(float3 x) => ShaderRuntimeMarker<float3>.Value;

    /// <summary>
    /// Returns the fragment-stage partial derivative in screen-space Y.
    /// </summary>
    public static float4 Ddy(float4 x) => ShaderRuntimeMarker<float4>.Value;

    /// <summary>
    /// Returns the lesser of two scalar values.
    /// </summary>
    public static float Min(float x, float y) => MathF.Min(x, y);

    /// <summary>Integer minimum, for clamping indices inside a kernel.</summary>
    public static int Min(int x, int y) => System.Math.Min(x, y);

    /// <summary>Returns the lesser integer for each component.</summary>
    public static int2 Min(int2 x, int2 y) => new(Min(x.X, y.X), Min(x.Y, y.Y));

    /// <summary>Returns the lesser integer for each component.</summary>
    public static int3 Min(int3 x, int3 y) => new(Min(x.X, y.X), Min(x.Y, y.Y), Min(x.Z, y.Z));

    /// <summary>Returns the lesser integer for each component.</summary>
    public static int4 Min(int4 x, int4 y) => new(Min(x.X, y.X), Min(x.Y, y.Y), Min(x.Z, y.Z), Min(x.W, y.W));

    /// <summary>Returns the lesser of each integer vector component and a scalar.</summary>
    public static int2 Min(int2 x, int y) => new(Min(x.X, y), Min(x.Y, y));

    /// <summary>Returns the lesser of each integer vector component and a scalar.</summary>
    public static int3 Min(int3 x, int y) => new(Min(x.X, y), Min(x.Y, y), Min(x.Z, y));

    /// <summary>Returns the lesser of each integer vector component and a scalar.</summary>
    public static int4 Min(int4 x, int y) => new(Min(x.X, y), Min(x.Y, y), Min(x.Z, y), Min(x.W, y));

    /// <summary>
    /// Returns the lesser value for each component.
    /// </summary>
    public static float2 Min(float2 x, float2 y) => new(Min(x.X, y.X), Min(x.Y, y.Y));

    /// <summary>
    /// Returns the lesser value for each component.
    /// </summary>
    public static float3 Min(float3 x, float3 y) => new(Min(x.X, y.X), Min(x.Y, y.Y), Min(x.Z, y.Z));

    /// <summary>
    /// Returns the lesser value for each component.
    /// </summary>
    public static float4 Min(float4 x, float4 y) => new(Min(x.X, y.X), Min(x.Y, y.Y), Min(x.Z, y.Z), Min(x.W, y.W));

    /// <summary>Returns the lesser of each vector component and a scalar.</summary>
    public static float2 Min(float2 x, float y) => new(Min(x.X, y), Min(x.Y, y));

    /// <summary>Returns the lesser of each vector component and a scalar.</summary>
    public static float3 Min(float3 x, float y) => new(Min(x.X, y), Min(x.Y, y), Min(x.Z, y));

    /// <summary>Returns the lesser of each vector component and a scalar.</summary>
    public static float4 Min(float4 x, float y) => new(Min(x.X, y), Min(x.Y, y), Min(x.Z, y), Min(x.W, y));

    /// <summary>
    /// Returns the greater of two scalar values.
    /// </summary>
    public static float Max(float x, float y) => MathF.Max(x, y);

    /// <summary>Integer maximum, for clamping indices inside a kernel.</summary>
    public static int Max(int x, int y) => System.Math.Max(x, y);

    /// <summary>Returns the greater integer for each component.</summary>
    public static int2 Max(int2 x, int2 y) => new(Max(x.X, y.X), Max(x.Y, y.Y));

    /// <summary>Returns the greater integer for each component.</summary>
    public static int3 Max(int3 x, int3 y) => new(Max(x.X, y.X), Max(x.Y, y.Y), Max(x.Z, y.Z));

    /// <summary>Returns the greater integer for each component.</summary>
    public static int4 Max(int4 x, int4 y) => new(Max(x.X, y.X), Max(x.Y, y.Y), Max(x.Z, y.Z), Max(x.W, y.W));

    /// <summary>Returns the greater of each integer vector component and a scalar.</summary>
    public static int2 Max(int2 x, int y) => new(Max(x.X, y), Max(x.Y, y));

    /// <summary>Returns the greater of each integer vector component and a scalar.</summary>
    public static int3 Max(int3 x, int y) => new(Max(x.X, y), Max(x.Y, y), Max(x.Z, y));

    /// <summary>Returns the greater of each integer vector component and a scalar.</summary>
    public static int4 Max(int4 x, int y) => new(Max(x.X, y), Max(x.Y, y), Max(x.Z, y), Max(x.W, y));

    /// <summary>
    /// Returns the greater value for each component.
    /// </summary>
    public static float2 Max(float2 x, float2 y) => new(Max(x.X, y.X), Max(x.Y, y.Y));

    /// <summary>
    /// Returns the greater value for each component.
    /// </summary>
    public static float3 Max(float3 x, float3 y) => new(Max(x.X, y.X), Max(x.Y, y.Y), Max(x.Z, y.Z));

    /// <summary>
    /// Returns the greater value for each component.
    /// </summary>
    public static float4 Max(float4 x, float4 y) => new(Max(x.X, y.X), Max(x.Y, y.Y), Max(x.Z, y.Z), Max(x.W, y.W));

    /// <summary>Returns the greater of each vector component and a scalar.</summary>
    public static float2 Max(float2 x, float y) => new(Max(x.X, y), Max(x.Y, y));

    /// <summary>Returns the greater of each vector component and a scalar.</summary>
    public static float3 Max(float3 x, float y) => new(Max(x.X, y), Max(x.Y, y), Max(x.Z, y));

    /// <summary>Returns the greater of each vector component and a scalar.</summary>
    public static float4 Max(float4 x, float y) => new(Max(x.X, y), Max(x.Y, y), Max(x.Z, y), Max(x.W, y));

    /// <summary>
    /// Clamps a scalar to a closed interval.
    /// </summary>
    public static float Clamp(float x, float min, float max) => System.Math.Clamp(x, min, max);

    /// <summary>Integer clamp, for keeping a texture index inside bounds.</summary>
    public static int Clamp(int x, int min, int max) => System.Math.Clamp(x, min, max);

    /// <summary>Clamps each integer vector component to a closed scalar interval.</summary>
    public static int2 Clamp(int2 x, int min, int max) => new(Clamp(x.X, min, max), Clamp(x.Y, min, max));

    /// <summary>Clamps each integer vector component to a closed scalar interval.</summary>
    public static int3 Clamp(int3 x, int min, int max) => new(Clamp(x.X, min, max), Clamp(x.Y, min, max), Clamp(x.Z, min, max));

    /// <summary>Clamps each integer vector component to a closed scalar interval.</summary>
    public static int4 Clamp(int4 x, int min, int max)
        => new(Clamp(x.X, min, max), Clamp(x.Y, min, max), Clamp(x.Z, min, max), Clamp(x.W, min, max));

    /// <summary>Clamps each integer vector component to the corresponding component interval.</summary>
    public static int2 Clamp(int2 x, int2 min, int2 max) => Max(min, Min(max, x));

    /// <summary>Clamps each integer vector component to the corresponding component interval.</summary>
    public static int3 Clamp(int3 x, int3 min, int3 max) => Max(min, Min(max, x));

    /// <summary>Clamps each integer vector component to the corresponding component interval.</summary>
    public static int4 Clamp(int4 x, int4 min, int4 max) => Max(min, Min(max, x));

    /// <summary>
    /// Clamps each vector component to a closed scalar interval.
    /// </summary>
    public static float2 Clamp(float2 x, float min, float max) => new(Clamp(x.X, min, max), Clamp(x.Y, min, max));

    /// <summary>
    /// Clamps each vector component to a closed scalar interval.
    /// </summary>
    public static float3 Clamp(float3 x, float min, float max) => new(Clamp(x.X, min, max), Clamp(x.Y, min, max), Clamp(x.Z, min, max));

    /// <summary>
    /// Clamps each vector component to a closed scalar interval.
    /// </summary>
    public static float4 Clamp(float4 x, float min, float max) => new(Clamp(x.X, min, max), Clamp(x.Y, min, max), Clamp(x.Z, min, max), Clamp(x.W, min, max));

    /// <summary>
    /// Clamps each vector component to the corresponding component interval.
    /// </summary>
    public static float2 Clamp(float2 x, float2 min, float2 max) => Max(min, Min(max, x));

    /// <summary>
    /// Clamps each vector component to the corresponding component interval.
    /// </summary>
    public static float3 Clamp(float3 x, float3 min, float3 max) => Max(min, Min(max, x));

    /// <summary>
    /// Clamps each vector component to the corresponding component interval.
    /// </summary>
    public static float4 Clamp(float4 x, float4 min, float4 max) => Max(min, Min(max, x));

    /// <summary>
    /// Clamps a scalar to the [0, 1] interval.
    /// </summary>
    public static float Saturate(float x) => Clamp(x, 0, 1);

    /// <summary>
    /// Clamps each component to the [0, 1] interval.
    /// </summary>
    public static float2 Saturate(float2 x) => Clamp(x, 0, 1);

    /// <summary>
    /// Clamps each component to the [0, 1] interval.
    /// </summary>
    public static float3 Saturate(float3 x) => Clamp(x, 0, 1);

    /// <summary>
    /// Clamps each component to the [0, 1] interval.
    /// </summary>
    public static float4 Saturate(float4 x) => Clamp(x, 0, 1);

    /// <summary>
    /// Linearly interpolates between two scalars.
    /// </summary>
    public static float Lerp(float a, float b, float t) => a + ((b - a) * t);

    /// <summary>
    /// Linearly interpolates between two vectors.
    /// </summary>
    public static float2 Lerp(float2 a, float2 b, float t) => a + ((b - a) * t);

    /// <summary>
    /// Linearly interpolates between two vectors.
    /// </summary>
    public static float3 Lerp(float3 a, float3 b, float t) => a + ((b - a) * t);

    /// <summary>
    /// Linearly interpolates between two vectors.
    /// </summary>
    public static float4 Lerp(float4 a, float4 b, float t) => a + ((b - a) * t);

    /// <summary>Linearly interpolates each component with its corresponding factor.</summary>
    public static float2 Lerp(float2 a, float2 b, float2 t) => a + ((b - a) * t);

    /// <summary>Linearly interpolates each component with its corresponding factor.</summary>
    public static float3 Lerp(float3 a, float3 b, float3 t) => a + ((b - a) * t);

    /// <summary>Linearly interpolates each component with its corresponding factor.</summary>
    public static float4 Lerp(float4 a, float4 b, float4 t) => a + ((b - a) * t);

    /// <summary>
    /// GLSL-style alias for <see cref="Lerp(float, float, float)"/>.
    /// </summary>
    public static float Mix(float a, float b, float t) => Lerp(a, b, t);

    /// <summary>
    /// GLSL-style alias for vector linear interpolation.
    /// </summary>
    public static float2 Mix(float2 a, float2 b, float t) => Lerp(a, b, t);

    /// <summary>
    /// GLSL-style alias for vector linear interpolation.
    /// </summary>
    public static float3 Mix(float3 a, float3 b, float t) => Lerp(a, b, t);

    /// <summary>
    /// GLSL-style alias for vector linear interpolation.
    /// </summary>
    public static float4 Mix(float4 a, float4 b, float t) => Lerp(a, b, t);

    /// <summary>GLSL-style alias for component-wise vector linear interpolation.</summary>
    public static float2 Mix(float2 a, float2 b, float2 t) => Lerp(a, b, t);

    /// <summary>GLSL-style alias for component-wise vector linear interpolation.</summary>
    public static float3 Mix(float3 a, float3 b, float3 t) => Lerp(a, b, t);

    /// <summary>GLSL-style alias for component-wise vector linear interpolation.</summary>
    public static float4 Mix(float4 a, float4 b, float4 t) => Lerp(a, b, t);

    /// <summary>
    /// Performs smooth Hermite interpolation.
    /// </summary>
    public static float Smoothstep(float edge0, float edge1, float x)
    {
        var t = Saturate((x - edge0) / (edge1 - edge0));
        return t * t * (3 - (2 * t));
    }

    /// <summary>Performs smooth Hermite interpolation component-wise.</summary>
    public static float2 Smoothstep(float2 edge0, float2 edge1, float2 x)
        => new(
            Smoothstep(edge0.X, edge1.X, x.X),
            Smoothstep(edge0.Y, edge1.Y, x.Y));

    /// <summary>Performs smooth Hermite interpolation component-wise.</summary>
    public static float3 Smoothstep(float3 edge0, float3 edge1, float3 x)
        => new(
            Smoothstep(edge0.X, edge1.X, x.X),
            Smoothstep(edge0.Y, edge1.Y, x.Y),
            Smoothstep(edge0.Z, edge1.Z, x.Z));

    /// <summary>Performs smooth Hermite interpolation component-wise.</summary>
    public static float4 Smoothstep(float4 edge0, float4 edge1, float4 x)
        => new(
            Smoothstep(edge0.X, edge1.X, x.X),
            Smoothstep(edge0.Y, edge1.Y, x.Y),
            Smoothstep(edge0.Z, edge1.Z, x.Z),
            Smoothstep(edge0.W, edge1.W, x.W));

    /// <summary>Performs smooth Hermite interpolation on each component using scalar edges.</summary>
    public static float2 Smoothstep(float edge0, float edge1, float2 x)
        => new(Smoothstep(edge0, edge1, x.X), Smoothstep(edge0, edge1, x.Y));

    /// <summary>Performs smooth Hermite interpolation on each component using scalar edges.</summary>
    public static float3 Smoothstep(float edge0, float edge1, float3 x)
        => new(Smoothstep(edge0, edge1, x.X), Smoothstep(edge0, edge1, x.Y), Smoothstep(edge0, edge1, x.Z));

    /// <summary>Performs smooth Hermite interpolation on each component using scalar edges.</summary>
    public static float4 Smoothstep(float edge0, float edge1, float4 x)
        => new(
            Smoothstep(edge0, edge1, x.X),
            Smoothstep(edge0, edge1, x.Y),
            Smoothstep(edge0, edge1, x.Z),
            Smoothstep(edge0, edge1, x.W));

    /// <summary>
    /// Multiplies a matrix by a vector.
    /// </summary>
    public static float2 Mul(float2x2 m, float2 v) => m * v;

    /// <summary>
    /// Multiplies a matrix by a vector.
    /// </summary>
    public static float3 Mul(float3x3 m, float3 v) => m * v;

    /// <summary>
    /// Multiplies a matrix by a vector.
    /// </summary>
    public static float4 Mul(float4x4 m, float4 v) => m * v;

    /// <summary>
    /// Multiplies two matrices.
    /// </summary>
    public static float2x2 Mul(float2x2 a, float2x2 b) => a * b;

    /// <summary>
    /// Multiplies two matrices.
    /// </summary>
    public static float3x3 Mul(float3x3 a, float3x3 b) => a * b;

    /// <summary>
    /// Multiplies two matrices.
    /// </summary>
    public static float4x4 Mul(float4x4 a, float4x4 b) => a * b;

    /// <summary>
    /// Returns the transposed matrix.
    /// </summary>
    public static float2x2 Transpose(float2x2 value) => value.Transposed();

    /// <summary>
    /// Returns the transposed matrix.
    /// </summary>
    public static float3x3 Transpose(float3x3 value) => value.Transposed();

    /// <summary>
    /// Returns the transposed matrix.
    /// </summary>
    public static float4x4 Transpose(float4x4 value) => value.Transposed();

    /// <summary>
    /// Returns the determinant of a matrix.
    /// </summary>
    public static float Determinant(float2x2 value) => value.Determinant();

    /// <summary>
    /// Returns the determinant of a matrix.
    /// </summary>
    public static float Determinant(float3x3 value) => value.Determinant();

    /// <summary>
    /// Returns the determinant of a matrix.
    /// </summary>
    public static float Determinant(float4x4 value) => value.Determinant();

    /// <summary>
    /// Returns the inverse of a matrix.
    /// </summary>
    public static float2x2 Inverse(float2x2 value) => value.Inverse();

    /// <summary>
    /// Returns the inverse of a matrix.
    /// </summary>
    public static float3x3 Inverse(float3x3 value) => value.Inverse();

    /// <summary>
    /// Returns the inverse of a matrix.
    /// </summary>
    public static float4x4 Inverse(float4x4 value) => value.Inverse();

    /// <summary>
    /// Multiplies two matrices component-wise.
    /// </summary>
    public static float2x2 Hadamard(float2x2 a, float2x2 b) => float2x2.Hadamard(a, b);

    /// <summary>
    /// Multiplies two matrices component-wise.
    /// </summary>
    public static float3x3 Hadamard(float3x3 a, float3x3 b) => float3x3.Hadamard(a, b);

    /// <summary>
    /// Multiplies two matrices component-wise.
    /// </summary>
    public static float4x4 Hadamard(float4x4 a, float4x4 b) => float4x4.Hadamard(a, b);
}

/// <summary>
/// Provides HLSL-style aliases for <see cref="ShaderMath"/>.
/// </summary>
public static class Hlsl
{
    /// <inheritdoc cref="ShaderMath.Dot(float2, float2)"/>
    public static float Dot(float2 a, float2 b) => ShaderMath.Dot(a, b);

    /// <inheritdoc cref="ShaderMath.Dot(float3, float3)"/>
    public static float Dot(float3 a, float3 b) => ShaderMath.Dot(a, b);

    /// <inheritdoc cref="ShaderMath.Dot(float4, float4)"/>
    public static float Dot(float4 a, float4 b) => ShaderMath.Dot(a, b);

    /// <inheritdoc cref="ShaderMath.Cross(float3, float3)"/>
    public static float3 Cross(float3 a, float3 b) => ShaderMath.Cross(a, b);

    /// <inheritdoc cref="ShaderMath.Length(float2)"/>
    public static float Length(float2 v) => ShaderMath.Length(v);

    /// <inheritdoc cref="ShaderMath.Length(float3)"/>
    public static float Length(float3 v) => ShaderMath.Length(v);

    /// <inheritdoc cref="ShaderMath.Length(float4)"/>
    public static float Length(float4 v) => ShaderMath.Length(v);

    /// <inheritdoc cref="ShaderMath.Normalize(float2)"/>
    public static float2 Normalize(float2 v) => ShaderMath.Normalize(v);

    /// <inheritdoc cref="ShaderMath.Normalize(float3)"/>
    public static float3 Normalize(float3 v) => ShaderMath.Normalize(v);

    /// <inheritdoc cref="ShaderMath.Normalize(float4)"/>
    public static float4 Normalize(float4 v) => ShaderMath.Normalize(v);

    /// <inheritdoc cref="ShaderMath.Sin(float)"/>
    public static float Sin(float x) => ShaderMath.Sin(x);

    /// <inheritdoc cref="ShaderMath.Sin(float2)"/>
    public static float2 Sin(float2 x) => ShaderMath.Sin(x);

    /// <inheritdoc cref="ShaderMath.Sin(float3)"/>
    public static float3 Sin(float3 x) => ShaderMath.Sin(x);

    /// <inheritdoc cref="ShaderMath.Sin(float4)"/>
    public static float4 Sin(float4 x) => ShaderMath.Sin(x);

    /// <inheritdoc cref="ShaderMath.Cos(float)"/>
    public static float Cos(float x) => ShaderMath.Cos(x);

    /// <inheritdoc cref="ShaderMath.Cos(float2)"/>
    public static float2 Cos(float2 x) => ShaderMath.Cos(x);

    /// <inheritdoc cref="ShaderMath.Cos(float3)"/>
    public static float3 Cos(float3 x) => ShaderMath.Cos(x);

    /// <inheritdoc cref="ShaderMath.Cos(float4)"/>
    public static float4 Cos(float4 x) => ShaderMath.Cos(x);

    /// <inheritdoc cref="ShaderMath.Tan(float)"/>
    public static float Tan(float x) => ShaderMath.Tan(x);

    /// <inheritdoc cref="ShaderMath.Tan(float2)"/>
    public static float2 Tan(float2 x) => ShaderMath.Tan(x);

    /// <inheritdoc cref="ShaderMath.Tan(float3)"/>
    public static float3 Tan(float3 x) => ShaderMath.Tan(x);

    /// <inheritdoc cref="ShaderMath.Tan(float4)"/>
    public static float4 Tan(float4 x) => ShaderMath.Tan(x);

    /// <inheritdoc cref="ShaderMath.Exp(float)"/>
    public static float Exp(float x) => ShaderMath.Exp(x);

    /// <inheritdoc cref="ShaderMath.Exp(float2)"/>
    public static float2 Exp(float2 x) => ShaderMath.Exp(x);

    /// <inheritdoc cref="ShaderMath.Exp(float3)"/>
    public static float3 Exp(float3 x) => ShaderMath.Exp(x);

    /// <inheritdoc cref="ShaderMath.Exp(float4)"/>
    public static float4 Exp(float4 x) => ShaderMath.Exp(x);

    /// <inheritdoc cref="ShaderMath.Log(float)"/>
    public static float Log(float x) => ShaderMath.Log(x);

    /// <inheritdoc cref="ShaderMath.Log(float2)"/>
    public static float2 Log(float2 x) => ShaderMath.Log(x);

    /// <inheritdoc cref="ShaderMath.Log(float3)"/>
    public static float3 Log(float3 x) => ShaderMath.Log(x);

    /// <inheritdoc cref="ShaderMath.Log(float4)"/>
    public static float4 Log(float4 x) => ShaderMath.Log(x);

    /// <inheritdoc cref="ShaderMath.Pow(float, float)"/>
    public static float Pow(float x, float y) => ShaderMath.Pow(x, y);

    /// <inheritdoc cref="ShaderMath.Pow(float2, float2)"/>
    public static float2 Pow(float2 x, float2 y) => ShaderMath.Pow(x, y);

    /// <inheritdoc cref="ShaderMath.Pow(float3, float3)"/>
    public static float3 Pow(float3 x, float3 y) => ShaderMath.Pow(x, y);

    /// <inheritdoc cref="ShaderMath.Pow(float4, float4)"/>
    public static float4 Pow(float4 x, float4 y) => ShaderMath.Pow(x, y);

    /// <inheritdoc cref="ShaderMath.Sqrt(float)"/>
    public static float Sqrt(float x) => ShaderMath.Sqrt(x);

    /// <inheritdoc cref="ShaderMath.Sqrt(float2)"/>
    public static float2 Sqrt(float2 x) => ShaderMath.Sqrt(x);

    /// <inheritdoc cref="ShaderMath.Sqrt(float3)"/>
    public static float3 Sqrt(float3 x) => ShaderMath.Sqrt(x);

    /// <inheritdoc cref="ShaderMath.Sqrt(float4)"/>
    public static float4 Sqrt(float4 x) => ShaderMath.Sqrt(x);

    /// <inheritdoc cref="ShaderMath.Abs(float)"/>
    public static float Abs(float x) => ShaderMath.Abs(x);

    /// <inheritdoc cref="ShaderMath.Abs(float2)"/>
    public static float2 Abs(float2 x) => ShaderMath.Abs(x);

    /// <inheritdoc cref="ShaderMath.Abs(float3)"/>
    public static float3 Abs(float3 x) => ShaderMath.Abs(x);

    /// <inheritdoc cref="ShaderMath.Abs(float4)"/>
    public static float4 Abs(float4 x) => ShaderMath.Abs(x);

    /// <inheritdoc cref="ShaderMath.Floor(float)"/>
    public static float Floor(float x) => ShaderMath.Floor(x);

    /// <inheritdoc cref="ShaderMath.Floor(float2)"/>
    public static float2 Floor(float2 x) => ShaderMath.Floor(x);

    /// <inheritdoc cref="ShaderMath.Floor(float3)"/>
    public static float3 Floor(float3 x) => ShaderMath.Floor(x);

    /// <inheritdoc cref="ShaderMath.Floor(float4)"/>
    public static float4 Floor(float4 x) => ShaderMath.Floor(x);

    /// <inheritdoc cref="ShaderMath.Ceil(float)"/>
    public static float Ceil(float x) => ShaderMath.Ceil(x);

    /// <inheritdoc cref="ShaderMath.Ceil(float2)"/>
    public static float2 Ceil(float2 x) => ShaderMath.Ceil(x);

    /// <inheritdoc cref="ShaderMath.Ceil(float3)"/>
    public static float3 Ceil(float3 x) => ShaderMath.Ceil(x);

    /// <inheritdoc cref="ShaderMath.Ceil(float4)"/>
    public static float4 Ceil(float4 x) => ShaderMath.Ceil(x);

    /// <inheritdoc cref="ShaderMath.Fract(float)"/>
    public static float Fract(float x) => ShaderMath.Fract(x);

    /// <inheritdoc cref="ShaderMath.Fract(float2)"/>
    public static float2 Fract(float2 x) => ShaderMath.Fract(x);

    /// <inheritdoc cref="ShaderMath.Fract(float3)"/>
    public static float3 Fract(float3 x) => ShaderMath.Fract(x);

    /// <inheritdoc cref="ShaderMath.Fract(float4)"/>
    public static float4 Fract(float4 x) => ShaderMath.Fract(x);

    /// <inheritdoc cref="ShaderMath.Clamp(float, float, float)"/>
    public static float Clamp(float x, float min, float max) => ShaderMath.Clamp(x, min, max);

    /// <inheritdoc cref="ShaderMath.Clamp(float2, float, float)"/>
    public static float2 Clamp(float2 x, float min, float max) => ShaderMath.Clamp(x, min, max);

    /// <inheritdoc cref="ShaderMath.Clamp(float3, float, float)"/>
    public static float3 Clamp(float3 x, float min, float max) => ShaderMath.Clamp(x, min, max);

    /// <inheritdoc cref="ShaderMath.Clamp(float4, float, float)"/>
    public static float4 Clamp(float4 x, float min, float max) => ShaderMath.Clamp(x, min, max);

    /// <inheritdoc cref="ShaderMath.Clamp(float2, float2, float2)"/>
    public static float2 Clamp(float2 x, float2 min, float2 max) => ShaderMath.Clamp(x, min, max);

    /// <inheritdoc cref="ShaderMath.Clamp(float3, float3, float3)"/>
    public static float3 Clamp(float3 x, float3 min, float3 max) => ShaderMath.Clamp(x, min, max);

    /// <inheritdoc cref="ShaderMath.Clamp(float4, float4, float4)"/>
    public static float4 Clamp(float4 x, float4 min, float4 max) => ShaderMath.Clamp(x, min, max);

    /// <inheritdoc cref="ShaderMath.Lerp(float, float, float)"/>
    public static float Lerp(float a, float b, float t) => ShaderMath.Lerp(a, b, t);

    /// <inheritdoc cref="ShaderMath.Lerp(float2, float2, float)"/>
    public static float2 Lerp(float2 a, float2 b, float t) => ShaderMath.Lerp(a, b, t);

    /// <inheritdoc cref="ShaderMath.Lerp(float3, float3, float)"/>
    public static float3 Lerp(float3 a, float3 b, float t) => ShaderMath.Lerp(a, b, t);

    /// <inheritdoc cref="ShaderMath.Lerp(float4, float4, float)"/>
    public static float4 Lerp(float4 a, float4 b, float t) => ShaderMath.Lerp(a, b, t);

    /// <inheritdoc cref="ShaderMath.Lerp(float2, float2, float2)"/>
    public static float2 Lerp(float2 a, float2 b, float2 t) => ShaderMath.Lerp(a, b, t);

    /// <inheritdoc cref="ShaderMath.Lerp(float3, float3, float3)"/>
    public static float3 Lerp(float3 a, float3 b, float3 t) => ShaderMath.Lerp(a, b, t);

    /// <inheritdoc cref="ShaderMath.Lerp(float4, float4, float4)"/>
    public static float4 Lerp(float4 a, float4 b, float4 t) => ShaderMath.Lerp(a, b, t);

    /// <inheritdoc cref="ShaderMath.Mix(float, float, float)"/>
    public static float Mix(float a, float b, float t) => ShaderMath.Mix(a, b, t);

    /// <inheritdoc cref="ShaderMath.Mix(float2, float2, float)"/>
    public static float2 Mix(float2 a, float2 b, float t) => ShaderMath.Mix(a, b, t);

    /// <inheritdoc cref="ShaderMath.Mix(float3, float3, float)"/>
    public static float3 Mix(float3 a, float3 b, float t) => ShaderMath.Mix(a, b, t);

    /// <inheritdoc cref="ShaderMath.Mix(float4, float4, float)"/>
    public static float4 Mix(float4 a, float4 b, float t) => ShaderMath.Mix(a, b, t);

    /// <inheritdoc cref="ShaderMath.Mix(float2, float2, float2)"/>
    public static float2 Mix(float2 a, float2 b, float2 t) => ShaderMath.Mix(a, b, t);

    /// <inheritdoc cref="ShaderMath.Mix(float3, float3, float3)"/>
    public static float3 Mix(float3 a, float3 b, float3 t) => ShaderMath.Mix(a, b, t);

    /// <inheritdoc cref="ShaderMath.Mix(float4, float4, float4)"/>
    public static float4 Mix(float4 a, float4 b, float4 t) => ShaderMath.Mix(a, b, t);

    /// <inheritdoc cref="ShaderMath.Mul(float2x2, float2)"/>
    public static float2 Mul(float2x2 m, float2 v) => ShaderMath.Mul(m, v);

    /// <inheritdoc cref="ShaderMath.Mul(float3x3, float3)"/>
    public static float3 Mul(float3x3 m, float3 v) => ShaderMath.Mul(m, v);

    /// <inheritdoc cref="ShaderMath.Mul(float4x4, float4)"/>
    public static float4 Mul(float4x4 m, float4 v) => ShaderMath.Mul(m, v);

    /// <inheritdoc cref="ShaderMath.Mul(float2x2, float2x2)"/>
    public static float2x2 Mul(float2x2 a, float2x2 b) => ShaderMath.Mul(a, b);

    /// <inheritdoc cref="ShaderMath.Mul(float3x3, float3x3)"/>
    public static float3x3 Mul(float3x3 a, float3x3 b) => ShaderMath.Mul(a, b);

    /// <inheritdoc cref="ShaderMath.Mul(float4x4, float4x4)"/>
    public static float4x4 Mul(float4x4 a, float4x4 b) => ShaderMath.Mul(a, b);

    /// <inheritdoc cref="ShaderMath.Transpose(float4x4)"/>
    public static float4x4 Transpose(float4x4 value) => ShaderMath.Transpose(value);

    /// <inheritdoc cref="ShaderMath.Inverse(float4x4)"/>
    public static float4x4 Inverse(float4x4 value) => ShaderMath.Inverse(value);
}
