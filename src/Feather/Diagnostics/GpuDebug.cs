using Feather.Math;

namespace Feather;

/// <summary>
/// Shader-visible, typed diagnostic markers. Ordinary kernels treat <see cref="Print(float)"/>
/// as an identity function and <c>Assert</c> as the supplied condition. An explicitly armed
/// profile capture substitutes a private instrumented kernel that emits bounded raw records.
/// </summary>
public static class GpuDebug
{
    public static bool Print(bool value) => value;
    public static bool2 Print(bool2 value) => value;
    public static bool3 Print(bool3 value) => value;
    public static bool4 Print(bool4 value) => value;
    public static int Print(int value) => value;
    public static int2 Print(int2 value) => value;
    public static int3 Print(int3 value) => value;
    public static int4 Print(int4 value) => value;
    public static uint Print(uint value) => value;
    public static float Print(float value) => value;
    public static float2 Print(float2 value) => value;
    public static float3 Print(float3 value) => value;
    public static float4 Print(float4 value) => value;

    public static bool Assert(bool condition) => condition;
    public static bool Assert(bool condition, bool payload) => condition;
    public static bool Assert(bool condition, bool2 payload) => condition;
    public static bool Assert(bool condition, bool3 payload) => condition;
    public static bool Assert(bool condition, bool4 payload) => condition;
    public static bool Assert(bool condition, int payload) => condition;
    public static bool Assert(bool condition, int2 payload) => condition;
    public static bool Assert(bool condition, int3 payload) => condition;
    public static bool Assert(bool condition, int4 payload) => condition;
    public static bool Assert(bool condition, uint payload) => condition;
    public static bool Assert(bool condition, float payload) => condition;
    public static bool Assert(bool condition, float2 payload) => condition;
    public static bool Assert(bool condition, float3 payload) => condition;
    public static bool Assert(bool condition, float4 payload) => condition;
}
