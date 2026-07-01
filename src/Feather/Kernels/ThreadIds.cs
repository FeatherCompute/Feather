using Feather.Math;

namespace Feather;

public static class ThreadIds
{
    public static int X => ShaderRuntimeMarker<int>.Value;
    public static int Y => ShaderRuntimeMarker<int>.Value;
    public static int Z => ShaderRuntimeMarker<int>.Value;
    public static int2 XY => new(X, Y);
    public static int3 XYZ => new(X, Y, Z);
}

public static class GroupIds
{
    public static int X => ShaderRuntimeMarker<int>.Value;
    public static int Y => ShaderRuntimeMarker<int>.Value;
    public static int Z => ShaderRuntimeMarker<int>.Value;
    public static int2 XY => new(X, Y);
    public static int3 XYZ => new(X, Y, Z);
}

public static class LocalIds
{
    public static int X => ShaderRuntimeMarker<int>.Value;
    public static int Y => ShaderRuntimeMarker<int>.Value;
    public static int Z => ShaderRuntimeMarker<int>.Value;
    public static int2 XY => new(X, Y);
    public static int3 XYZ => new(X, Y, Z);
}

public static class DispatchSize
{
    public static int X => ShaderRuntimeMarker<int>.Value;
    public static int Y => ShaderRuntimeMarker<int>.Value;
    public static int Z => ShaderRuntimeMarker<int>.Value;
    public static int2 XY => new(X, Y);
    public static int3 XYZ => new(X, Y, Z);
}

public static class GroupSize
{
    public static int X => ShaderRuntimeMarker<int>.Value;
    public static int Y => ShaderRuntimeMarker<int>.Value;
    public static int Z => ShaderRuntimeMarker<int>.Value;
    public static int2 XY => new(X, Y);
    public static int3 XYZ => new(X, Y, Z);
}

internal static class ShaderRuntimeMarker<T>
{
    public static T Value => throw new InvalidOperationException("This member is a shader marker and can only be used inside Feather-generated GPU code.");
    public static ref T RefValue => throw new InvalidOperationException("This member is a shader marker and can only be used inside Feather-generated GPU code.");
}
