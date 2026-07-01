using Feather.Math;

namespace Feather.Graphics;

/// <summary>
/// Provides shader-only vertex identifiers for generated graphics code.
/// </summary>
public static class VertexIds
{
    /// <summary>
    /// Gets the zero-based vertex index for the current vertex shader invocation.
    /// </summary>
    public static int Index => ShaderRuntimeMarker<int>.Value;

    /// <summary>
    /// Gets the zero-based instance index for the current vertex shader invocation.
    /// </summary>
    public static int Instance => ShaderRuntimeMarker<int>.Value;
}

/// <summary>
/// Provides shader-only fragment identifiers for generated graphics code.
/// </summary>
public static class FragmentIds
{
    /// <summary>
    /// Gets the fragment coordinate for the current fragment shader invocation.
    /// </summary>
    public static float4 Coord => ShaderRuntimeMarker<float4>.Value;
}
