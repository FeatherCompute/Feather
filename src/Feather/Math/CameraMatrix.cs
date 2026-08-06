namespace Feather.Math;

/// <summary>
/// Builds view and projection matrices in the clip-space convention Feather renders with.
/// </summary>
/// <remarks>
/// <para>
/// A pass that renders the scene from somewhere other than the camera it was handed -- a shadow map is
/// the usual reason -- has to build its own view-projection, and until this existed there was nowhere
/// to get one. Every caller wrote its own, which is how the sample renderer ended up with a private
/// <c>PerspectiveVk</c>, and a hand-rolled projection whose depth range or handedness is subtly wrong
/// produces an image that looks plausible and compares wrongly.
/// </para>
/// <para>
/// Everything here targets Feather's Vulkan-style viewport: clip Y points down, and clip depth
/// runs 0 at the near plane to 1 at the far one. That is deliberately not OpenGL's -1..1, and it is the
/// same space <c>RenderCamera.ViewProjection</c> arrives in, so a matrix from this class and one from
/// the host can be compared and mixed. The view half is right-handed and looks down -Z, which is the
/// convention Blender itself uses for a camera and for a lamp.
/// </para>
/// </remarks>
public static class CameraMatrix
{
    // Below this the requested up direction is treated as parallel to the view direction and replaced.
    // Comparing the cross product's length rather than a dot product keeps the test scale-free without
    // normalising either input first.
    private const float DegenerateUpEpsilon = 1e-4f;

    /// <summary>
    /// Builds a right-handed view matrix that places the eye at <paramref name="eye"/> looking toward
    /// <paramref name="target"/>.
    /// </summary>
    /// <remarks>
    /// When <paramref name="up"/> is parallel to the view direction the basis it would produce is
    /// degenerate, and a substitute axis is chosen instead of throwing. That case is the common one
    /// rather than a mistake: a ceiling lamp aims straight down, so a caller passing world up gets it
    /// every time, and there is no answer it wants other than "roll the frame however you like".
    /// </remarks>
    public static float4x4 LookAt(float3 eye, float3 target, float3 up)
        => LookTowards(eye, target - eye, up);

    /// <summary>
    /// Builds a right-handed view matrix from an eye and a direction rather than a target point.
    /// </summary>
    /// <remarks>
    /// The form a light wants. <c>SceneLight.Direction</c> is already the direction a lamp emits along,
    /// so a caller with a light in hand has a direction and no target, and manufacturing one by adding
    /// an arbitrary distance would only be thrown away here.
    /// </remarks>
    public static float4x4 LookTowards(float3 eye, float3 direction, float3 up)
    {
        var forward = ShaderMath.Normalize(direction);
        var right = ShaderMath.Cross(forward, up);
        if (ShaderMath.Length(right) < DegenerateUpEpsilon)
        {
            // Any axis not parallel to the view direction will do, and one of these two cannot be:
            // they are perpendicular to each other.
            var fallback = System.MathF.Abs(forward.Z) < 0.9f
                ? new float3(0.0f, 0.0f, 1.0f)
                : new float3(0.0f, 1.0f, 0.0f);
            right = ShaderMath.Cross(forward, fallback);
        }

        right = ShaderMath.Normalize(right);
        var trueUp = ShaderMath.Cross(right, forward);

        return new float4x4(
            new float4(right.X, trueUp.X, -forward.X, 0.0f),
            new float4(right.Y, trueUp.Y, -forward.Y, 0.0f),
            new float4(right.Z, trueUp.Z, -forward.Z, 0.0f),
            new float4(
                -ShaderMath.Dot(right, eye),
                -ShaderMath.Dot(trueUp, eye),
                ShaderMath.Dot(forward, eye),
                1.0f));
    }

    /// <summary>
    /// Builds a perspective projection from a vertical field of view in radians.
    /// </summary>
    /// <remarks>
    /// Radians rather than degrees because that is what the rest of Feather and Blender's own
    /// <c>angle_y</c> use, and a silent unit mismatch here renders a frame that is merely oddly framed
    /// rather than obviously broken.
    /// </remarks>
    public static float4x4 Perspective(float verticalFieldOfView, float aspect, float near, float far)
    {
        var focal = 1.0f / System.MathF.Tan(verticalFieldOfView * 0.5f);
        return new float4x4(
            new float4(focal / aspect, 0.0f, 0.0f, 0.0f),
            // Negated: clip Y points down in the Vulkan viewport, so a projection that did not flip
            // here would render the scene upside down.
            new float4(0.0f, -focal, 0.0f, 0.0f),
            new float4(0.0f, 0.0f, far / (near - far), -1.0f),
            new float4(0.0f, 0.0f, (near * far) / (near - far), 0.0f));
    }

    /// <summary>
    /// Builds an orthographic projection spanning <paramref name="width"/> by <paramref name="height"/>
    /// centred on the view axis.
    /// </summary>
    /// <remarks>
    /// What a directional light needs: a sun's rays are parallel, so its shadow map has no eye point to
    /// diverge from and a perspective projection would distort the depths it stores.
    /// </remarks>
    public static float4x4 Orthographic(float width, float height, float near, float far)
        => new(
            new float4(2.0f / width, 0.0f, 0.0f, 0.0f),
            new float4(0.0f, -2.0f / height, 0.0f, 0.0f),
            new float4(0.0f, 0.0f, 1.0f / (near - far), 0.0f),
            new float4(0.0f, 0.0f, near / (near - far), 1.0f));
}
