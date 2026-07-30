// Injected into consuming projects as SOURCE by FeatherCompute.targets.
//
// It cannot live in Feather.dll: the generator lowers [Callable] bodies from syntax, so a callable
// that only exists as metadata is rejected with FE0008 ("must be source-available"). Shipping the
// text is the only way a shared shader helper can cross an assembly boundary.

using Feather.Math;
using Feather.RenderGraph;

namespace Feather.Shaders;

/// <summary>
/// A camera reduced to what a per-pixel shader actually needs: the inverse of the view-projection
/// used to unproject pixels, and the world-space eye the rays start from.
/// </summary>
/// <remarks>
/// 80 bytes as a push constant (64 for the matrix, 16 for the padded eye), comfortably inside the
/// 128 bytes Vulkan guarantees.
/// </remarks>
[GpuStruct]
public partial struct GpuCamera
{
    /// <summary>Unprojects clip space back to world space, already in the shader's axis convention.</summary>
    public float4x4 InverseViewProjection;

    /// <summary>The eye, in the same convention as <see cref="InverseViewProjection"/>.</summary>
    public float3 WorldPosition;
}

/// <summary>
/// Builds <see cref="GpuCamera"/> from the host's <see cref="RenderCamera"/>. Runs on the CPU
/// inside a pass, so it is ordinary C# rather than shader code.
/// </summary>
public static class GpuCameraFactory
{
    /// <summary>
    /// Takes the camera as Blender hands it over, with Z as the up axis.
    /// </summary>
    /// <remarks>
    /// Shaders are conventionally written Y-up, and porting every shader to Z-up instead is the
    /// wrong trade: the swap is a property of the host, not of the effect. Folding it into the
    /// matrix here means the GPU side never sees a swizzle, and an author never has to know which
    /// application supplied the camera.
    /// </remarks>
    public static GpuCamera FromBlender(RenderCamera camera)
        => Create(camera, upAxisIsZ: true);

    /// <summary>
    /// Takes a camera whose world is already Y-up, applying no axis conversion.
    /// </summary>
    public static GpuCamera FromYUp(RenderCamera camera)
        => Create(camera, upAxisIsZ: false);

    private static GpuCamera Create(RenderCamera camera, bool upAxisIsZ)
    {
        var inverse = camera.InverseViewProjection;
        var eye = camera.WorldPosition;
        if (upAxisIsZ)
        {
            // Unproject into the host's world first, then swap into the shader's, so the conversion
            // rides along with every ray the matrix produces.
            inverse = SwapYZ * inverse;
            eye = new float3(eye.X, eye.Z, eye.Y);
        }

        return new GpuCamera
        {
            InverseViewProjection = inverse,
            WorldPosition = eye
        };
    }

    /// <summary>
    /// Exchanges the Y and Z axes, converting between a Z-up and a Y-up world.
    /// </summary>
    /// <remarks>
    /// Columns, because <see cref="float4x4"/> is column-major. The swap is its own inverse, so the
    /// same matrix converts in both directions.
    /// </remarks>
    private static float4x4 SwapYZ { get; } = new(
        new float4(1.0f, 0.0f, 0.0f, 0.0f),
        new float4(0.0f, 0.0f, 1.0f, 0.0f),
        new float4(0.0f, 1.0f, 0.0f, 0.0f),
        new float4(0.0f, 0.0f, 0.0f, 1.0f));
}

/// <summary>
/// Turns pixels into world-space rays. This is the whole point of the layer: an effect asks for a
/// ray and never touches normalised device coordinates, a W divide, or an axis convention.
/// </summary>
[ShaderLibrary]
public static class FeatherCamera
{
    /// <summary>
    /// Rebuilds the camera inside a kernel from the two values a kernel can actually receive.
    /// </summary>
    /// <remarks>
    /// A <see cref="GpuCamera"/> cannot cross the uniform boundary directly: EasyGPU rejects a
    /// user struct as a push constant even though the generator accepts one, so a kernel takes the
    /// matrix and the eye as separate uniforms and pairs them here. Everything past this call sees
    /// one camera rather than two loose uniforms, which is what keeps the helpers below shareable.
    /// </remarks>
    [Callable]
    public static GpuCamera FromUniforms(float4x4 inverseViewProjection, float3 worldPosition)
    {
        return new GpuCamera
        {
            InverseViewProjection = inverseViewProjection,
            WorldPosition = worldPosition
        };
    }

    /// <summary>
    /// The world-space direction through the centre of a pixel, normalised.
    /// </summary>
    /// <remarks>
    /// Frames are stored bottom-up, so a pixel's Y already climbs in the same direction as clip
    /// space and needs no flip. Getting that backwards renders the world upside down, which is why
    /// it lives here once rather than in every effect.
    /// </remarks>
    [Callable]
    public static float3 RayDirection(GpuCamera camera, float2 pixel, float2 size)
    {
        var ndc = new float2(
            (((pixel.X + 0.5f) / size.X) * 2.0f) - 1.0f,
            (((pixel.Y + 0.5f) / size.Y) * 2.0f) - 1.0f);

        var near = UnprojectDepth(camera, ndc, -1.0f);
        var far = UnprojectDepth(camera, ndc, 1.0f);
        return ShaderMath.Normalize(far - near);
    }

    /// <summary>The world-space point a ray starts from.</summary>
    [Callable]
    public static float3 RayOrigin(GpuCamera camera)
    {
        return camera.WorldPosition;
    }

    /// <summary>Unprojects a clip-space depth onto its world-space point.</summary>
    [Callable]
    public static float3 UnprojectDepth(GpuCamera camera, float2 ndc, float depth)
    {
        var clip = ShaderMath.Mul(camera.InverseViewProjection, new float4(ndc.X, ndc.Y, depth, 1.0f));
        var scale = 1.0f / clip.W;
        return new float3(clip.X * scale, clip.Y * scale, clip.Z * scale);
    }
}
