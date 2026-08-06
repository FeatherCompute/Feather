using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.RenderGraph;
using Feather.Resources;
using Feather.Shaders;

namespace Feather.Integration.Tests;

/// <summary>
/// Guards the camera helper layer that effects build rays with.
/// </summary>
/// <remarks>
/// The layer rests on three things the generator has to do together: accept a user
/// <c>[GpuStruct]</c> as a push constant, lower <c>[Callable]</c> helpers imported from a
/// <c>[ShaderLibrary]</c>, and follow those helpers when they call each other. Any one of them
/// failing turns into a shader that will not build, so they are asserted on the GLSL rather than
/// left to a rendered image to reveal.
/// </remarks>
public class ShaderCameraTests
{
    [Fact]
    public void BlenderConversionMovesTheUpAxisOntoY()
    {
        // A camera 10 units up in Blender's world, where up is Z.
        var camera = new RenderCamera(
            float4x4.Identity,
            float4x4.Identity,
            new float3(1.0f, 2.0f, 10.0f));

        var converted = GpuCameraFactory.FromBlender(camera);

        // The height must land on Y, because that is where a Y-up shader looks for it.
        Assert.Equal(1.0f, converted.WorldPosition.X, 5);
        Assert.Equal(10.0f, converted.WorldPosition.Y, 5);
        Assert.Equal(2.0f, converted.WorldPosition.Z, 5);
    }

    [Fact]
    public void YUpConversionLeavesTheCameraAlone()
    {
        var camera = new RenderCamera(
            float4x4.Identity,
            float4x4.Identity,
            new float3(1.0f, 2.0f, 10.0f));

        var converted = GpuCameraFactory.FromYUp(camera);

        Assert.Equal(1.0f, converted.WorldPosition.X, 5);
        Assert.Equal(2.0f, converted.WorldPosition.Y, 5);
        Assert.Equal(10.0f, converted.WorldPosition.Z, 5);
    }

    [Fact]
    public void BlenderConversionSwapsTheMatrixRowsThatUnprojectRays()
    {
        // Identity in, so the conversion is the only thing the result can show.
        var camera = new RenderCamera(
            float4x4.Identity,
            float4x4.Identity,
            new float3(0.0f, 0.0f, 0.0f));

        var converted = GpuCameraFactory.FromBlender(camera);

        // A Blender-world point straight up (+Z) must come out as shader-world up (+Y).
        var up = ShaderMath.Mul(converted.InverseViewProjection, new float4(0.0f, 0.0f, 1.0f, 1.0f));
        Assert.Equal(0.0f, up.X, 5);
        Assert.Equal(1.0f, up.Y, 5);
        Assert.Equal(0.0f, up.Z, 5);
    }

    /// <summary>
    /// A camera at the origin looking down -Z with a 90 degree vertical field of view, as the inverse
    /// a kernel actually receives. An identity matrix cannot stand in here: unprojecting through it
    /// leaves the near and far points differing only in Z, so every ray comes out as (0, 0, +-1) and
    /// an orientation assertion would read zero regardless of which way the frame is wound.
    /// </summary>
    private static float4x4 SquarePerspectiveInverse()
    {
        const float Near = 0.1f;
        const float Far = 100.0f;
        var depthScale = (Far + Near) / (Near - Far);
        var depthOffset = (2.0f * Far * Near) / (Near - Far);

        // Column-major, so each group of four arguments is one column.
        var projection = new float4x4(
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, depthScale, -1.0f,
            0.0f, 0.0f, depthOffset, 0.0f);
        return projection.Inverse();
    }

    [Fact]
    public void TheTopRowOfAFrameLooksUpward()
    {
        // A kernel's row zero is the top of the frame, and the frame header agrees (FrameFileWriter
        // declares a top-left origin). Clip space climbs the other way, so RayDirection has to flip Y.
        // Without this test the mistake is invisible to every structural check an example makes -- the
        // frame still has a smooth band and a detailed band, just swapped -- and surfaces only as an
        // upside-down render.
        var camera = FeatherCamera.FromUniforms(SquarePerspectiveInverse(), float3.Zero);
        var size = new float2(64.0f, 64.0f);

        var top = FeatherCamera.RayDirection(camera, new float2(32.0f, 0.0f), size);
        var bottom = FeatherCamera.RayDirection(camera, new float2(32.0f, 63.0f), size);

        Assert.True(top.Y > 0.0f, $"top row aimed at Y={top.Y}");
        Assert.True(bottom.Y < 0.0f, $"bottom row aimed at Y={bottom.Y}");
    }

    [Fact]
    public void PixelXRunsLeftToRight()
    {
        // Pinned alongside the vertical flip because a horizontal mirror survives every tonal
        // assertion the examples make: rows keep their brightness when a frame is mirrored sideways.
        var camera = FeatherCamera.FromUniforms(SquarePerspectiveInverse(), float3.Zero);
        var size = new float2(64.0f, 64.0f);

        var left = FeatherCamera.RayDirection(camera, new float2(0.0f, 32.0f), size);
        var right = FeatherCamera.RayDirection(camera, new float2(63.0f, 32.0f), size);

        Assert.True(left.X < 0.0f, $"left column aimed at X={left.X}");
        Assert.True(right.X > 0.0f, $"right column aimed at X={right.X}");
    }

    [Fact]
    public void RaysAimAwayFromTheCamera()
    {
        // The unprojection subtracts the near point from the far one, and getting that order backwards
        // marches every ray behind the eye -- which reads as an empty frame rather than as a flip.
        var camera = FeatherCamera.FromUniforms(SquarePerspectiveInverse(), float3.Zero);
        var centre = FeatherCamera.RayDirection(camera, new float2(32.0f, 32.0f), new float2(64.0f, 64.0f));

        Assert.True(centre.Z < 0.0f, $"centre ray aimed at Z={centre.Z}");
    }

    [Fact]
    public void ConversionIsItsOwnInverse()
    {
        var camera = new RenderCamera(
            float4x4.Identity,
            float4x4.Identity,
            new float3(3.0f, -4.0f, 5.0f));

        var once = GpuCameraFactory.FromBlender(camera);
        var twice = GpuCameraFactory.FromBlender(
            new RenderCamera(float4x4.Identity, once.InverseViewProjection, once.WorldPosition));

        Assert.Equal(3.0f, twice.WorldPosition.X, 5);
        Assert.Equal(-4.0f, twice.WorldPosition.Y, 5);
        Assert.Equal(5.0f, twice.WorldPosition.Z, 5);
    }
}

/// <summary>
/// Exists so the generator has a kernel that consumes the camera layer the way an effect would.
/// </summary>
/// <summary>
/// The shape the camera layer cannot use, kept so the limitation stays asserted rather than
/// remembered. See <see cref="ShaderCameraTests.CameraStructIsRejectedAsAPushConstant"/>.
/// </summary>
[Kernel]
[ThreadGroupSize(64)]
public readonly partial struct CameraStructUniformKernel(
    ReadWriteBuffer<float4> output,
    Uniform<GpuCamera> camera) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        var eye = camera.Value.WorldPosition;
        output[i] = new float4(eye.X, eye.Y, eye.Z, i);
    }
}

[Kernel]
[ThreadGroupSize(64)]
public readonly partial struct CameraRayKernel(
    ReadWriteBuffer<float4> output,
    Uniform<float4x4> inverseViewProjection,
    Uniform<float3> cameraWorldPosition) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        var size = new float2(64.0f, 64.0f);

        var camera = FeatherCamera.FromUniforms(
            inverseViewProjection.Value,
            cameraWorldPosition.Value);

        var origin = FeatherCamera.RayOrigin(camera);
        var direction = FeatherCamera.RayDirection(camera, new float2(i, i), size);

        output[i] = new float4(
            direction.X,
            direction.Y,
            direction.Z,
            origin.Y);
    }
}
