using Feather.Math;

namespace Feather.Tests;

/// <summary>
/// Covers <see cref="CameraMatrix"/> by the properties its callers depend on rather than by transcribing
/// the matrix entries.
/// </summary>
/// <remarks>
/// A projection is easy to write down wrongly in a way that still renders: the wrong depth range, or an
/// unflipped Y, produces a picture that looks like a picture. So every check below asserts on where a
/// known point lands after projection -- the near plane at clip depth 0, the far plane at 1, a point
/// above the axis at negative clip Y -- because those are the facts a shadow-map comparison and the
/// Vulkan viewport actually rest on.
/// </remarks>
public class CameraMatrixTests
{
    [Fact]
    public void LookAtPutsTheEyeAtTheOriginAndTheTargetDownNegativeZ()
    {
        var eye = new float3(3.0f, -4.0f, 2.5f);
        var target = new float3(-1.0f, 2.0f, 0.5f);
        var view = CameraMatrix.LookAt(eye, target, new float3(0.0f, 0.0f, 1.0f));

        var viewedEye = view * new float4(eye, 1.0f);
        AssertNear(0.0f, viewedEye.X);
        AssertNear(0.0f, viewedEye.Y);
        AssertNear(0.0f, viewedEye.Z);

        // The target sits straight ahead, so it lands on the view axis at minus its distance: this is a
        // right-handed view looking down -Z, the same convention Blender uses for a camera and a lamp.
        var viewedTarget = view * new float4(target, 1.0f);
        AssertNear(0.0f, viewedTarget.X);
        AssertNear(0.0f, viewedTarget.Y);
        AssertNear(-ShaderMath.Length(target - eye), viewedTarget.Z);
    }

    [Fact]
    public void LookAtPreservesDistancesBecauseItIsRigid()
    {
        var view = CameraMatrix.LookAt(
            new float3(2.0f, 1.0f, 6.0f),
            new float3(0.0f, 0.0f, 0.0f),
            new float3(0.0f, 0.0f, 1.0f));

        var first = new float3(-1.5f, 0.25f, 3.0f);
        var second = new float3(2.0f, -3.0f, 1.0f);
        var viewedFirst = view * new float4(first, 1.0f);
        var viewedSecond = view * new float4(second, 1.0f);

        // A view matrix that had picked up a scale or a shear would still place the eye and the target
        // correctly while distorting everything between them, which on a shadow map reads as depths that
        // compare wrongly rather than as a visibly broken image.
        AssertNear(
            ShaderMath.Length(second - first),
            ShaderMath.Length(viewedSecond.XYZ - viewedFirst.XYZ));
    }

    [Fact]
    public void LookTowardsSurvivesAnUpDirectionParallelToTheView()
    {
        // A ceiling lamp aims straight down, so a caller handing over world up hits this every time. The
        // frame's roll is arbitrary here; that it stays a usable basis is not.
        var view = CameraMatrix.LookTowards(
            new float3(0.0f, 0.0f, 4.0f),
            new float3(0.0f, 0.0f, -1.0f),
            new float3(0.0f, 0.0f, 1.0f));

        var floor = view * new float4(0.0f, 0.0f, 0.0f, 1.0f);
        AssertNear(0.0f, floor.X);
        AssertNear(0.0f, floor.Y);
        AssertNear(-4.0f, floor.Z);

        // Still rigid, which a degenerate basis would not be: a zero axis collapses one dimension and
        // the determinant with it.
        AssertNear(1.0f, MathF.Abs(view.Determinant()));
    }

    [Fact]
    public void PerspectiveMapsTheNearPlaneToZeroAndTheFarPlaneToOne()
    {
        const float near = 0.1f;
        const float far = 25.0f;
        var projection = CameraMatrix.Perspective(1.0f, 16.0f / 9.0f, near, far);

        // Depth 0..1 rather than OpenGL's -1..1. EasyGPU's Vulkan viewport is the reason, and a shadow
        // map stores these values directly, so the range is part of the comparison's meaning.
        AssertNear(0.0f, ClipDepth(projection, near));
        AssertNear(1.0f, ClipDepth(projection, far));
        // And monotonic in between, or a depth test would order fragments wrongly.
        var middle = ClipDepth(projection, 5.0f);
        Assert.True(middle > 0.0f && middle < 1.0f, $"Mid-range depth {middle} left the unit range.");
    }

    [Fact]
    public void PerspectiveFlipsYForTheVulkanViewport()
    {
        var projection = CameraMatrix.Perspective(1.0f, 1.0f, 0.1f, 100.0f);

        // A point above the view axis has to land at negative clip Y: clip Y points down here. Getting
        // this wrong renders the scene upside down, which is obvious in a colour image and invisible in
        // a shadow map -- it just samples the wrong texel.
        var above = projection * new float4(0.0f, 1.0f, -4.0f, 1.0f);
        Assert.True(above.Y < 0.0f, $"Expected negative clip Y, got {above.Y}.");
        Assert.True(above.W > 0.0f, $"Expected a point in front of the eye to have positive W, got {above.W}.");
    }

    [Fact]
    public void PerspectivePutsTheVerticalFieldOfViewEdgeOnTheClipBoundary()
    {
        const float fieldOfView = 1.0f;
        var projection = CameraMatrix.Perspective(fieldOfView, 2.0f, 0.1f, 100.0f);

        // The top of the frustum at unit depth, so the field of view means the vertical angle rather than
        // the horizontal one the default sensor fit would give.
        var edgeHeight = MathF.Tan(fieldOfView * 0.5f);
        var edge = projection * new float4(0.0f, edgeHeight, -1.0f, 1.0f);
        AssertNear(-1.0f, edge.Y / edge.W);

        // And the aspect widens the horizontal extent rather than the vertical one.
        var wide = projection * new float4(edgeHeight * 2.0f, 0.0f, -1.0f, 1.0f);
        AssertNear(1.0f, wide.X / wide.W);
    }

    [Fact]
    public void OrthographicKeepsParallelRaysParallelAcrossTheSameDepthRange()
    {
        const float near = 0.5f;
        const float far = 12.0f;
        var projection = CameraMatrix.Orthographic(6.0f, 4.0f, near, far);

        AssertNear(0.0f, OrthographicDepth(projection, near));
        AssertNear(1.0f, OrthographicDepth(projection, far));

        // The half-extents land on the clip boundary, and W stays 1: no divide, which is the whole point
        // for a sun whose rays do not converge.
        var corner = projection * new float4(3.0f, 2.0f, -4.0f, 1.0f);
        AssertNear(1.0f, corner.W);
        AssertNear(1.0f, corner.X);
        AssertNear(-1.0f, corner.Y);

        // Two points at different depths but the same offset project to the same place, which is what
        // distinguishes this from Perspective and the reason a directional light needs it.
        var nearOffset = projection * new float4(1.5f, 1.0f, -2.0f, 1.0f);
        var farOffset = projection * new float4(1.5f, 1.0f, -9.0f, 1.0f);
        AssertNear(nearOffset.X, farOffset.X);
        AssertNear(nearOffset.Y, farOffset.Y);
    }

    [Fact]
    public void AViewProjectionBuiltHereLandsTheSceneInsideTheClipVolume()
    {
        // The composition callers actually use, checked end to end: a lamp above a room, looking down at
        // a point on the floor, has to see that point somewhere inside its own clip volume. Each half
        // being individually right does not guarantee it -- a handedness mismatch between them puts the
        // whole scene behind the eye, and each half's own test still passes.
        var view = CameraMatrix.LookTowards(
            new float3(0.0f, 0.0f, 3.9f),
            new float3(0.0f, 0.0f, -1.0f),
            new float3(0.0f, 0.0f, 1.0f));
        var projection = CameraMatrix.Perspective(1.6f, 1.0f, 0.05f, 20.0f);
        var viewProjection = projection * view;

        var floor = viewProjection * new float4(0.4f, -0.55f, 0.0f, 1.0f);
        Assert.True(floor.W > 0.0f, $"The floor fell behind the lamp: W = {floor.W}.");
        var ndc = floor.XYZ / floor.W;
        Assert.True(MathF.Abs(ndc.X) <= 1.0f, $"Clip X {ndc.X} left the volume.");
        Assert.True(MathF.Abs(ndc.Y) <= 1.0f, $"Clip Y {ndc.Y} left the volume.");
        Assert.True(ndc.Z is >= 0.0f and <= 1.0f, $"Clip depth {ndc.Z} left the unit range.");
    }

    private static float ClipDepth(float4x4 projection, float distance)
    {
        var clip = projection * new float4(0.0f, 0.0f, -distance, 1.0f);
        return clip.Z / clip.W;
    }

    private static float OrthographicDepth(float4x4 projection, float distance)
        => (projection * new float4(0.0f, 0.0f, -distance, 1.0f)).Z;

    private static void AssertNear(float expected, float actual, float tolerance = 0.0001f)
        => Assert.True(MathF.Abs(expected - actual) <= tolerance, $"Expected {expected}, actual {actual}.");
}
