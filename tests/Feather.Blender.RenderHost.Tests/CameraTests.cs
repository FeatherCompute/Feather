using Feather.Math;

namespace Feather.Blender.RenderHost.Tests;

public sealed class CameraTests
{
    // A camera at (0, 0, 10) looking down -Z with a square OpenGL perspective. Built once and
    // reused so both the round-trip and the ray tests exercise a realistic view-projection.
    private static readonly float4x4 View = FromRowMajor(
    [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, -10,
        0, 0, 0, 1
    ]);

    private static readonly float4x4 Projection = Perspective(
        fovYRadians: MathF.PI / 3.0f, aspect: 1.0f, near: 0.1f, far: 100.0f);

    [Fact]
    public void InverseViewProjectionUndoesTheRawViewProjection()
    {
        var rawViewProjection = ToRowMajor(Projection * View);
        using var fixture = new ProtocolFixture();
        fixture.WriteRequest(clipSpace: "blender-opengl", viewProjection: rawViewProjection);

        var resolved = RenderRequest.Load(fixture.RequestPath);

        // The stored ViewProjection carries the vulkan viewport fixup, so only the raw matrix is a
        // mutual inverse of InverseViewProjection.
        var product = MatrixProtocol.FromRowMajor(rawViewProjection) * resolved.InverseViewProjection;
        AssertApproximatelyIdentity(product);
    }

    [Fact]
    public void ScreenCentreUnprojectsToAForwardRay()
    {
        var rawViewProjection = ToRowMajor(Projection * View);
        using var fixture = new ProtocolFixture();
        fixture.WriteRequest(
            clipSpace: "blender-opengl",
            viewProjection: rawViewProjection,
            cameraPosition: [0.0f, 0.0f, 10.0f]);

        var resolved = RenderRequest.Load(fixture.RequestPath);
        var invVp = resolved.InverseViewProjection;

        // Screen centre in OpenGL NDC is (0, 0); unproject the near and far plane points.
        var near = invVp * new float4(0.0f, 0.0f, -1.0f, 1.0f);
        var far = invVp * new float4(0.0f, 0.0f, 1.0f, 1.0f);
        var nearWorld = new float3(near.X / near.W, near.Y / near.W, near.Z / near.W);
        var farWorld = new float3(far.X / far.W, far.Y / far.W, far.Z / far.W);
        var direction = Normalize(farWorld - nearWorld);

        Assert.True(MathF.Abs(direction.X) < 1e-3f, $"ray X should be ~0, was {direction.X}");
        Assert.True(MathF.Abs(direction.Y) < 1e-3f, $"ray Y should be ~0, was {direction.Y}");
        Assert.True(direction.Z < -0.99f, $"ray should point down -Z, was {direction.Z}");

        // The near plane point sits directly in front of the eye at (0, 0, 10).
        Assert.True(MathF.Abs(nearWorld.X) < 1e-3f);
        Assert.True(MathF.Abs(nearWorld.Y) < 1e-3f);
        Assert.True(nearWorld.Z < 10.0f, "near plane should be in front of the eye");
    }

    [Fact]
    public void OffCentrePixelSkewsTheRayInTheExpectedDirection()
    {
        var rawViewProjection = ToRowMajor(Projection * View);
        using var fixture = new ProtocolFixture();
        fixture.WriteRequest(clipSpace: "blender-opengl", viewProjection: rawViewProjection);

        var invVp = RenderRequest.Load(fixture.RequestPath).InverseViewProjection;

        // NDC (+1, +1) is the top-right corner: the ray should tilt toward +X and +Y.
        var near = invVp * new float4(0.5f, 0.5f, -1.0f, 1.0f);
        var far = invVp * new float4(0.5f, 0.5f, 1.0f, 1.0f);
        var nearWorld = new float3(near.X / near.W, near.Y / near.W, near.Z / near.W);
        var farWorld = new float3(far.X / far.W, far.Y / far.W, far.Z / far.W);
        var direction = Normalize(farWorld - nearWorld);

        Assert.True(direction.X > 0.0f, $"ray should tilt toward +X, was {direction.X}");
        Assert.True(direction.Y > 0.0f, $"ray should tilt toward +Y, was {direction.Y}");
        Assert.True(direction.Z < 0.0f, $"ray should still travel into the scene, was {direction.Z}");
    }

    [Fact]
    public void CameraPositionIsParsedWhenPresent()
    {
        var rawViewProjection = ToRowMajor(Projection * View);
        using var fixture = new ProtocolFixture();
        fixture.WriteRequest(
            clipSpace: "blender-opengl",
            viewProjection: rawViewProjection,
            cameraPosition: [1.5f, -2.0f, 10.0f]);

        var resolved = RenderRequest.Load(fixture.RequestPath);

        Assert.Equal(1.5f, resolved.CameraPosition.X, 5);
        Assert.Equal(-2.0f, resolved.CameraPosition.Y, 5);
        Assert.Equal(10.0f, resolved.CameraPosition.Z, 5);
    }

    [Fact]
    public void CameraPositionDefaultsWhenAbsent()
    {
        var rawViewProjection = ToRowMajor(Projection * View);
        using var fixture = new ProtocolFixture();
        fixture.WriteRequest(clipSpace: "blender-opengl", viewProjection: rawViewProjection);

        var resolved = RenderRequest.Load(fixture.RequestPath);

        // Absent cameraPosition falls back to the unprojected near-plane point, which stays finite
        // and in front of the eye rather than throwing.
        Assert.True(float.IsFinite(resolved.CameraPosition.X));
        Assert.True(float.IsFinite(resolved.CameraPosition.Y));
        Assert.True(float.IsFinite(resolved.CameraPosition.Z));
        Assert.True(resolved.CameraPosition.Z < 10.0f);
    }

    private static void AssertApproximatelyIdentity(float4x4 matrix)
    {
        var identity = float4x4.Identity;
        AssertColumn(matrix.C0, identity.C0);
        AssertColumn(matrix.C1, identity.C1);
        AssertColumn(matrix.C2, identity.C2);
        AssertColumn(matrix.C3, identity.C3);

        static void AssertColumn(float4 actual, float4 expected)
        {
            Assert.Equal(expected.X, actual.X, 4);
            Assert.Equal(expected.Y, actual.Y, 4);
            Assert.Equal(expected.Z, actual.Z, 4);
            Assert.Equal(expected.W, actual.W, 4);
        }
    }

    private static float3 Normalize(float3 value)
    {
        var length = MathF.Sqrt((value.X * value.X) + (value.Y * value.Y) + (value.Z * value.Z));
        return new float3(value.X / length, value.Y / length, value.Z / length);
    }

    private static float4x4 Perspective(float fovYRadians, float aspect, float near, float far)
    {
        var f = 1.0f / MathF.Tan(fovYRadians * 0.5f);
        return FromRowMajor(
        [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) / (near - far), (2.0f * far * near) / (near - far),
            0, 0, -1, 0
        ]);
    }

    private static float4x4 FromRowMajor(float[] values)
        => MatrixProtocol.FromRowMajor(values);

    private static float[] ToRowMajor(float4x4 matrix)
    {
        var columns = new[] { matrix.C0, matrix.C1, matrix.C2, matrix.C3 };
        var values = new float[16];
        for (var column = 0; column < 4; column++)
        {
            values[(0 * 4) + column] = columns[column].X;
            values[(1 * 4) + column] = columns[column].Y;
            values[(2 * 4) + column] = columns[column].Z;
            values[(3 * 4) + column] = columns[column].W;
        }
        return values;
    }
}
