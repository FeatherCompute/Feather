using Feather.Math;
using Feather.RayTracing;
using Feather.Resources;

namespace Feather.Integration.Tests;

/// <summary>
/// Hardware ray tracing integration tests: two triangles at z=0 and z=-2,
/// traced from both sides with TraceClosest, asserting the closest-hit
/// primitive and parametric distance on the Luisa backend.
/// </summary>
public class RayTracingHardwareTests
{
    private const float ExpectedT0 = 5.0f; // z=0 triangle from origin z=+5
    private const float ExpectedT1 = 3.0f; // z=-2 triangle from origin z=-5

    [Fact]
    [Trait("Category", "Gpu")]
    public void TraceClosestReturnsNearestTriangleFromBothDirections()
    {
        using var vertices = GPU.CreateBuffer(
        [
            0f, 0f, 0f, 2f, 0f, 0f, 0f, 2f, 0f,      // tri 0 at z=0
            0f, 0f, -2f, 2f, 0f, -2f, 0f, 2f, -2f,   // tri 1 at z=-2
        ], BufferAccess.ReadOnly);
        using var indices = GPU.CreateBuffer<uint>([0, 1, 2, 3, 4, 5], BufferAccess.ReadOnly);
        using var accel = GPU.CreateAccel((vertices, indices));
        using var result = GPU.CreateBuffer<uint>(5);

        GPU.Dispatch(new RayTracingTraceKernel(accel.AsReadOnly(), result.AsReadWrite()), 1);
        var hits = result.ToArray();

        // From z=+5 looking -z: nearest is the z=0 triangle (prim 0, t 5).
        Assert.Equal(0u, hits[0]);
        Assert.Equal(0u, hits[1]); // instance 0
        Assert.Equal(5u, hits[2]); // (uint)5.0f
        // From z=-5 looking +z: nearest is the z=-2 triangle (prim 1, t 3).
        Assert.Equal(1u, hits[3]);
        Assert.Equal(3u, hits[4]); // (uint)3.0f
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void CreateMeshAndCreateAccelOverloadsBuildTheSameStructure()
    {
        using var vertices = GPU.CreateBuffer(
        [
            0f, 0f, 0f, 2f, 0f, 0f, 0f, 2f, 0f,      // tri 0 at z=0
            0f, 0f, -2f, 2f, 0f, -2f, 0f, 2f, -2f,   // tri 1 at z=-2
        ], BufferAccess.ReadOnly);
        using var indices = GPU.CreateBuffer<uint>([0, 1, 2, 3, 4, 5], BufferAccess.ReadOnly);
        using var mesh = GPU.CreateMesh(vertices, indices);
        using var accel = GPU.CreateAccel(mesh);
        using var result = GPU.CreateBuffer<uint>(5);

        GPU.Dispatch(new RayTracingTraceKernel(accel.AsReadOnly(), result.AsReadWrite()), 1);
        var hits = result.ToArray();

        Assert.Equal(0u, hits[0]); // prim of nearest triangle from z=+5
        Assert.Equal(5u, hits[2]); // (uint)5.0f (hitA.T)
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct RayTracingTraceKernel(
    ReadOnlyAccel accel,
    ReadWriteBuffer<uint> result) : IKernel1D
{
    public void Execute()
    {
        // From z=+5 looking -z: nearest is the z=0 triangle (prim 0, t 5).
        var hitA = accel.TraceClosest(new Ray(new float3(0.5f, 0.5f, 5.0f), new float3(0.0f, 0.0f, -1.0f), 0.0f, 1e30f));
        result[0] = hitA.Prim;
        result[1] = hitA.Inst;
        result[2] = (uint)hitA.T;
        // From z=-5 looking +z: nearest is the z=-2 triangle (prim 1, t 3).
        var hitB = accel.TraceClosest(new Ray(new float3(0.5f, 0.5f, -5.0f), new float3(0.0f, 0.0f, 1.0f), 0.0f, 1e30f));
        result[3] = hitB.Prim;
        result[4] = (uint)hitB.T;
    }
}
