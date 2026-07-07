using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

SampleProof.PrintBackend(GPU.Context);
SampleProof.AssertMonomorphizedGlsl<ShapeDistanceKernel>();

Sphere[] sphereData =
[
    new() { Center = new float3(0, 0, 0), Radius = 1 },
    new() { Center = new float3(0, 0, 0), Radius = 2 },
    new() { Center = new float3(1, 1, 0), Radius = 0.5f },
    new() { Center = new float3(-1, 0, 0), Radius = 1 }
];

GroundPlane[] planeData =
[
    new() { Normal = new float3(0, 1, 0), Distance = 0 },
    new() { Normal = new float3(0, 1, 0), Distance = -1 },
    new() { Normal = new float3(0, 0, 1), Distance = -1 },
    new() { Normal = new float3(1, 0, 0), Distance = 0 }
];

float3[] points =
[
    new(3, 0, 0),
    new(0, 3, 0),
    new(1, 1, 2),
    new(-1, 0, 0)
];

float3[] moves =
[
    new(1, 0, 0),
    new(0, 1, 0),
    new(0, 0, 1),
    new(0, 0, 0)
];

float[] expected = [1, 1, 0.5f, -2];

using var spheres = GPU.CreateBuffer<Sphere>(sphereData, BufferAccess.ReadOnly);
using var planes = GPU.CreateBuffer<GroundPlane>(planeData, BufferAccess.ReadOnly);
using var pointBuffer = GPU.CreateBuffer<float3>(points, BufferAccess.ReadOnly);
using var moveBuffer = GPU.CreateBuffer<float3>(moves, BufferAccess.ReadOnly);
using var output = GPU.CreateBuffer<float>(expected.Length, BufferAccess.ReadWrite);

var path = GPU.DispatchAndGetPath(
    new ShapeDistanceKernel(
        spheres.AsReadOnly(),
        planes.AsReadOnly(),
        pointBuffer.AsReadOnly(),
        moveBuffer.AsReadOnly(),
        output.AsReadWrite()),
    expected.Length);
SampleProof.AssertTypedEasyGpu(path);

var actual = output.ToArray();
AssertNear(expected, actual);
AssertSpheresUnchanged(sphereData, spheres.ToArray());
AssertPlanesUnchanged(planeData, planes.ToArray());

Console.WriteLine("GpuStruct + generic interface sample");
Console.WriteLine($"Dispatch path: {path}");
Console.WriteLine("Combined moved SDF: " + string.Join(", ", actual.Select(static value => value.ToString("0.###"))));
Console.WriteLine("PASS");

static void AssertNear(IReadOnlyList<float> expected, IReadOnlyList<float> actual)
{
    if (expected.Count != actual.Count)
    {
        throw new InvalidOperationException("Result length mismatch.");
    }

    for (var i = 0; i < expected.Count; i++)
    {
        if (MathF.Abs(expected[i] - actual[i]) > 1e-5f)
        {
            throw new InvalidOperationException($"Output[{i}] expected {expected[i]}, got {actual[i]}.");
        }
    }
}

static void AssertSpheresUnchanged(IReadOnlyList<Sphere> expected, IReadOnlyList<Sphere> actual)
{
    for (var i = 0; i < expected.Count; i++)
    {
        if (expected[i].Center != actual[i].Center || MathF.Abs(expected[i].Radius - actual[i].Radius) > 1e-6f)
        {
            throw new InvalidOperationException($"Sphere input[{i}] was unexpectedly modified.");
        }
    }
}

static void AssertPlanesUnchanged(IReadOnlyList<GroundPlane> expected, IReadOnlyList<GroundPlane> actual)
{
    for (var i = 0; i < expected.Count; i++)
    {
        if (expected[i].Normal != actual[i].Normal || MathF.Abs(expected[i].Distance - actual[i].Distance) > 1e-6f)
        {
            throw new InvalidOperationException($"Plane input[{i}] was unexpectedly modified.");
        }
    }
}

public interface ISignedDistanceShape
{
    void Move(float3 delta);
    float Sdf(float3 point);
}

[GpuStruct]
public partial struct Sphere : ISignedDistanceShape
{
    public float3 Center;
    public float Radius;

    [Callable]
    public void Move(float3 delta)
    {
        Center += delta;
    }

    [Callable]
    public float Sdf(float3 point)
    {
        return ShaderMath.Length(point - Center) - Radius;
    }
}

[GpuStruct]
public partial struct GroundPlane : ISignedDistanceShape
{
    public float3 Normal;
    public float Distance;

    [Callable]
    public void Move(float3 delta)
    {
        Distance -= ShaderMath.Dot(Normal, delta);
    }

    [Callable]
    public float Sdf(float3 point)
    {
        return ShaderMath.Dot(Normal, point) + Distance;
    }
}

[ShaderLibrary]
public static class ShapeOps
{
    [Callable]
    public static float MovedSdf<TShape>(TShape shape, float3 point, float3 delta)
        where TShape : ISignedDistanceShape
    {
        shape.Move(delta);
        return shape.Sdf(point);
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct ShapeDistanceKernel(
    ReadOnlyBuffer<Sphere> spheres,
    ReadOnlyBuffer<GroundPlane> planes,
    ReadOnlyBuffer<float3> points,
    ReadOnlyBuffer<float3> moves,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float3 point = points[i];
        float3 delta = moves[i];
        float sphereDistance = ShapeOps.MovedSdf(spheres[i], point, delta);
        float planeDistance = ShapeOps.MovedSdf(planes[i], point, delta);
        output[i] = sphereDistance + planeDistance;
    }
}

internal static class SampleProof
{
    public static void PrintBackend(GpuContext context)
    {
        var caps = context.Caps;
        Console.WriteLine($"Backend: {caps.BackendType}");
        Console.WriteLine($"Max workgroup size: {caps.MaxWorkGroupSizeX}x{caps.MaxWorkGroupSizeY}x{caps.MaxWorkGroupSizeZ}");
    }

    public static void AssertMonomorphizedGlsl<TKernel>()
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        var glsl = ShaderInspection.GetGLSL<TKernel>();
        string[] required =
        [
            "ShapeOps_MovedSdf_T_global__Sphere",
            "ShapeOps_MovedSdf_T_global__GroundPlane",
            "inout Sphere fe_this",
            "inout GroundPlane fe_this",
            "Sphere_Sdf",
            "GroundPlane_Sdf"
        ];

        foreach (var marker in required)
        {
            if (!glsl.Contains(marker, StringComparison.Ordinal))
            {
                throw new InvalidOperationException($"Generated GLSL did not contain '{marker}'.");
            }
        }

        if (glsl.Contains("ISignedDistanceShape", StringComparison.Ordinal) ||
            glsl.Contains("switch", StringComparison.Ordinal) ||
            glsl.Contains("Feather native stub", StringComparison.Ordinal))
        {
            throw new InvalidOperationException("Expected generic monomorphization, not interface dispatch or fallback GLSL.");
        }

        Console.WriteLine("Generic interface monomorphization: OK");
    }

    public static void AssertTypedEasyGpu(DispatchPath path)
    {
        if (path != DispatchPath.TypedEasyGpu)
        {
            throw new InvalidOperationException($"Expected TypedEasyGpu dispatch, got {path}.");
        }
    }
}
