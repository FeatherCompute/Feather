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
    public void CameraUniformsLowerToAPushConstantBlock()
    {
        var glsl = ShaderInspection.GetGLSL<CameraRayKernel>();

        Assert.Contains("layout(push_constant) uniform EasyGPUUniformBlock", glsl, StringComparison.Ordinal);
        Assert.Contains("mat4", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void CameraStructIsRejectedAsAPushConstant()
    {
        // Documents why FromUniforms exists rather than a Uniform<GpuCamera> parameter. The
        // generator accepts a user [GpuStruct] as a push constant, but EasyGPU cannot bind one, so
        // the failure lands at shader build time. If this ever starts passing, the two camera
        // uniforms can collapse back into a single struct parameter.
        var failure = Assert.ThrowsAny<Exception>(
            static () => ShaderInspection.GetGLSL<CameraStructUniformKernel>());

        Assert.Contains("could not be matched to bound native resources", failure.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RayHelpersLowerAsRealFunctions()
    {
        var glsl = ShaderInspection.GetGLSL<CameraRayKernel>();

        // Emitted as functions, not inlined away, which is what makes them shareable at all.
        Assert.Contains("RayDirection", glsl, StringComparison.Ordinal);
        Assert.Contains("RayOrigin", glsl, StringComparison.Ordinal);
        // RayDirection calls UnprojectDepth, so the transitive import has to be pulled in too.
        Assert.Contains("UnprojectDepth", glsl, StringComparison.Ordinal);
        // The struct survives as a struct, which is what lets it pass between those functions.
        Assert.Contains("FromUniforms", glsl, StringComparison.Ordinal);
    }

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
