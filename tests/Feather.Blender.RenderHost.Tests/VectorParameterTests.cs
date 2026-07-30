using System.Text.Json;
using Feather.Math;
using Feather.RenderGraph;

namespace Feather.Blender.RenderHost.Tests;

public sealed class VectorParameterTests
{
    private const string TintGuid = "5c3a8e17-9d42-4b6f-8175-2e9c6a4b3d58";
    private const string ExtentGuid = "7f1d4b92-6a35-4e87-9c24-8b3f5d1a6e94";
    private const string PlaneGuid = "2b8e6c45-3f19-4d72-a586-1c9d7b4e3a62";
    private const string ExposureGuid = "9a4c7e23-8b51-4f96-8d37-6e2b1a5c4f89";

    [Fact]
    public void VectorParametersBindFromJsonArrays()
    {
        // Blender serialises a vector parameter as a JSON array, because that is what its
        // FloatVectorProperty produces. Area-light extents and tint colours are the motivating case.
        var pass = new VectorPass();

        PassMemberBinder.BindParameters(
            pass,
            Parameters("""
            {
                "Tint": [0.25, 0.5, 0.75],
                "Extent": [2.0, 3.5],
                "Plane": [1.0, 0.0, 0.0, -4.0],
                "Exposure": 1.5
            }
            """));

        Assert.Equal(new float3(0.25f, 0.5f, 0.75f), pass.Tint);
        Assert.Equal(new float2(2.0f, 3.5f), pass.Extent);
        Assert.Equal(new float4(1.0f, 0.0f, 0.0f, -4.0f), pass.Plane);
        Assert.Equal(1.5f, pass.Exposure);
    }

    [Fact]
    public void VectorParametersRejectWrongComponentCounts()
    {
        var pass = new VectorPass();

        // A truncated array would otherwise bind silently with a zero in the missing slot.
        Assert.Throws<InvalidDataException>(
            () => PassMemberBinder.BindParameters(pass, Parameters("""{"Tint": [0.5, 0.5]}""")));
        Assert.Throws<InvalidDataException>(
            () => PassMemberBinder.BindParameters(
                pass, Parameters("""{"Tint": [0.5, 0.5, 0.5, 0.5]}""")));
    }

    [Fact]
    public void VectorParametersRejectNonNumericComponents()
    {
        var pass = new VectorPass();

        Assert.Throws<InvalidDataException>(
            () => PassMemberBinder.BindParameters(
                pass, Parameters("""{"Tint": [0.5, "green", 0.5]}""")));
    }

    [Fact]
    public void ScalarParametersStillBind()
    {
        // Vector support must not disturb the existing scalar path.
        var pass = new VectorPass();

        PassMemberBinder.BindParameters(pass, Parameters("""{"Exposure": 2.25}"""));

        Assert.Equal(2.25f, pass.Exposure);
        Assert.Equal(default, pass.Tint);
    }

    private static JsonElement Parameters(string json)
        => JsonDocument.Parse(json).RootElement;

    private sealed class VectorPass : IComputePass
    {
        [Parameter(TintGuid)]
        public float3 Tint { get; set; }

        [Parameter(ExtentGuid)]
        public float2 Extent { get; set; }

        [Parameter(PlaneGuid)]
        public float4 Plane { get; set; }

        [Parameter(ExposureGuid, Min = 0.0, Max = 8.0)]
        public float Exposure { get; set; } = 1.0f;

        public void Execute(RenderContext context)
        {
        }
    }
}
