using System.Text.Json;
using Feather.RenderGraph;

namespace Feather.Blender.RenderHost.Tests;

public sealed class PassParameterRangeTests
{
    private const string IterationsGuid = "8f2d3c41-5a6b-4c7d-8e9f-0a1b2c3d4e5f";
    private const string StrengthGuid = "1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d";
    private const string FreeGuid = "9f8e7d6c-5b4a-4392-8180-7f6e5d4c3b2a";

    [Theory]
    [InlineData(1)]
    [InlineData(64)]
    public void DeclaredRangeAcceptsValuesInsideBounds(int iterations)
    {
        var pass = new RangedPass();

        PassMemberBinder.BindParameters(pass, Parameters($"{{\"Iterations\": {iterations}}}"));

        Assert.Equal(iterations, pass.Iterations);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(65)]
    [InlineData(1_000_000)]
    public void DeclaredRangeRejectsIterationCountsOutsideBounds(int iterations)
    {
        // An iterative pass dispatches once per step and a dispatch in flight cannot be
        // cancelled, so a saved graph must not be able to stall the host.
        var pass = new RangedPass();

        var exception = Assert.Throws<InvalidDataException>(
            () => PassMemberBinder.BindParameters(pass, Parameters($"{{\"Iterations\": {iterations}}}")));

        Assert.Contains("Iterations", exception.Message, StringComparison.Ordinal);
        Assert.Equal(8, pass.Iterations);
    }

    [Fact]
    public void DeclaredRangeAppliesToFloatingPointParameters()
    {
        var pass = new RangedPass();

        Assert.Throws<InvalidDataException>(
            () => PassMemberBinder.BindParameters(pass, Parameters("{\"Strength\": 2.5}")));
        PassMemberBinder.BindParameters(pass, Parameters("{\"Strength\": 0.25}"));

        Assert.Equal(0.25f, pass.Strength);
    }

    [Fact]
    public void ParametersWithoutDeclaredRangeAreLeftAlone()
    {
        var pass = new RangedPass();

        PassMemberBinder.BindParameters(pass, Parameters("{\"Unbounded\": -12345}"));

        Assert.Equal(-12345, pass.Unbounded);
    }

    private static JsonElement Parameters(string json)
        => JsonDocument.Parse(json).RootElement;

    private sealed class RangedPass : IComputePass
    {
        [Parameter(IterationsGuid, Min = 1, Max = 64)]
        public int Iterations { get; set; } = 8;

        [Parameter(StrengthGuid, Min = 0.0, Max = 1.0)]
        public float Strength { get; set; } = 0.5f;

        [Parameter(FreeGuid)]
        public int Unbounded { get; set; }

        public void Execute(RenderContext context)
        {
        }
    }
}
