using Feather.AD;
using Feather.Math;

namespace Feather.Integration.Tests;

public class AutoDiffManagedReadbackTests
{
    [Fact]
    public void GradientSetNormalizesNativeVectorTypeNames()
    {
        var gradients = new GradientSet();

        gradients.RegisterNative("scalar", "float", [1f, 2f]);
        gradients.RegisterNative("v2", "Feather.Math.float2", [1f, 2f, 3f, 4f]);
        gradients.RegisterNative("v3", "global::Feather.Math.float3", [1f, 2f, 3f]);
        gradients.RegisterNative("v4", "Feather.Math.float4", [1f, 2f, 3f, 4f]);

        Assert.Equal([1f, 2f], gradients.Get<float>("scalar"));
        Assert.Equal([new float2(1f, 2f), new float2(3f, 4f)], gradients.Get<float2>("v2"));
        Assert.Equal(new float3(1f, 2f, 3f), Assert.Single(gradients.Get<float3>("v3")));
        Assert.Equal(new float4(1f, 2f, 3f, 4f), Assert.Single(gradients.Get<float4>("v4")));
    }

    [Fact]
    public void GradientSetRejectsUnsupportedNativeTypeAndBadVectorLayout()
    {
        var gradients = new GradientSet();

        Assert.Throws<NotSupportedException>(() => gradients.RegisterNative("bad", "int", [1f]));
        Assert.Throws<InvalidOperationException>(() => gradients.RegisterNative("bad2", "float2", [1f, 2f, 3f]));
        Assert.Throws<InvalidOperationException>(() => gradients.RegisterNative("bad3", "float3", [1f, 2f, 3f, 4f]));
        Assert.Throws<InvalidOperationException>(() => gradients.RegisterNative("bad4", "float4", [1f, 2f, 3f]));
    }
}
