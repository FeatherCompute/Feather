using Feather.Math;
using System.Reflection;

namespace Feather.Tests;

public class MathSurfaceTests
{
    [Fact]
    public void FloatVectorsSupportComponentArithmeticAndNormalization()
    {
        var a = new float3(1, 2, 3);
        var b = new float3(4, 5, 6);

        Assert.Equal(new float3(5, 7, 9), a + b);
        Assert.Equal(new float3(3, 3, 3), b - a);
        Assert.Equal(new float3(4, 10, 18), a * b);
        Assert.Equal(new float3(-1, -2, -3), -a);
        Assert.Equal(new float2(1, 2), a.XY);
        Assert.Equal(32, ShaderMath.Dot(a, b));
        Assert.Equal(new float3(-3, 6, -3), ShaderMath.Cross(a, b));
        AssertNear(1, ShaderMath.Length(ShaderMath.Normalize(a)));
    }

    [Fact]
    public void FloatVectorsExposeCompleteReadSwizzleSurface()
    {
        var f2 = new float2(1, 2);
        var f3 = new float3(3, 4, 5);
        var f4 = new float4(6, 7, 8, 9);

        Assert.Equal(new float2(2, 1), f2.YX);
        Assert.Equal(new float4(2, 1, 2, 1), f2.YXYX);
        Assert.Equal(new float3(5, 3, 4), f3.ZXY);
        Assert.Equal(new float4(5, 4, 3, 5), f3.ZYXZ);
        Assert.Equal(new float2(9, 6), f4.WX);
        Assert.Equal(new float4(9, 8, 7, 6), f4.WZYX);
        Assert.Equal(new float4(8, 7, 6, 9), f4.BGRA);
        Assert.Equal(new float2(2, 1), f2.TS);
        Assert.Equal(new float3(5, 3, 4), f3.PST);
        Assert.Equal(new float4(9, 8, 7, 6), f4.QPTS);
    }

    [Fact]
    public void IntegerAndBooleanVectorsExposeCompleteReadSwizzleSurface()
    {
        var i4 = new int4(1, 2, 3, 4);
        var b4 = new bool4(true, false, true, false);

        Assert.Equal(new int2(2, 1), i4.YX);
        Assert.Equal(new int4(4, 3, 2, 1), i4.WZYX);
        Assert.Equal(new int4(4, 3, 2, 1), i4.QPTS);
        Assert.Equal(new bool3(true, true, false), b4.ZXY);
        Assert.Equal(new bool4(false, true, false, true), b4.WZYX);
        Assert.Equal(new bool4(false, true, false, true), b4.QPTS);
    }

    [Fact]
    public void VectorsExposeCompleteCoordinateColorAndTextureCoordinateReadSwizzleSurface()
    {
        foreach (var scalarType in new[] { "float", "int", "bool" })
        {
            AssertSwizzleSurface(scalarType, 2, "XY", "RG", "ST");
            AssertSwizzleSurface(scalarType, 3, "XYZ", "RGB", "STP");
            AssertSwizzleSurface(scalarType, 4, "XYZW", "RGBA", "STPQ");
        }
    }

    [Fact]
    public void IntegerVectorsSupportBasicArithmetic()
    {
        var a = new int3(1, 2, 3);
        var b = new int3(4, 5, 6);

        Assert.Equal(new int3(5, 7, 9), a + b);
        Assert.Equal(new int3(3, 3, 3), b - a);
        Assert.Equal(new int3(2, 4, 6), a * 2);
        Assert.Equal(new int3(-1, -2, -3), -a);
    }

    [Fact]
    public void MatrixIdentityAndScaleTransformVectors()
    {
        var identity = float4x4.Identity;
        var scale = new float4x4(
            2, 0, 0, 0,
            0, 3, 0, 0,
            0, 0, 4, 0,
            0, 0, 0, 1);
        var value = new float4(1, 2, 3, 1);

        Assert.Equal(value, identity * value);
        Assert.Equal(new float4(2, 6, 12, 1), scale * value);
        Assert.Equal(new float4(2, 6, 12, 1), ShaderMath.Mul(scale, value));
    }

    [Fact]
    public void MatrixAdditionMultiplicationTransposeAndHadamardWork()
    {
        var a = new float2x2(1, 2, 3, 4);
        var b = new float2x2(5, 6, 7, 8);

        Assert.Equal(new float2x2(6, 8, 10, 12), a + b);
        Assert.Equal(new float2x2(23, 34, 31, 46), a * b);
        Assert.Equal(new float2x2(1, 3, 2, 4), a.Transposed());
        Assert.Equal(new float2x2(new float2(5, 12), new float2(21, 32)), float2x2.Hadamard(a, b));
    }

    [Fact]
    public void MatrixDeterminantAndInverseFollowColumnMajorLayout()
    {
        var m2 = new float2x2(4, 7, 2, 6);
        var inv2 = m2.Inverse();
        AssertNear(10, m2.Determinant());
        AssertMatrixNear(float2x2.Identity, m2 * inv2);

        var m3 = new float3x3(
            3, 0, 2,
            2, 0, -2,
            0, 1, 1);
        var inv3 = m3.Inverse();
        AssertNear(10, m3.Determinant());
        AssertMatrixNear(float3x3.Identity, m3 * inv3);

        var m4 = new float4x4(
            4, 0, 0, 0,
            0, 5, 0, 0,
            0, 0, 2, 0,
            0, 0, 0, 1);
        var inv4 = m4.Inverse();
        AssertNear(40, m4.Determinant());
        AssertMatrixNear(float4x4.Identity, m4 * inv4);
    }

    [Fact]
    public void MatrixInverseRejectsSingularMatrices()
    {
        Assert.Throws<InvalidOperationException>(() => new float2x2(1, 2, 2, 4).Inverse());
        Assert.Throws<InvalidOperationException>(() => float3x3.Zero.Inverse());
        Assert.Throws<InvalidOperationException>(() => float4x4.Zero.Inverse());
    }

    [Fact]
    public void ShaderMathSupportsCommonScalarAndVectorIntrinsics()
    {
        AssertNear(0.5f, ShaderMath.Sin(MathF.PI / 6));
        AssertNear(1, ShaderMath.Cos(0));
        AssertNear(8, ShaderMath.Pow(2, 3));
        AssertNear(0.25f, ShaderMath.Fract(2.25f));
        Assert.Equal(new float3(0, 0.5f, 1), ShaderMath.Saturate(new float3(-1, 0.5f, 2)));
        Assert.Equal(new float3(2, 3, 4), ShaderMath.Lerp(new float3(0, 0, 0), new float3(4, 6, 8), 0.5f));
        Assert.Equal(new float2(1, 2), ShaderMath.Clamp(new float2(-1, 3), new float2(1, 1), new float2(2, 2)));
    }

    private static void AssertMatrixNear(float2x2 expected, float2x2 actual)
    {
        AssertNear(expected.M00, actual.M00);
        AssertNear(expected.M10, actual.M10);
        AssertNear(expected.M01, actual.M01);
        AssertNear(expected.M11, actual.M11);
    }

    private static void AssertMatrixNear(float3x3 expected, float3x3 actual)
    {
        AssertNear(expected.M00, actual.M00);
        AssertNear(expected.M10, actual.M10);
        AssertNear(expected.M20, actual.M20);
        AssertNear(expected.M01, actual.M01);
        AssertNear(expected.M11, actual.M11);
        AssertNear(expected.M21, actual.M21);
        AssertNear(expected.M02, actual.M02);
        AssertNear(expected.M12, actual.M12);
        AssertNear(expected.M22, actual.M22);
    }

    private static void AssertMatrixNear(float4x4 expected, float4x4 actual)
    {
        AssertNear(expected.M00, actual.M00);
        AssertNear(expected.M10, actual.M10);
        AssertNear(expected.M20, actual.M20);
        AssertNear(expected.M30, actual.M30);
        AssertNear(expected.M01, actual.M01);
        AssertNear(expected.M11, actual.M11);
        AssertNear(expected.M21, actual.M21);
        AssertNear(expected.M31, actual.M31);
        AssertNear(expected.M02, actual.M02);
        AssertNear(expected.M12, actual.M12);
        AssertNear(expected.M22, actual.M22);
        AssertNear(expected.M32, actual.M32);
        AssertNear(expected.M03, actual.M03);
        AssertNear(expected.M13, actual.M13);
        AssertNear(expected.M23, actual.M23);
        AssertNear(expected.M33, actual.M33);
    }

    private static void AssertNear(float expected, float actual, float tolerance = 0.0001f)
        => Assert.True(MathF.Abs(expected - actual) <= tolerance, $"Expected {expected}, actual {actual}.");

    private static void AssertSwizzleSurface(string scalarType, int componentCount, params string[] aliases)
    {
        var vectorType = typeof(float2).Assembly.GetType($"Feather.Math.{scalarType}{componentCount}", throwOnError: true)!;

        foreach (var alias in aliases)
        {
            for (var length = 1; length <= 4; length++)
            {
                foreach (var swizzle in EnumerateSwizzles(alias, length))
                {
                    var property = vectorType.GetProperty(swizzle, BindingFlags.Instance | BindingFlags.Public);
                    Assert.NotNull(property);

                    var expectedTypeName = length == 1
                        ? scalarType
                        : $"Feather.Math.{scalarType}{length}";
                    var actualTypeName = length == 1
                        ? PrimitiveName(property!.PropertyType)
                        : property!.PropertyType.FullName;
                    Assert.Equal(expectedTypeName, actualTypeName);
                }
            }
        }
    }

    private static string PrimitiveName(Type type)
        => type == typeof(float) ? "float"
            : type == typeof(int) ? "int"
            : type == typeof(bool) ? "bool"
            : type.FullName ?? type.Name;

    private static IEnumerable<string> EnumerateSwizzles(string components, int length)
    {
        if (length == 0)
        {
            yield return string.Empty;
            yield break;
        }

        foreach (var component in components)
        {
            foreach (var suffix in EnumerateSwizzles(components, length - 1))
            {
                yield return component + suffix;
            }
        }
    }
}
