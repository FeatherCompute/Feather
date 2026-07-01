namespace Feather.Math;

/// <summary>
/// Represents a two-component signed integer vector.
/// </summary>
/// <param name="X">The first component.</param>
/// <param name="Y">The second component.</param>
public readonly partial record struct int2(int X, int Y)
{
    /// <summary>
    /// Gets a vector with all components set to zero.
    /// </summary>
    public static int2 Zero => new(0, 0);

    /// <summary>
    /// Adds two vectors component-wise.
    /// </summary>
    public static int2 operator +(int2 left, int2 right) => new(left.X + right.X, left.Y + right.Y);

    /// <summary>
    /// Subtracts two vectors component-wise.
    /// </summary>
    public static int2 operator -(int2 left, int2 right) => new(left.X - right.X, left.Y - right.Y);

    /// <summary>
    /// Negates each component.
    /// </summary>
    public static int2 operator -(int2 value) => new(-value.X, -value.Y);

    /// <summary>
    /// Multiplies each component by a scalar.
    /// </summary>
    public static int2 operator *(int2 left, int right) => new(left.X * right, left.Y * right);

    /// <summary>
    /// Multiplies each component by a scalar.
    /// </summary>
    public static int2 operator *(int left, int2 right) => right * left;

    /// <summary>
    /// Divides each component by a scalar.
    /// </summary>
    public static int2 operator /(int2 left, int right) => new(left.X / right, left.Y / right);
}

/// <summary>
/// Represents a three-component signed integer vector.
/// </summary>
/// <param name="X">The first component.</param>
/// <param name="Y">The second component.</param>
/// <param name="Z">The third component.</param>
public readonly partial record struct int3(int X, int Y, int Z)
{
    /// <summary>
    /// Gets a vector with all components set to zero.
    /// </summary>
    public static int3 Zero => new(0, 0, 0);

    /// <summary>
    /// Adds two vectors component-wise.
    /// </summary>
    public static int3 operator +(int3 left, int3 right) => new(left.X + right.X, left.Y + right.Y, left.Z + right.Z);

    /// <summary>
    /// Subtracts two vectors component-wise.
    /// </summary>
    public static int3 operator -(int3 left, int3 right) => new(left.X - right.X, left.Y - right.Y, left.Z - right.Z);

    /// <summary>
    /// Negates each component.
    /// </summary>
    public static int3 operator -(int3 value) => new(-value.X, -value.Y, -value.Z);

    /// <summary>
    /// Multiplies each component by a scalar.
    /// </summary>
    public static int3 operator *(int3 left, int right) => new(left.X * right, left.Y * right, left.Z * right);

    /// <summary>
    /// Multiplies each component by a scalar.
    /// </summary>
    public static int3 operator *(int left, int3 right) => right * left;

    /// <summary>
    /// Divides each component by a scalar.
    /// </summary>
    public static int3 operator /(int3 left, int right) => new(left.X / right, left.Y / right, left.Z / right);
}

/// <summary>
/// Represents a four-component signed integer vector.
/// </summary>
/// <param name="X">The first component.</param>
/// <param name="Y">The second component.</param>
/// <param name="Z">The third component.</param>
/// <param name="W">The fourth component.</param>
public readonly partial record struct int4(int X, int Y, int Z, int W)
{
    /// <summary>
    /// Gets a vector with all components set to zero.
    /// </summary>
    public static int4 Zero => new(0, 0, 0, 0);

    /// <summary>
    /// Adds two vectors component-wise.
    /// </summary>
    public static int4 operator +(int4 left, int4 right) => new(left.X + right.X, left.Y + right.Y, left.Z + right.Z, left.W + right.W);

    /// <summary>
    /// Subtracts two vectors component-wise.
    /// </summary>
    public static int4 operator -(int4 left, int4 right) => new(left.X - right.X, left.Y - right.Y, left.Z - right.Z, left.W - right.W);

    /// <summary>
    /// Negates each component.
    /// </summary>
    public static int4 operator -(int4 value) => new(-value.X, -value.Y, -value.Z, -value.W);

    /// <summary>
    /// Multiplies each component by a scalar.
    /// </summary>
    public static int4 operator *(int4 left, int right) => new(left.X * right, left.Y * right, left.Z * right, left.W * right);

    /// <summary>
    /// Multiplies each component by a scalar.
    /// </summary>
    public static int4 operator *(int left, int4 right) => right * left;

    /// <summary>
    /// Divides each component by a scalar.
    /// </summary>
    public static int4 operator /(int4 left, int right) => new(left.X / right, left.Y / right, left.Z / right, left.W / right);
}

/// <summary>
/// Represents a two-component Boolean vector.
/// </summary>
/// <param name="X">The first component.</param>
/// <param name="Y">The second component.</param>
public readonly partial record struct bool2(bool X, bool Y);

/// <summary>
/// Represents a three-component Boolean vector.
/// </summary>
/// <param name="X">The first component.</param>
/// <param name="Y">The second component.</param>
/// <param name="Z">The third component.</param>
public readonly partial record struct bool3(bool X, bool Y, bool Z);

/// <summary>
/// Represents a four-component Boolean vector.
/// </summary>
/// <param name="X">The first component.</param>
/// <param name="Y">The second component.</param>
/// <param name="Z">The third component.</param>
/// <param name="W">The fourth component.</param>
public readonly partial record struct bool4(bool X, bool Y, bool Z, bool W);

/// <summary>
/// Represents a two-component single-precision floating-point vector.
/// </summary>
/// <param name="X">The first component.</param>
/// <param name="Y">The second component.</param>
public readonly partial record struct float2(float X, float Y)
{
    /// <summary>
    /// Creates a vector with all components initialized to the same value.
    /// </summary>
    /// <param name="value">The value used for every component.</param>
    public float2(float value) : this(value, value)
    {
    }

    /// <summary>
    /// Gets the red color alias for <see cref="X"/>.
    /// </summary>
    public float R => X;

    /// <summary>
    /// Gets the green color alias for <see cref="Y"/>.
    /// </summary>
    public float G => Y;

    /// <summary>
    /// Gets this vector as an XY swizzle.
    /// </summary>
    public float2 XY => this;

    /// <summary>
    /// Gets this vector as an RG swizzle.
    /// </summary>
    public float2 RG => this;

    /// <summary>
    /// Gets a vector with all components set to zero.
    /// </summary>
    public static float2 Zero => new(0, 0);

    /// <summary>
    /// Gets a vector with all components set to one.
    /// </summary>
    public static float2 One => new(1, 1);

    /// <summary>
    /// Adds two vectors component-wise.
    /// </summary>
    public static float2 operator +(float2 left, float2 right) => new(left.X + right.X, left.Y + right.Y);

    /// <summary>
    /// Subtracts two vectors component-wise.
    /// </summary>
    public static float2 operator -(float2 left, float2 right) => new(left.X - right.X, left.Y - right.Y);

    /// <summary>
    /// Negates each component.
    /// </summary>
    public static float2 operator -(float2 value) => new(-value.X, -value.Y);

    /// <summary>
    /// Multiplies two vectors component-wise.
    /// </summary>
    public static float2 operator *(float2 left, float2 right) => new(left.X * right.X, left.Y * right.Y);

    /// <summary>
    /// Multiplies each component by a scalar.
    /// </summary>
    public static float2 operator *(float2 left, float right) => new(left.X * right, left.Y * right);

    /// <summary>
    /// Multiplies each component by a scalar.
    /// </summary>
    public static float2 operator *(float right, float2 left) => left * right;

    /// <summary>
    /// Divides two vectors component-wise.
    /// </summary>
    public static float2 operator /(float2 left, float2 right) => new(left.X / right.X, left.Y / right.Y);

    /// <summary>
    /// Divides each component by a scalar.
    /// </summary>
    public static float2 operator /(float2 left, float right) => new(left.X / right, left.Y / right);

    /// <summary>
    /// Returns a culture-invariant component representation of this vector.
    /// </summary>
    public override string ToString() => $"float2({X}, {Y})";
}

/// <summary>
/// Represents a three-component single-precision floating-point vector.
/// </summary>
/// <param name="X">The first component.</param>
/// <param name="Y">The second component.</param>
/// <param name="Z">The third component.</param>
public readonly partial record struct float3(float X, float Y, float Z)
{
    /// <summary>
    /// Creates a vector with all components initialized to the same value.
    /// </summary>
    /// <param name="value">The value used for every component.</param>
    public float3(float value) : this(value, value, value)
    {
    }

    /// <summary>
    /// Creates a vector from an XY pair and a Z component.
    /// </summary>
    /// <param name="xy">The first two components.</param>
    /// <param name="z">The third component.</param>
    public float3(float2 xy, float z) : this(xy.X, xy.Y, z)
    {
    }

    /// <summary>
    /// Gets the red color alias for <see cref="X"/>.
    /// </summary>
    public float R => X;

    /// <summary>
    /// Gets the green color alias for <see cref="Y"/>.
    /// </summary>
    public float G => Y;

    /// <summary>
    /// Gets the blue color alias for <see cref="Z"/>.
    /// </summary>
    public float B => Z;

    /// <summary>
    /// Gets the XY swizzle.
    /// </summary>
    public float2 XY => new(X, Y);

    /// <summary>
    /// Gets the RG swizzle.
    /// </summary>
    public float2 RG => new(R, G);

    /// <summary>
    /// Gets this vector as an XYZ swizzle.
    /// </summary>
    public float3 XYZ => this;

    /// <summary>
    /// Gets this vector as an RGB swizzle.
    /// </summary>
    public float3 RGB => this;

    /// <summary>
    /// Gets a vector with all components set to zero.
    /// </summary>
    public static float3 Zero => new(0, 0, 0);

    /// <summary>
    /// Gets a vector with all components set to one.
    /// </summary>
    public static float3 One => new(1, 1, 1);

    /// <summary>
    /// Adds two vectors component-wise.
    /// </summary>
    public static float3 operator +(float3 left, float3 right) => new(left.X + right.X, left.Y + right.Y, left.Z + right.Z);

    /// <summary>
    /// Subtracts two vectors component-wise.
    /// </summary>
    public static float3 operator -(float3 left, float3 right) => new(left.X - right.X, left.Y - right.Y, left.Z - right.Z);

    /// <summary>
    /// Negates each component.
    /// </summary>
    public static float3 operator -(float3 value) => new(-value.X, -value.Y, -value.Z);

    /// <summary>
    /// Multiplies two vectors component-wise.
    /// </summary>
    public static float3 operator *(float3 left, float3 right) => new(left.X * right.X, left.Y * right.Y, left.Z * right.Z);

    /// <summary>
    /// Multiplies each component by a scalar.
    /// </summary>
    public static float3 operator *(float3 left, float right) => new(left.X * right, left.Y * right, left.Z * right);

    /// <summary>
    /// Multiplies each component by a scalar.
    /// </summary>
    public static float3 operator *(float right, float3 left) => left * right;

    /// <summary>
    /// Divides two vectors component-wise.
    /// </summary>
    public static float3 operator /(float3 left, float3 right) => new(left.X / right.X, left.Y / right.Y, left.Z / right.Z);

    /// <summary>
    /// Divides each component by a scalar.
    /// </summary>
    public static float3 operator /(float3 left, float right) => new(left.X / right, left.Y / right, left.Z / right);

    /// <summary>
    /// Returns a culture-invariant component representation of this vector.
    /// </summary>
    public override string ToString() => $"float3({X}, {Y}, {Z})";
}

/// <summary>
/// Represents a four-component single-precision floating-point vector.
/// </summary>
/// <param name="X">The first component.</param>
/// <param name="Y">The second component.</param>
/// <param name="Z">The third component.</param>
/// <param name="W">The fourth component.</param>
public readonly partial record struct float4(float X, float Y, float Z, float W)
{
    /// <summary>
    /// Creates a vector with all components initialized to the same value.
    /// </summary>
    /// <param name="value">The value used for every component.</param>
    public float4(float value) : this(value, value, value, value)
    {
    }

    /// <summary>
    /// Creates a vector from an XYZ triple and a W component.
    /// </summary>
    /// <param name="xyz">The first three components.</param>
    /// <param name="w">The fourth component.</param>
    public float4(float3 xyz, float w) : this(xyz.X, xyz.Y, xyz.Z, w)
    {
    }

    /// <summary>
    /// Creates a vector from an XY pair plus Z and W components.
    /// </summary>
    /// <param name="xy">The first two components.</param>
    /// <param name="z">The third component.</param>
    /// <param name="w">The fourth component.</param>
    public float4(float2 xy, float z, float w) : this(xy.X, xy.Y, z, w)
    {
    }

    /// <summary>
    /// Gets the red color alias for <see cref="X"/>.
    /// </summary>
    public float R => X;

    /// <summary>
    /// Gets the green color alias for <see cref="Y"/>.
    /// </summary>
    public float G => Y;

    /// <summary>
    /// Gets the blue color alias for <see cref="Z"/>.
    /// </summary>
    public float B => Z;

    /// <summary>
    /// Gets the alpha color alias for <see cref="W"/>.
    /// </summary>
    public float A => W;

    /// <summary>
    /// Gets the XY swizzle.
    /// </summary>
    public float2 XY => new(X, Y);

    /// <summary>
    /// Gets the ZW swizzle.
    /// </summary>
    public float2 ZW => new(Z, W);

    /// <summary>
    /// Gets the XYZ swizzle.
    /// </summary>
    public float3 XYZ => new(X, Y, Z);

    /// <summary>
    /// Gets the RGB swizzle.
    /// </summary>
    public float3 RGB => new(R, G, B);

    /// <summary>
    /// Gets this vector as an RGBA swizzle.
    /// </summary>
    public float4 RGBA => this;

    /// <summary>
    /// Gets a vector with all components set to zero.
    /// </summary>
    public static float4 Zero => new(0, 0, 0, 0);

    /// <summary>
    /// Gets a vector with all components set to one.
    /// </summary>
    public static float4 One => new(1, 1, 1, 1);

    /// <summary>
    /// Adds two vectors component-wise.
    /// </summary>
    public static float4 operator +(float4 left, float4 right) => new(left.X + right.X, left.Y + right.Y, left.Z + right.Z, left.W + right.W);

    /// <summary>
    /// Subtracts two vectors component-wise.
    /// </summary>
    public static float4 operator -(float4 left, float4 right) => new(left.X - right.X, left.Y - right.Y, left.Z - right.Z, left.W - right.W);

    /// <summary>
    /// Negates each component.
    /// </summary>
    public static float4 operator -(float4 value) => new(-value.X, -value.Y, -value.Z, -value.W);

    /// <summary>
    /// Multiplies two vectors component-wise.
    /// </summary>
    public static float4 operator *(float4 left, float4 right) => new(left.X * right.X, left.Y * right.Y, left.Z * right.Z, left.W * right.W);

    /// <summary>
    /// Multiplies each component by a scalar.
    /// </summary>
    public static float4 operator *(float4 left, float right) => new(left.X * right, left.Y * right, left.Z * right, left.W * right);

    /// <summary>
    /// Multiplies each component by a scalar.
    /// </summary>
    public static float4 operator *(float right, float4 left) => left * right;

    /// <summary>
    /// Divides two vectors component-wise.
    /// </summary>
    public static float4 operator /(float4 left, float4 right) => new(left.X / right.X, left.Y / right.Y, left.Z / right.Z, left.W / right.W);

    /// <summary>
    /// Divides each component by a scalar.
    /// </summary>
    public static float4 operator /(float4 left, float right) => new(left.X / right, left.Y / right, left.Z / right, left.W / right);

    /// <summary>
    /// Returns a culture-invariant component representation of this vector.
    /// </summary>
    public override string ToString() => $"float4({X}, {Y}, {Z}, {W})";
}

/// <summary>
/// Represents a column-major 2x2 single-precision floating-point matrix.
/// </summary>
/// <param name="C0">The first column.</param>
/// <param name="C1">The second column.</param>
public readonly record struct float2x2(float2 C0, float2 C1)
{
    private const float SingularEpsilon = 1e-8f;

    /// <summary>
    /// Creates a matrix from column-major components.
    /// </summary>
    public float2x2(float m00, float m10, float m01, float m11)
        : this(new float2(m00, m10), new float2(m01, m11))
    {
    }

    /// <summary>
    /// Gets the identity matrix.
    /// </summary>
    public static float2x2 Identity => new(new float2(1, 0), new float2(0, 1));

    /// <summary>
    /// Gets the zero matrix.
    /// </summary>
    public static float2x2 Zero => new(float2.Zero, float2.Zero);

    /// <summary>
    /// Gets element row 0, column 0.
    /// </summary>
    public float M00 => C0.X;

    /// <summary>
    /// Gets element row 1, column 0.
    /// </summary>
    public float M10 => C0.Y;

    /// <summary>
    /// Gets element row 0, column 1.
    /// </summary>
    public float M01 => C1.X;

    /// <summary>
    /// Gets element row 1, column 1.
    /// </summary>
    public float M11 => C1.Y;

    /// <summary>
    /// Returns the transposed matrix.
    /// </summary>
    public float2x2 Transposed() => new(new float2(M00, M01), new float2(M10, M11));

    /// <summary>
    /// Returns the determinant of the matrix.
    /// </summary>
    public float Determinant() => (M00 * M11) - (M01 * M10);

    /// <summary>
    /// Returns the inverse matrix.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when the matrix is singular.</exception>
    public float2x2 Inverse()
    {
        var determinant = Determinant();
        if (System.MathF.Abs(determinant) <= SingularEpsilon)
        {
            throw new InvalidOperationException("Cannot invert a singular float2x2 matrix.");
        }

        return new float2x2(new float2(M11, -M10), new float2(-M01, M00)) / determinant;
    }

    /// <summary>
    /// Adds two matrices component-wise.
    /// </summary>
    public static float2x2 operator +(float2x2 left, float2x2 right) => new(left.C0 + right.C0, left.C1 + right.C1);

    /// <summary>
    /// Subtracts two matrices component-wise.
    /// </summary>
    public static float2x2 operator -(float2x2 left, float2x2 right) => new(left.C0 - right.C0, left.C1 - right.C1);

    /// <summary>
    /// Negates every matrix element.
    /// </summary>
    public static float2x2 operator -(float2x2 value) => new(-value.C0, -value.C1);

    /// <summary>
    /// Multiplies a matrix by a scalar.
    /// </summary>
    public static float2x2 operator *(float2x2 left, float right) => new(left.C0 * right, left.C1 * right);

    /// <summary>
    /// Multiplies a matrix by a scalar.
    /// </summary>
    public static float2x2 operator *(float left, float2x2 right) => right * left;

    /// <summary>
    /// Multiplies a matrix by a vector.
    /// </summary>
    public static float2 operator *(float2x2 left, float2 right) => (left.C0 * right.X) + (left.C1 * right.Y);

    /// <summary>
    /// Multiplies two matrices.
    /// </summary>
    public static float2x2 operator *(float2x2 left, float2x2 right) => new(left * right.C0, left * right.C1);

    /// <summary>
    /// Divides a matrix by a scalar.
    /// </summary>
    public static float2x2 operator /(float2x2 left, float right) => new(left.C0 / right, left.C1 / right);

    /// <summary>
    /// Multiplies two matrices component-wise.
    /// </summary>
    public static float2x2 Hadamard(float2x2 left, float2x2 right) => new(left.C0 * right.C0, left.C1 * right.C1);
}

/// <summary>
/// Represents a column-major 3x3 single-precision floating-point matrix.
/// </summary>
/// <param name="C0">The first column.</param>
/// <param name="C1">The second column.</param>
/// <param name="C2">The third column.</param>
public readonly record struct float3x3(float3 C0, float3 C1, float3 C2)
{
    private const float SingularEpsilon = 1e-8f;

    /// <summary>
    /// Creates a matrix from column-major components.
    /// </summary>
    public float3x3(float m00, float m10, float m20, float m01, float m11, float m21, float m02, float m12, float m22)
        : this(new float3(m00, m10, m20), new float3(m01, m11, m21), new float3(m02, m12, m22))
    {
    }

    /// <summary>
    /// Gets the identity matrix.
    /// </summary>
    public static float3x3 Identity => new(new float3(1, 0, 0), new float3(0, 1, 0), new float3(0, 0, 1));

    /// <summary>
    /// Gets the zero matrix.
    /// </summary>
    public static float3x3 Zero => new(float3.Zero, float3.Zero, float3.Zero);

    /// <summary>
    /// Gets element row 0, column 0.
    /// </summary>
    public float M00 => C0.X;

    /// <summary>
    /// Gets element row 1, column 0.
    /// </summary>
    public float M10 => C0.Y;

    /// <summary>
    /// Gets element row 2, column 0.
    /// </summary>
    public float M20 => C0.Z;

    /// <summary>
    /// Gets element row 0, column 1.
    /// </summary>
    public float M01 => C1.X;

    /// <summary>
    /// Gets element row 1, column 1.
    /// </summary>
    public float M11 => C1.Y;

    /// <summary>
    /// Gets element row 2, column 1.
    /// </summary>
    public float M21 => C1.Z;

    /// <summary>
    /// Gets element row 0, column 2.
    /// </summary>
    public float M02 => C2.X;

    /// <summary>
    /// Gets element row 1, column 2.
    /// </summary>
    public float M12 => C2.Y;

    /// <summary>
    /// Gets element row 2, column 2.
    /// </summary>
    public float M22 => C2.Z;

    /// <summary>
    /// Returns the transposed matrix.
    /// </summary>
    public float3x3 Transposed()
        => new(
            new float3(M00, M01, M02),
            new float3(M10, M11, M12),
            new float3(M20, M21, M22));

    /// <summary>
    /// Returns the determinant of the matrix.
    /// </summary>
    public float Determinant()
    {
        var cof00 = (M11 * M22) - (M12 * M21);
        var cof01 = -((M10 * M22) - (M12 * M20));
        var cof02 = (M10 * M21) - (M11 * M20);
        return (M00 * cof00) + (M01 * cof01) + (M02 * cof02);
    }

    /// <summary>
    /// Returns the inverse matrix.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when the matrix is singular.</exception>
    public float3x3 Inverse()
    {
        var cof00 = (M11 * M22) - (M12 * M21);
        var cof01 = -((M10 * M22) - (M12 * M20));
        var cof02 = (M10 * M21) - (M11 * M20);
        var cof10 = -((M01 * M22) - (M02 * M21));
        var cof11 = (M00 * M22) - (M02 * M20);
        var cof12 = -((M00 * M21) - (M01 * M20));
        var cof20 = (M01 * M12) - (M02 * M11);
        var cof21 = -((M00 * M12) - (M02 * M10));
        var cof22 = (M00 * M11) - (M01 * M10);
        var determinant = (M00 * cof00) + (M01 * cof01) + (M02 * cof02);
        if (System.MathF.Abs(determinant) <= SingularEpsilon)
        {
            throw new InvalidOperationException("Cannot invert a singular float3x3 matrix.");
        }

        return new float3x3(
            new float3(cof00, cof01, cof02),
            new float3(cof10, cof11, cof12),
            new float3(cof20, cof21, cof22)) / determinant;
    }

    /// <summary>
    /// Adds two matrices component-wise.
    /// </summary>
    public static float3x3 operator +(float3x3 left, float3x3 right) => new(left.C0 + right.C0, left.C1 + right.C1, left.C2 + right.C2);

    /// <summary>
    /// Subtracts two matrices component-wise.
    /// </summary>
    public static float3x3 operator -(float3x3 left, float3x3 right) => new(left.C0 - right.C0, left.C1 - right.C1, left.C2 - right.C2);

    /// <summary>
    /// Negates every matrix element.
    /// </summary>
    public static float3x3 operator -(float3x3 value) => new(-value.C0, -value.C1, -value.C2);

    /// <summary>
    /// Multiplies a matrix by a scalar.
    /// </summary>
    public static float3x3 operator *(float3x3 left, float right) => new(left.C0 * right, left.C1 * right, left.C2 * right);

    /// <summary>
    /// Multiplies a matrix by a scalar.
    /// </summary>
    public static float3x3 operator *(float left, float3x3 right) => right * left;

    /// <summary>
    /// Multiplies a matrix by a vector.
    /// </summary>
    public static float3 operator *(float3x3 left, float3 right) => (left.C0 * right.X) + (left.C1 * right.Y) + (left.C2 * right.Z);

    /// <summary>
    /// Multiplies two matrices.
    /// </summary>
    public static float3x3 operator *(float3x3 left, float3x3 right) => new(left * right.C0, left * right.C1, left * right.C2);

    /// <summary>
    /// Divides a matrix by a scalar.
    /// </summary>
    public static float3x3 operator /(float3x3 left, float right) => new(left.C0 / right, left.C1 / right, left.C2 / right);

    /// <summary>
    /// Multiplies two matrices component-wise.
    /// </summary>
    public static float3x3 Hadamard(float3x3 left, float3x3 right) => new(left.C0 * right.C0, left.C1 * right.C1, left.C2 * right.C2);
}

/// <summary>
/// Represents a column-major 4x4 single-precision floating-point matrix.
/// </summary>
/// <param name="C0">The first column.</param>
/// <param name="C1">The second column.</param>
/// <param name="C2">The third column.</param>
/// <param name="C3">The fourth column.</param>
public readonly record struct float4x4(float4 C0, float4 C1, float4 C2, float4 C3)
{
    private const float SingularEpsilon = 1e-8f;
    private const int Size = 4;
    private const int AugmentedStride = 8;

    /// <summary>
    /// Creates a matrix from column-major components.
    /// </summary>
    public float4x4(
        float m00,
        float m10,
        float m20,
        float m30,
        float m01,
        float m11,
        float m21,
        float m31,
        float m02,
        float m12,
        float m22,
        float m32,
        float m03,
        float m13,
        float m23,
        float m33)
        : this(
            new float4(m00, m10, m20, m30),
            new float4(m01, m11, m21, m31),
            new float4(m02, m12, m22, m32),
            new float4(m03, m13, m23, m33))
    {
    }

    /// <summary>
    /// Gets the identity matrix.
    /// </summary>
    public static float4x4 Identity => new(new float4(1, 0, 0, 0), new float4(0, 1, 0, 0), new float4(0, 0, 1, 0), new float4(0, 0, 0, 1));

    /// <summary>
    /// Gets the zero matrix.
    /// </summary>
    public static float4x4 Zero => new(float4.Zero, float4.Zero, float4.Zero, float4.Zero);

    /// <summary>
    /// Gets element row 0, column 0.
    /// </summary>
    public float M00 => C0.X;

    /// <summary>
    /// Gets element row 1, column 0.
    /// </summary>
    public float M10 => C0.Y;

    /// <summary>
    /// Gets element row 2, column 0.
    /// </summary>
    public float M20 => C0.Z;

    /// <summary>
    /// Gets element row 3, column 0.
    /// </summary>
    public float M30 => C0.W;

    /// <summary>
    /// Gets element row 0, column 1.
    /// </summary>
    public float M01 => C1.X;

    /// <summary>
    /// Gets element row 1, column 1.
    /// </summary>
    public float M11 => C1.Y;

    /// <summary>
    /// Gets element row 2, column 1.
    /// </summary>
    public float M21 => C1.Z;

    /// <summary>
    /// Gets element row 3, column 1.
    /// </summary>
    public float M31 => C1.W;

    /// <summary>
    /// Gets element row 0, column 2.
    /// </summary>
    public float M02 => C2.X;

    /// <summary>
    /// Gets element row 1, column 2.
    /// </summary>
    public float M12 => C2.Y;

    /// <summary>
    /// Gets element row 2, column 2.
    /// </summary>
    public float M22 => C2.Z;

    /// <summary>
    /// Gets element row 3, column 2.
    /// </summary>
    public float M32 => C2.W;

    /// <summary>
    /// Gets element row 0, column 3.
    /// </summary>
    public float M03 => C3.X;

    /// <summary>
    /// Gets element row 1, column 3.
    /// </summary>
    public float M13 => C3.Y;

    /// <summary>
    /// Gets element row 2, column 3.
    /// </summary>
    public float M23 => C3.Z;

    /// <summary>
    /// Gets element row 3, column 3.
    /// </summary>
    public float M33 => C3.W;

    /// <summary>
    /// Returns the transposed matrix.
    /// </summary>
    public float4x4 Transposed()
        => new(
            new float4(M00, M01, M02, M03),
            new float4(M10, M11, M12, M13),
            new float4(M20, M21, M22, M23),
            new float4(M30, M31, M32, M33));

    /// <summary>
    /// Returns the determinant of the matrix.
    /// </summary>
    public float Determinant()
    {
        var minor0 = new float3x3(M11, M21, M31, M12, M22, M32, M13, M23, M33).Determinant();
        var minor1 = new float3x3(M10, M20, M30, M12, M22, M32, M13, M23, M33).Determinant();
        var minor2 = new float3x3(M10, M20, M30, M11, M21, M31, M13, M23, M33).Determinant();
        var minor3 = new float3x3(M10, M20, M30, M11, M21, M31, M12, M22, M32).Determinant();
        return (M00 * minor0) - (M01 * minor1) + (M02 * minor2) - (M03 * minor3);
    }

    /// <summary>
    /// Returns the inverse matrix.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when the matrix is singular.</exception>
    public float4x4 Inverse()
    {
        Span<float> augmented = stackalloc float[Size * AugmentedStride];

        for (var row = 0; row < Size; row++)
        {
            for (var column = 0; column < Size; column++)
            {
                augmented[Index(row, column)] = Get(row, column);
                augmented[Index(row, column + Size)] = row == column ? 1.0f : 0.0f;
            }
        }

        for (var pivotColumn = 0; pivotColumn < Size; pivotColumn++)
        {
            var pivotRow = pivotColumn;
            var pivotMagnitude = System.MathF.Abs(augmented[Index(pivotRow, pivotColumn)]);
            for (var row = pivotColumn + 1; row < Size; row++)
            {
                var candidateMagnitude = System.MathF.Abs(augmented[Index(row, pivotColumn)]);
                if (candidateMagnitude > pivotMagnitude)
                {
                    pivotMagnitude = candidateMagnitude;
                    pivotRow = row;
                }
            }

            if (pivotMagnitude <= SingularEpsilon)
            {
                throw new InvalidOperationException("Cannot invert a singular float4x4 matrix.");
            }

            if (pivotRow != pivotColumn)
            {
                SwapRows(augmented, pivotRow, pivotColumn);
            }

            var pivot = augmented[Index(pivotColumn, pivotColumn)];
            for (var column = 0; column < AugmentedStride; column++)
            {
                augmented[Index(pivotColumn, column)] /= pivot;
            }

            for (var row = 0; row < Size; row++)
            {
                if (row == pivotColumn)
                {
                    continue;
                }

                var factor = augmented[Index(row, pivotColumn)];
                if (factor == 0)
                {
                    continue;
                }

                for (var column = 0; column < AugmentedStride; column++)
                {
                    augmented[Index(row, column)] -= factor * augmented[Index(pivotColumn, column)];
                }
            }
        }

        return new float4x4(
            new float4(augmented[Index(0, 4)], augmented[Index(1, 4)], augmented[Index(2, 4)], augmented[Index(3, 4)]),
            new float4(augmented[Index(0, 5)], augmented[Index(1, 5)], augmented[Index(2, 5)], augmented[Index(3, 5)]),
            new float4(augmented[Index(0, 6)], augmented[Index(1, 6)], augmented[Index(2, 6)], augmented[Index(3, 6)]),
            new float4(augmented[Index(0, 7)], augmented[Index(1, 7)], augmented[Index(2, 7)], augmented[Index(3, 7)]));
    }

    /// <summary>
    /// Adds two matrices component-wise.
    /// </summary>
    public static float4x4 operator +(float4x4 left, float4x4 right) => new(left.C0 + right.C0, left.C1 + right.C1, left.C2 + right.C2, left.C3 + right.C3);

    /// <summary>
    /// Subtracts two matrices component-wise.
    /// </summary>
    public static float4x4 operator -(float4x4 left, float4x4 right) => new(left.C0 - right.C0, left.C1 - right.C1, left.C2 - right.C2, left.C3 - right.C3);

    /// <summary>
    /// Negates every matrix element.
    /// </summary>
    public static float4x4 operator -(float4x4 value) => new(-value.C0, -value.C1, -value.C2, -value.C3);

    /// <summary>
    /// Multiplies a matrix by a scalar.
    /// </summary>
    public static float4x4 operator *(float4x4 left, float right) => new(left.C0 * right, left.C1 * right, left.C2 * right, left.C3 * right);

    /// <summary>
    /// Multiplies a matrix by a scalar.
    /// </summary>
    public static float4x4 operator *(float left, float4x4 right) => right * left;

    /// <summary>
    /// Multiplies a matrix by a vector.
    /// </summary>
    public static float4 operator *(float4x4 left, float4 right) => (left.C0 * right.X) + (left.C1 * right.Y) + (left.C2 * right.Z) + (left.C3 * right.W);

    /// <summary>
    /// Multiplies two matrices.
    /// </summary>
    public static float4x4 operator *(float4x4 left, float4x4 right) => new(left * right.C0, left * right.C1, left * right.C2, left * right.C3);

    /// <summary>
    /// Divides a matrix by a scalar.
    /// </summary>
    public static float4x4 operator /(float4x4 left, float right) => new(left.C0 / right, left.C1 / right, left.C2 / right, left.C3 / right);

    /// <summary>
    /// Multiplies two matrices component-wise.
    /// </summary>
    public static float4x4 Hadamard(float4x4 left, float4x4 right) => new(left.C0 * right.C0, left.C1 * right.C1, left.C2 * right.C2, left.C3 * right.C3);

    private static int Index(int row, int column) => (row * AugmentedStride) + column;

    private static void SwapRows(Span<float> augmented, int first, int second)
    {
        for (var column = 0; column < AugmentedStride; column++)
        {
            (augmented[Index(first, column)], augmented[Index(second, column)]) = (augmented[Index(second, column)], augmented[Index(first, column)]);
        }
    }

    private float Get(int row, int column)
    {
        var vector = column switch
        {
            0 => C0,
            1 => C1,
            2 => C2,
            3 => C3,
            _ => throw new ArgumentOutOfRangeException(nameof(column))
        };

        return row switch
        {
            0 => vector.X,
            1 => vector.Y,
            2 => vector.Z,
            3 => vector.W,
            _ => throw new ArgumentOutOfRangeException(nameof(row))
        };
    }
}
