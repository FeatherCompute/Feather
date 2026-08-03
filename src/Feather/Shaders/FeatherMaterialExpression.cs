// Injected as source so generated raster projects can lower these callables with their shaders.
using Feather.Math;
using Feather.Resources;

namespace Feather.Shaders;

[GpuStruct]
public partial struct RasterMaterialInstruction
{
    public float4 Value;
    public float4 Parameters;
    public int Op;
    public int A; public int B; public int C; public int D;
    public int E; public int F; public int G; public int H;
    public int ParameterOffset;
    public int ParameterCount;
    public int Reserved;
}

[GpuStruct]
public partial struct RasterMaterialRegisters
{
    public float4 R0; public float4 R1; public float4 R2; public float4 R3;
    public float4 R4; public float4 R5; public float4 R6; public float4 R7;
    public float4 R8; public float4 R9; public float4 R10; public float4 R11;
    public float4 R12; public float4 R13; public float4 R14; public float4 R15;
    public float4 R16; public float4 R17; public float4 R18; public float4 R19;
    public float4 R20; public float4 R21; public float4 R22; public float4 R23;
    public float4 R24; public float4 R25; public float4 R26; public float4 R27;
    public float4 R28; public float4 R29; public float4 R30; public float4 R31;
    public float4 R32; public float4 R33; public float4 R34; public float4 R35;
    public float4 R36; public float4 R37; public float4 R38; public float4 R39;
    public float4 R40; public float4 R41; public float4 R42; public float4 R43;
    public float4 R44; public float4 R45; public float4 R46; public float4 R47;
    public float4 R48; public float4 R49; public float4 R50; public float4 R51;
    public float4 R52; public float4 R53; public float4 R54; public float4 R55;
    public float4 R56; public float4 R57; public float4 R58; public float4 R59;
    public float4 R60; public float4 R61; public float4 R62; public float4 R63;
    public float4 R64; public float4 R65; public float4 R66; public float4 R67;
    public float4 R68; public float4 R69; public float4 R70; public float4 R71;
    public float4 R72; public float4 R73; public float4 R74; public float4 R75;
    public float4 R76; public float4 R77; public float4 R78; public float4 R79;
    public float4 R80; public float4 R81; public float4 R82; public float4 R83;
    public float4 R84; public float4 R85; public float4 R86; public float4 R87;
    public float4 R88; public float4 R89; public float4 R90; public float4 R91;
    public float4 R92; public float4 R93; public float4 R94; public float4 R95;
    public float4 R96; public float4 R97; public float4 R98; public float4 R99;
    public float4 R100; public float4 R101; public float4 R102; public float4 R103;
    public float4 R104; public float4 R105; public float4 R106; public float4 R107;
    public float4 R108; public float4 R109; public float4 R110; public float4 R111;
    public float4 R112; public float4 R113; public float4 R114; public float4 R115;
    public float4 R116; public float4 R117; public float4 R118; public float4 R119;
    public float4 R120; public float4 R121; public float4 R122; public float4 R123;
    public float4 R124; public float4 R125; public float4 R126; public float4 R127;
}

/// <summary>
/// Buffer-backed parameters for the fixed canonical material topologies. The layout is intentionally
/// independent of any one renderer so raster previews and path kernels use the same evaluator.
/// </summary>
[GpuStruct]
public partial struct RasterCompiledMaterialProgram
{
    public float4 Parameter0;
    public float4 Parameter1;
    public float4 Parameter2;
    public float4 Parameter3;
    public int Variant;
    public int Texture0;
    public int Texture1;
    public int Texture2;
    public int Texture3;
    public int Texture4;
    public int Channel0;
    public int Channel1;
    public int Channel2;
    public int Channel3;
    public int Channel4;
    public int Target;
    public int Pad0;
    public int Pad1;
    public int Pad2;
    public int Pad3;
}

/// <summary>Only fields marked present replace the flattened material defaults.</summary>
[GpuStruct]
public partial struct RasterCompiledMaterialResult
{
    public float3 BaseColor;
    public int HasBaseColor;
    public float3 TangentNormal;
    public int HasNormal;
    public float Metallic;
    public float Roughness;
    public float Alpha;
    public int HasMetallic;
    public int HasRoughness;
    public int HasAlpha;
    public int Pad0;
    public int Pad1;
}

[ShaderLibrary]
public static class FeatherMaterialExpression
{
    /// <summary>
    /// Executes one of the bounded canonical topology functions. Samples are resolved by the caller
    /// because shader resources cannot cross a callable boundary; all graph/register addressing was
    /// resolved by the host compiler.
    /// </summary>
    [Callable]
    public static RasterCompiledMaterialResult EvaluateCompiled(
        RasterCompiledMaterialProgram program,
        float4 sample0,
        float4 sample1,
        float4 sample2,
        float4 sample3,
        float4 sample4)
    {
        RasterCompiledMaterialResult result = default;
        if (program.Variant == 1)
        {
            return result;
        }

        if (program.Variant == 2)
        {
            if (program.Texture0 >= 0)
            {
                result.BaseColor = sample0.XYZ;
                result.HasBaseColor = 1;
            }
            if (program.Texture1 >= 0)
            {
                result.Metallic = Channel(sample1, program.Channel1);
                result.HasMetallic = 1;
            }
            if (program.Texture2 >= 0)
            {
                result.Roughness = Channel(sample2, program.Channel2);
                result.HasRoughness = 1;
            }
            if (program.Texture3 >= 0)
            {
                result.Alpha = Channel(sample3, program.Channel3);
                result.HasAlpha = 1;
            }
            if (program.Texture4 >= 0)
            {
                result.TangentNormal = NormalMap(sample4, program.Parameter0.X);
                result.HasNormal = 1;
            }
            return result;
        }

        if (program.Variant >= 3 && program.Variant <= 6)
        {
            if (program.Texture0 >= 0)
            {
                result.BaseColor = sample0.XYZ;
                result.HasBaseColor = 1;
            }
            var value = Channel(sample1, program.Channel1);
            if (program.Variant == 3) value *= program.Parameter0.X;
            else if (program.Variant == 4) value += program.Parameter0.X;
            else if (program.Variant == 5) value = program.Parameter0.X - value;
            else value = (value * program.Parameter0.X) + program.Parameter0.Y;
            if (program.Variant != 6 && program.Parameter0.Y > 0.5f)
            {
                value = ShaderMath.Saturate(value);
            }
            if (program.Target == 1)
            {
                result.Metallic = value;
                result.HasMetallic = 1;
            }
            else if (program.Target == 2)
            {
                result.Roughness = value;
                result.HasRoughness = 1;
            }
            else
            {
                result.Alpha = value;
                result.HasAlpha = 1;
            }
            if (program.Texture4 >= 0)
            {
                result.TangentNormal = NormalMap(sample4, program.Parameter1.X);
                result.HasNormal = 1;
            }
            return result;
        }

        if (program.Variant == 7)
        {
            var a = program.Texture0 >= 0 ? sample0 : program.Parameter0;
            var b = program.Texture1 >= 0 ? sample1 : program.Parameter1;
            var factor = program.Texture2 >= 0
                ? Channel(sample2, program.Channel2)
                : program.Parameter2.X;
            if (program.Parameter2.Y > 0.5f)
            {
                factor = ShaderMath.Saturate(factor);
            }
            result.BaseColor = ShaderMath.Lerp(a, b, factor).XYZ;
            result.HasBaseColor = 1;
            return result;
        }

        // The three two-stop ramp variants differ by one fixed interpolation expression. There is no
        // parameter-table loop and no dynamic register access.
        if (program.Variant >= 8 && program.Variant <= 10)
        {
            var factor = program.Texture0 >= 0
                ? Channel(sample0, program.Channel0)
                : program.Parameter0.X;
            var p0 = program.Parameter3.X;
            var p1 = program.Parameter3.Y;
            var color = program.Parameter1;
            if (factor >= p1)
            {
                color = program.Parameter2;
            }
            else if (factor >= p0 && program.Variant != 8)
            {
                var blend = ShaderMath.Saturate((factor - p0) / ShaderMath.Max(p1 - p0, 1e-6f));
                if (program.Variant == 10)
                {
                    blend = blend * blend * (3.0f - (2.0f * blend));
                }
                color = ShaderMath.Lerp(program.Parameter1, program.Parameter2, blend);
            }
            result.BaseColor = color.XYZ;
            result.HasBaseColor = 1;
        }
        return result;
    }

    [Callable]
    private static float Channel(float4 value, int channel)
    {
        if (channel == 1) return value.Y;
        if (channel == 2) return value.Z;
        if (channel == 3) return value.W;
        return value.X;
    }

    [Callable]
    private static float3 NormalMap(float4 color, float strength)
    {
        var mapped = (color.XYZ * 2.0f) - new float3(1.0f);
        return ShaderMath.Normalize(ShaderMath.Lerp(
            new float3(0.0f, 0.0f, 1.0f),
            mapped,
            ShaderMath.Max(strength, 0.0f)));
    }

    [Callable]
    public static float4 Evaluate(
        RasterMaterialInstruction instruction,
        RasterMaterialRegisters registers,
        float2 uv,
        float3 geometricNormal,
        float3 view)
    {
        var a = Get(registers, instruction.A);
        var b = Get(registers, instruction.B);
        var c = Get(registers, instruction.C);
        var d = Get(registers, instruction.D);
        var e = Get(registers, instruction.E);
        var f = Get(registers, instruction.F);
        var result = instruction.Value;
        if (instruction.Op == 1)
            result = new float4(uv + instruction.Parameters.ZW, 0.0f, 1.0f);
        else if (instruction.Op == 4)
        {
            var p = a.XYZ * b.X;
            var total = 0.0f; var weight = 0.0f; var amplitude = 1.0f; var frequency = 1.0f;
            for (var octave = 0; octave < 8; octave++)
            {
                if ((float)octave <= ShaderMath.Min(c.X, 8.0f))
                {
                    total += Noise(p * frequency) * amplitude;
                    weight += amplitude;
                    amplitude *= ShaderMath.Saturate(d.X);
                    frequency *= ShaderMath.Max(e.X, 1.0f);
                }
            }
            var n = ShaderMath.Saturate((total / ShaderMath.Max(weight, 1e-5f)) +
                (f.X * (Noise(p + new float3(9.2f, 3.7f, 5.1f)) - 0.5f)));
            result = instruction.Parameters.X > 0.5f
                ? new float4(n, Noise(p + new float3(17.0f, 3.0f, 1.0f)), Noise(p + new float3(2.0f, 11.0f, 7.0f)), 1.0f)
                : new float4(n, n, n, 1.0f);
        }
        else if (instruction.Op == 5)
        {
            var p = a.XYZ * b.X; var cell = ShaderMath.Floor(p);
            var nearest = 1000.0f; var second = 1000.0f; var selected = float3.Zero;
            for (var z = -1; z <= 1; z++)
            for (var y = -1; y <= 1; y++) for (var x = -1; x <= 1; x++)
            {
                var candidate = cell + new float3((float)x, (float)y, (float)z);
                var jitter = new float3(
                    Hash(candidate), Hash(candidate + new float3(7.0f)),
                    Hash(candidate + new float3(17.0f))) * c.X;
                var difference = ShaderMath.Abs((candidate + jitter) - p);
                var distance = ShaderMath.Length(difference);
                if (instruction.Parameters.Z == 1.0f)
                    distance = difference.X + difference.Y + difference.Z;
                else if (instruction.Parameters.Z == 2.0f)
                    distance = ShaderMath.Max(difference.X, ShaderMath.Max(difference.Y, difference.Z));
                else if (instruction.Parameters.Z == 3.0f)
                {
                    var exponent = ShaderMath.Max(d.X, 1e-3f);
                    distance = ShaderMath.Pow(
                        ShaderMath.Pow(difference.X, exponent) +
                        ShaderMath.Pow(difference.Y, exponent) +
                        ShaderMath.Pow(difference.Z, exponent),
                        1.0f / exponent);
                }
                if (distance < nearest)
                {
                    second = nearest; nearest = distance; selected = candidate;
                }
                else if (distance < second) second = distance;
            }
            result = instruction.Parameters.X > 0.5f
                ? new float4(Hash(selected), Hash(selected + new float3(13.0f)), Hash(selected + new float3(29.0f)), 1.0f)
                : new float4(instruction.Parameters.Y > 0.5f ? second : nearest);
        }
        else if (instruction.Op == 6)
        {
            var value = a.X;
            if (instruction.Parameters.X == 1.0f) value *= value;
            else if (instruction.Parameters.X == 2.0f) value = ShaderMath.Smoothstep(0.0f, 1.0f, value);
            else if (instruction.Parameters.X == 3.0f) value = (a.X + a.Y) * 0.5f;
            else if (instruction.Parameters.X >= 4.0f) { value = ShaderMath.Length(a.XYZ); if (instruction.Parameters.X == 5.0f) value *= value; }
            if (instruction.Parameters.X == 6.0f)
            {
                value = ShaderMath.Fract((Atan2Approx(a.Y, a.X) / 6.28318530718f) + 1.0f);
            }
            value = ShaderMath.Saturate(value); result = new float4(value);
        }
        else if (instruction.Op == 7)
        {
            var tile = ShaderMath.Floor(a.X * d.X) + ShaderMath.Floor(a.Y * d.X) + ShaderMath.Floor(a.Z * d.X);
            var factor = ShaderMath.Fract(tile * 0.5f) < 0.25f ? 0.0f : 1.0f;
            result = instruction.Parameters.X > 0.5f ? new float4(factor) : ShaderMath.Lerp(b, c, factor);
        }
        else if (instruction.Op == 8 || instruction.Op == 22)
        {
            var factor = instruction.Op == 22 || instruction.Parameters.X > 0.5f ? ShaderMath.Saturate(a.X) : a.X;
            var blended = instruction.Op == 22 ? c : Blend(instruction.Parameters.Z, b, c);
            result = ShaderMath.Lerp(b, blended, factor);
            if (instruction.Op == 8 && instruction.Parameters.Y > 0.5f) result = ShaderMath.Saturate(result);
        }
        else if (instruction.Op == 11)
        {
            var value = Math(instruction.Parameters.X, a.X, b.X, c.X);
            if (instruction.Parameters.Y > 0.5f) value = ShaderMath.Saturate(value);
            result = new float4(value);
        }
        else if (instruction.Op == 12) result = VectorMath(instruction.Parameters.X, a, b, d);
        else if (instruction.Op == 13)
        {
            var t = (a.X - b.X) / ShaderMath.Max(c.X - b.X, 1e-6f);
            if (instruction.Parameters.X > 0.5f) t = ShaderMath.Saturate(t);
            if (instruction.Parameters.Y == 1.0f)
            {
                t = ShaderMath.Saturate(t);
                t = t * t * (3.0f - (2.0f * t));
            }
            else if (instruction.Parameters.Y == 2.0f)
            {
                t = ShaderMath.Saturate(t);
                t = t * t * t * (t * ((t * 6.0f) - 15.0f) + 10.0f);
            }
            result = new float4(ShaderMath.Lerp(d.X, e.X, t));
        }
        else if (instruction.Op == 14)
        {
            var blend = Blend(instruction.Parameters.X, b, c);
            result = ShaderMath.Lerp(b, blend, ShaderMath.Saturate(a.X));
            if (instruction.Parameters.Y > 0.5f) result = ShaderMath.Saturate(result);
        }
        else if (instruction.Op == 15)
        {
            var hsv = RgbToHsv(a.XYZ);
            hsv = new float3(ShaderMath.Fract(hsv.X + c.X - 0.5f), ShaderMath.Max(hsv.Y * d.X, 0.0f), hsv.Z * e.X);
            result = ShaderMath.Lerp(a, new float4(HsvToRgb(hsv), a.W), ShaderMath.Saturate(b.X));
        }
        else if (instruction.Op == 16)
        {
            var mapped = a.XYZ * d.XYZ;
            if (instruction.Parameters.X <= 1.0f) mapped += instruction.Parameters.X == 0.0f ? b.XYZ : -b.XYZ;
            result = new float4(Rotate(mapped, c.XYZ), 1.0f);
        }
        else if (instruction.Op == 17)
        {
            var mapped = (a.XYZ * 2.0f) - new float3(1.0f);
            result = new float4(ShaderMath.Normalize(ShaderMath.Lerp(new float3(0.0f, 0.0f, 1.0f), mapped, ShaderMath.Max(b.X, 0.0f))), 0.0f);
        }
        else if (instruction.Op == 18)
        {
            var value = instruction.Parameters.X == 0.0f ? a.X : (instruction.Parameters.X == 1.0f ? a.Y : a.Z);
            result = new float4(value);
        }
        else if (instruction.Op == 19) result = new float4(a.X, b.X, c.X, 1.0f);
        else if (instruction.Op == 20 || instruction.Op == 21)
        {
            var n = ShaderMath.Dot(b.XYZ, b.XYZ) > 1e-6f ? ShaderMath.Normalize(b.XYZ) : geometricNormal;
            var facing = ShaderMath.Saturate(ShaderMath.Abs(ShaderMath.Dot(n, view)));
            var value = facing;
            if (instruction.Op == 20)
            {
                var ratio = (a.X - 1.0f) / ShaderMath.Max(a.X + 1.0f, 1e-6f); var f0 = ratio * ratio;
                value = f0 + ((1.0f - f0) * ShaderMath.Pow(1.0f - facing, 5.0f));
            }
            else if (instruction.Parameters.X < 0.5f) value = ShaderMath.Pow(1.0f - facing, ShaderMath.Max(a.X, 1e-3f));
            result = new float4(value);
        }
        else if (instruction.Op == 23) result = (a + b) * 0.5f;
        else if (instruction.Op == 24 || instruction.Op == 25)
        {
            var baseNormal = ShaderMath.Dot(d.XYZ, d.XYZ) > 1e-6f
                ? ShaderMath.Normalize(d.XYZ)
                : new float3(0.0f, 0.0f, 1.0f);
            if (instruction.Parameters.Y > 0.5f)
            {
                var delta = 1.0f / 256.0f;
                var bumpScale = (instruction.Parameters.X > 0.5f ? -1.0f : 1.0f)
                    * f.Y * f.Z;
                var derivativeX = (e.Y - e.X) / (2.0f * delta);
                var derivativeY = (f.X - e.Z) / (2.0f * delta);
                baseNormal = ShaderMath.Normalize(new float3(
                    baseNormal.X - (derivativeX * bumpScale),
                    baseNormal.Y - (derivativeY * bumpScale),
                    baseNormal.Z));
            }
            result = new float4(baseNormal, 0.0f);
        }
        return result;
    }

    [Callable] public static RasterMaterialRegisters Set(RasterMaterialRegisters r, int i, float4 v)
    {
        if(i==0)r.R0=v;else if(i==1)r.R1=v;else if(i==2)r.R2=v;else if(i==3)r.R3=v;
        else if(i==4)r.R4=v;else if(i==5)r.R5=v;else if(i==6)r.R6=v;else if(i==7)r.R7=v;
        else if(i==8)r.R8=v;else if(i==9)r.R9=v;else if(i==10)r.R10=v;else if(i==11)r.R11=v;
        else if(i==12)r.R12=v;else if(i==13)r.R13=v;else if(i==14)r.R14=v;else if(i==15)r.R15=v;
        else if(i==16)r.R16=v;else if(i==17)r.R17=v;else if(i==18)r.R18=v;else if(i==19)r.R19=v;
        else if(i==20)r.R20=v;else if(i==21)r.R21=v;else if(i==22)r.R22=v;else if(i==23)r.R23=v;
        else if(i==24)r.R24=v;else if(i==25)r.R25=v;else if(i==26)r.R26=v;else if(i==27)r.R27=v;
        else if(i==28)r.R28=v;else if(i==29)r.R29=v;else if(i==30)r.R30=v;else if(i==31)r.R31=v;
        else if(i==32)r.R32=v;else if(i==33)r.R33=v;else if(i==34)r.R34=v;else if(i==35)r.R35=v;
        else if(i==36)r.R36=v;else if(i==37)r.R37=v;else if(i==38)r.R38=v;else if(i==39)r.R39=v;
        else if(i==40)r.R40=v;else if(i==41)r.R41=v;else if(i==42)r.R42=v;else if(i==43)r.R43=v;
        else if(i==44)r.R44=v;else if(i==45)r.R45=v;else if(i==46)r.R46=v;else if(i==47)r.R47=v;
        else if(i==48)r.R48=v;else if(i==49)r.R49=v;else if(i==50)r.R50=v;else if(i==51)r.R51=v;
        else if(i==52)r.R52=v;else if(i==53)r.R53=v;else if(i==54)r.R54=v;else if(i==55)r.R55=v;
        else if(i==56)r.R56=v;else if(i==57)r.R57=v;else if(i==58)r.R58=v;else if(i==59)r.R59=v;
        else if(i==60)r.R60=v;else if(i==61)r.R61=v;else if(i==62)r.R62=v;else if(i==63)r.R63=v;
        else if(i==64)r.R64=v;else if(i==65)r.R65=v;else if(i==66)r.R66=v;else if(i==67)r.R67=v;
        else if(i==68)r.R68=v;else if(i==69)r.R69=v;else if(i==70)r.R70=v;else if(i==71)r.R71=v;
        else if(i==72)r.R72=v;else if(i==73)r.R73=v;else if(i==74)r.R74=v;else if(i==75)r.R75=v;
        else if(i==76)r.R76=v;else if(i==77)r.R77=v;else if(i==78)r.R78=v;else if(i==79)r.R79=v;
        else if(i==80)r.R80=v;else if(i==81)r.R81=v;else if(i==82)r.R82=v;else if(i==83)r.R83=v;
        else if(i==84)r.R84=v;else if(i==85)r.R85=v;else if(i==86)r.R86=v;else if(i==87)r.R87=v;
        else if(i==88)r.R88=v;else if(i==89)r.R89=v;else if(i==90)r.R90=v;else if(i==91)r.R91=v;
        else if(i==92)r.R92=v;else if(i==93)r.R93=v;else if(i==94)r.R94=v;else if(i==95)r.R95=v;
        else if(i==96)r.R96=v;else if(i==97)r.R97=v;else if(i==98)r.R98=v;else if(i==99)r.R99=v;
        else if(i==100)r.R100=v;else if(i==101)r.R101=v;else if(i==102)r.R102=v;else if(i==103)r.R103=v;
        else if(i==104)r.R104=v;else if(i==105)r.R105=v;else if(i==106)r.R106=v;else if(i==107)r.R107=v;
        else if(i==108)r.R108=v;else if(i==109)r.R109=v;else if(i==110)r.R110=v;else if(i==111)r.R111=v;
        else if(i==112)r.R112=v;else if(i==113)r.R113=v;else if(i==114)r.R114=v;else if(i==115)r.R115=v;
        else if(i==116)r.R116=v;else if(i==117)r.R117=v;else if(i==118)r.R118=v;else if(i==119)r.R119=v;
        else if(i==120)r.R120=v;else if(i==121)r.R121=v;else if(i==122)r.R122=v;else if(i==123)r.R123=v;
        else if(i==124)r.R124=v;else if(i==125)r.R125=v;else if(i==126)r.R126=v;else if(i==127)r.R127=v;
        return r;
    }

    [Callable] public static float4 Get(RasterMaterialRegisters r, int i)
    {
        if(i==0)return r.R0;if(i==1)return r.R1;if(i==2)return r.R2;if(i==3)return r.R3;
        if(i==4)return r.R4;if(i==5)return r.R5;if(i==6)return r.R6;if(i==7)return r.R7;
        if(i==8)return r.R8;if(i==9)return r.R9;if(i==10)return r.R10;if(i==11)return r.R11;
        if(i==12)return r.R12;if(i==13)return r.R13;if(i==14)return r.R14;if(i==15)return r.R15;
        if(i==16)return r.R16;if(i==17)return r.R17;if(i==18)return r.R18;if(i==19)return r.R19;
        if(i==20)return r.R20;if(i==21)return r.R21;if(i==22)return r.R22;if(i==23)return r.R23;
        if(i==24)return r.R24;if(i==25)return r.R25;if(i==26)return r.R26;if(i==27)return r.R27;
        if(i==28)return r.R28;if(i==29)return r.R29;if(i==30)return r.R30;if(i==31)return r.R31;
        if(i==32)return r.R32;if(i==33)return r.R33;if(i==34)return r.R34;if(i==35)return r.R35;
        if(i==36)return r.R36;if(i==37)return r.R37;if(i==38)return r.R38;if(i==39)return r.R39;
        if(i==40)return r.R40;if(i==41)return r.R41;if(i==42)return r.R42;if(i==43)return r.R43;
        if(i==44)return r.R44;if(i==45)return r.R45;if(i==46)return r.R46;if(i==47)return r.R47;
        if(i==48)return r.R48;if(i==49)return r.R49;if(i==50)return r.R50;if(i==51)return r.R51;
        if(i==52)return r.R52;if(i==53)return r.R53;if(i==54)return r.R54;if(i==55)return r.R55;
        if(i==56)return r.R56;if(i==57)return r.R57;if(i==58)return r.R58;if(i==59)return r.R59;
        if(i==60)return r.R60;if(i==61)return r.R61;if(i==62)return r.R62;if(i==63)return r.R63;
        if(i==64)return r.R64;if(i==65)return r.R65;if(i==66)return r.R66;if(i==67)return r.R67;
        if(i==68)return r.R68;if(i==69)return r.R69;if(i==70)return r.R70;if(i==71)return r.R71;
        if(i==72)return r.R72;if(i==73)return r.R73;if(i==74)return r.R74;if(i==75)return r.R75;
        if(i==76)return r.R76;if(i==77)return r.R77;if(i==78)return r.R78;if(i==79)return r.R79;
        if(i==80)return r.R80;if(i==81)return r.R81;if(i==82)return r.R82;if(i==83)return r.R83;
        if(i==84)return r.R84;if(i==85)return r.R85;if(i==86)return r.R86;if(i==87)return r.R87;
        if(i==88)return r.R88;if(i==89)return r.R89;if(i==90)return r.R90;if(i==91)return r.R91;
        if(i==92)return r.R92;if(i==93)return r.R93;if(i==94)return r.R94;if(i==95)return r.R95;
        if(i==96)return r.R96;if(i==97)return r.R97;if(i==98)return r.R98;if(i==99)return r.R99;
        if(i==100)return r.R100;if(i==101)return r.R101;if(i==102)return r.R102;if(i==103)return r.R103;
        if(i==104)return r.R104;if(i==105)return r.R105;if(i==106)return r.R106;if(i==107)return r.R107;
        if(i==108)return r.R108;if(i==109)return r.R109;if(i==110)return r.R110;if(i==111)return r.R111;
        if(i==112)return r.R112;if(i==113)return r.R113;if(i==114)return r.R114;if(i==115)return r.R115;
        if(i==116)return r.R116;if(i==117)return r.R117;if(i==118)return r.R118;if(i==119)return r.R119;
        if(i==120)return r.R120;if(i==121)return r.R121;if(i==122)return r.R122;if(i==123)return r.R123;
        if(i==124)return r.R124;if(i==125)return r.R125;if(i==126)return r.R126;if(i==127)return r.R127;
        return float4.Zero;
    }

    [Callable] private static float Hash(float3 p) => ShaderMath.Fract(ShaderMath.Sin(ShaderMath.Dot(p,new float3(127.1f,311.7f,74.7f)))*43758.5453f);
    [Callable] private static float Noise(float3 p)
    {
        var cell=ShaderMath.Floor(p);var f=ShaderMath.Fract(p);f=f*f*(new float3(3.0f)-(f*2.0f));var value=0.0f;
        for(var i=0;i<8;i++){var x=(float)(i%2);var y=(float)((i/2)%2);var z=(float)(i/4);var o=new float3(x,y,z);value+=Hash(cell+o)*(1.0f-ShaderMath.Abs(f.X-x))*(1.0f-ShaderMath.Abs(f.Y-y))*(1.0f-ShaderMath.Abs(f.Z-z));}
        return value;
    }
    [Callable] private static float Math(float op,float a,float b,float c)
    {if(op==0)return a+b;if(op==1)return a-b;if(op==2)return a*b;if(op==3)return ShaderMath.Abs(b)<1e-8f?0:a/b;if(op==4)return(a*b)+c;if(op==5)return ShaderMath.Pow(ShaderMath.Abs(a),b);if(op==6)return ShaderMath.Min(a,b);if(op==7)return ShaderMath.Max(a,b);if(op==8)return a<b?1:0;if(op==9)return a>b?1:0;if(op==10)return ShaderMath.Abs(a);if(op==11)return ShaderMath.Sqrt(ShaderMath.Max(a,0));if(op==12)return ShaderMath.Floor(a);if(op==13)return ShaderMath.Ceil(a);if(op==14)return ShaderMath.Fract(a);if(op==16)return ShaderMath.Sin(a);if(op==17)return ShaderMath.Cos(a);if(op==18)return ShaderMath.Tan(a);if(op==24)return AtanApprox(a);return a;}
    [Callable] private static float4 Blend(float op,float4 a,float4 b)
    {
        if(op==1)return a+b;if(op==2)return a*b;if(op==3)return a-b;
        if(op==4)return new float4(1.0f)-((new float4(1.0f)-a)*(new float4(1.0f)-b));
        if(op==5)return new float4(ShaderMath.Abs(b.X)<1e-8f?0:a.X/b.X,ShaderMath.Abs(b.Y)<1e-8f?0:a.Y/b.Y,ShaderMath.Abs(b.Z)<1e-8f?0:a.Z/b.Z,a.W);
        if(op==6)return ShaderMath.Abs(a-b);if(op==7)return ShaderMath.Min(a,b);if(op==8)return ShaderMath.Max(a,b);
        if(op==9)return new float4(Overlay(a.X,b.X),Overlay(a.Y,b.Y),Overlay(a.Z,b.Z),a.W);
        return b;
    }
    [Callable] private static float Overlay(float a,float b)=>a<0.5f?2.0f*a*b:1.0f-(2.0f*(1.0f-a)*(1.0f-b));
    [Callable] private static float AtanApprox(float value)
    {
        var magnitude=ShaderMath.Abs(value);
        if(magnitude<=1.0f)return value/(1.0f+(0.280872f*value*value));
        var result=1.57079632679f-(magnitude/((magnitude*magnitude)+0.280872f));
        return value<0.0f?-result:result;
    }
    [Callable] private static float Atan2Approx(float y,float x)
    {
        if(ShaderMath.Abs(x)<1e-8f)return y<0.0f?-1.57079632679f:1.57079632679f;
        var angle=AtanApprox(y/x);
        if(x<0.0f)angle+=y<0.0f?-3.14159265359f:3.14159265359f;
        return angle;
    }
    [Callable] private static float4 VectorMath(float op,float4 a,float4 b,float4 scale)
    {var av=a.XYZ;var bv=b.XYZ;if(op==0)return new float4(av+bv,1);if(op==1)return new float4(av-bv,1);if(op==2)return new float4(av*bv,1);if(op==4)return new float4(ShaderMath.Cross(av,bv),1);if(op==5)return new float4(ShaderMath.Dot(av,bv));if(op==6)return new float4(ShaderMath.Length(av-bv));if(op==7)return new float4(ShaderMath.Length(av));if(op==8)return new float4(av*scale.X,1);if(op==9)return new float4(ShaderMath.Normalize(av),1);if(op==10)return new float4(ShaderMath.Abs(av),1);return a;}
    [Callable] private static float3 RgbToHsv(float3 c){var max=ShaderMath.Max(c.X,ShaderMath.Max(c.Y,c.Z));var min=ShaderMath.Min(c.X,ShaderMath.Min(c.Y,c.Z));var delta=max-min;var h=0.0f;if(delta>1e-6f){if(max==c.X)h=(c.Y-c.Z)/delta;else if(max==c.Y)h=2+((c.Z-c.X)/delta);else h=4+((c.X-c.Y)/delta);h=ShaderMath.Fract(h/6);}return new float3(h,max<=1e-6f?0:delta/max,max);}
    [Callable] private static float3 HsvToRgb(float3 h){var x=ShaderMath.Fract(h.X)*6;var s=ShaderMath.Floor(x);var f=x-s;var p=h.Z*(1-h.Y);var q=h.Z*(1-(h.Y*f));var t=h.Z*(1-(h.Y*(1-f)));if(s<1)return new float3(h.Z,t,p);if(s<2)return new float3(q,h.Z,p);if(s<3)return new float3(p,h.Z,t);if(s<4)return new float3(p,q,h.Z);if(s<5)return new float3(t,p,h.Z);return new float3(h.Z,p,q);}
    [Callable] private static float3 Rotate(float3 v,float3 r){var cx=ShaderMath.Cos(r.X);var sx=ShaderMath.Sin(r.X);var cy=ShaderMath.Cos(r.Y);var sy=ShaderMath.Sin(r.Y);var cz=ShaderMath.Cos(r.Z);var sz=ShaderMath.Sin(r.Z);var x=new float3(v.X,(v.Y*cx)-(v.Z*sx),(v.Y*sx)+(v.Z*cx));var y=new float3((x.X*cy)+(x.Z*sy),x.Y,(-x.X*sy)+(x.Z*cy));return new float3((y.X*cz)-(y.Y*sz),(y.X*sz)+(y.Y*cz),y.Z);}
}
