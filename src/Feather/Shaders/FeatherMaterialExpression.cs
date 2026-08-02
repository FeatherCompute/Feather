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
}

[ShaderLibrary]
public static class FeatherMaterialExpression
{
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
        if (instruction.Op == 1) result = new float4(uv, 0.0f, 1.0f);
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
            var nearest = 1000.0f; var selected = float3.Zero;
            for (var y = -1; y <= 1; y++) for (var x = -1; x <= 1; x++)
            {
                var candidate = cell + new float3((float)x, (float)y, 0.0f);
                var jitter = new float3(Hash(candidate), Hash(candidate + new float3(7.0f)), 0.0f) * c.X;
                var distance = ShaderMath.Length((candidate + jitter) - p);
                if (distance < nearest) { nearest = distance; selected = candidate; }
            }
            result = instruction.Parameters.X > 0.5f
                ? new float4(Hash(selected), Hash(selected + new float3(13.0f)), Hash(selected + new float3(29.0f)), 1.0f)
                : new float4(nearest);
        }
        else if (instruction.Op == 6)
        {
            var value = a.X;
            if (instruction.Parameters.X == 1.0f) value *= value;
            else if (instruction.Parameters.X == 2.0f) value = ShaderMath.Smoothstep(0.0f, 1.0f, value);
            else if (instruction.Parameters.X == 3.0f) value = (a.X + a.Y) * 0.5f;
            else if (instruction.Parameters.X >= 4.0f) { value = ShaderMath.Length(a.XYZ); if (instruction.Parameters.X == 5.0f) value *= value; }
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
            result = ShaderMath.Lerp(b, c, factor);
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
            result = new float4(ShaderMath.Lerp(d.X, e.X, t));
        }
        else if (instruction.Op == 14)
        {
            var blend = c;
            if (instruction.Parameters.X == 1.0f) blend = b + c;
            else if (instruction.Parameters.X == 2.0f) blend = b * c;
            else if (instruction.Parameters.X == 3.0f) blend = b - c;
            else if (instruction.Parameters.X == 6.0f) blend = ShaderMath.Abs(b - c);
            else if (instruction.Parameters.X == 7.0f) blend = ShaderMath.Min(b, c);
            else if (instruction.Parameters.X == 8.0f) blend = ShaderMath.Max(b, c);
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
        return result;
    }

    [Callable] public static RasterMaterialRegisters Set(RasterMaterialRegisters r, int i, float4 v)
    {
        if (i == 0) r.R0=v; else if(i==1)r.R1=v; else if(i==2)r.R2=v; else if(i==3)r.R3=v;
        else if(i==4)r.R4=v; else if(i==5)r.R5=v; else if(i==6)r.R6=v; else if(i==7)r.R7=v;
        else if(i==8)r.R8=v; else if(i==9)r.R9=v; else if(i==10)r.R10=v; else if(i==11)r.R11=v;
        else if(i==12)r.R12=v; else if(i==13)r.R13=v; else if(i==14)r.R14=v; else if(i==15)r.R15=v;
        else if(i==16)r.R16=v; else if(i==17)r.R17=v; else if(i==18)r.R18=v; else if(i==19)r.R19=v;
        else if(i==20)r.R20=v; else if(i==21)r.R21=v; else if(i==22)r.R22=v; else if(i==23)r.R23=v;
        else if(i==24)r.R24=v; else if(i==25)r.R25=v; else if(i==26)r.R26=v; else if(i==27)r.R27=v;
        else if(i==28)r.R28=v; else if(i==29)r.R29=v; else if(i==30)r.R30=v; else if(i==31)r.R31=v;
        else if(i==32)r.R32=v; else if(i==33)r.R33=v; else if(i==34)r.R34=v; else if(i==35)r.R35=v;
        else if(i==36)r.R36=v; else if(i==37)r.R37=v; else if(i==38)r.R38=v; else if(i==39)r.R39=v;
        else if(i==40)r.R40=v; else if(i==41)r.R41=v; else if(i==42)r.R42=v; else if(i==43)r.R43=v;
        else if(i==44)r.R44=v; else if(i==45)r.R45=v; else if(i==46)r.R46=v; else if(i==47)r.R47=v;
        else if(i==48)r.R48=v; else if(i==49)r.R49=v; else if(i==50)r.R50=v; else if(i==51)r.R51=v;
        else if(i==52)r.R52=v; else if(i==53)r.R53=v; else if(i==54)r.R54=v; else if(i==55)r.R55=v;
        else if(i==56)r.R56=v; else if(i==57)r.R57=v; else if(i==58)r.R58=v; else if(i==59)r.R59=v;
        else if(i==60)r.R60=v; else if(i==61)r.R61=v; else if(i==62)r.R62=v; else if(i==63)r.R63=v;
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
    {if(op==0)return a+b;if(op==1)return a-b;if(op==2)return a*b;if(op==3)return ShaderMath.Abs(b)<1e-8f?0:a/b;if(op==4)return(a*b)+c;if(op==5)return ShaderMath.Pow(ShaderMath.Abs(a),b);if(op==6)return ShaderMath.Min(a,b);if(op==7)return ShaderMath.Max(a,b);if(op==8)return a<b?1:0;if(op==9)return a>b?1:0;if(op==10)return ShaderMath.Abs(a);if(op==11)return ShaderMath.Sqrt(ShaderMath.Max(a,0));if(op==12)return ShaderMath.Floor(a);if(op==13)return ShaderMath.Ceil(a);if(op==14)return ShaderMath.Fract(a);if(op==16)return ShaderMath.Sin(a);if(op==17)return ShaderMath.Cos(a);if(op==18)return ShaderMath.Tan(a);return a;}
    [Callable] private static float4 VectorMath(float op,float4 a,float4 b,float4 scale)
    {var av=a.XYZ;var bv=b.XYZ;if(op==0)return new float4(av+bv,1);if(op==1)return new float4(av-bv,1);if(op==2)return new float4(av*bv,1);if(op==4)return new float4(ShaderMath.Cross(av,bv),1);if(op==5)return new float4(ShaderMath.Dot(av,bv));if(op==6)return new float4(ShaderMath.Length(av-bv));if(op==7)return new float4(ShaderMath.Length(av));if(op==8)return new float4(av*scale.X,1);if(op==9)return new float4(ShaderMath.Normalize(av),1);if(op==10)return new float4(ShaderMath.Abs(av),1);return a;}
    [Callable] private static float3 RgbToHsv(float3 c){var max=ShaderMath.Max(c.X,ShaderMath.Max(c.Y,c.Z));var min=ShaderMath.Min(c.X,ShaderMath.Min(c.Y,c.Z));var delta=max-min;var h=0.0f;if(delta>1e-6f){if(max==c.X)h=(c.Y-c.Z)/delta;else if(max==c.Y)h=2+((c.Z-c.X)/delta);else h=4+((c.X-c.Y)/delta);h=ShaderMath.Fract(h/6);}return new float3(h,max<=1e-6f?0:delta/max,max);}
    [Callable] private static float3 HsvToRgb(float3 h){var x=ShaderMath.Fract(h.X)*6;var s=ShaderMath.Floor(x);var f=x-s;var p=h.Z*(1-h.Y);var q=h.Z*(1-(h.Y*f));var t=h.Z*(1-(h.Y*(1-f)));if(s<1)return new float3(h.Z,t,p);if(s<2)return new float3(q,h.Z,p);if(s<3)return new float3(p,h.Z,t);if(s<4)return new float3(p,q,h.Z);if(s<5)return new float3(t,p,h.Z);return new float3(h.Z,p,q);}
    [Callable] private static float3 Rotate(float3 v,float3 r){var cx=ShaderMath.Cos(r.X);var sx=ShaderMath.Sin(r.X);var cy=ShaderMath.Cos(r.Y);var sy=ShaderMath.Sin(r.Y);var cz=ShaderMath.Cos(r.Z);var sz=ShaderMath.Sin(r.Z);var x=new float3(v.X,(v.Y*cx)-(v.Z*sx),(v.Y*sx)+(v.Z*cx));var y=new float3((x.X*cy)+(x.Z*sy),x.Y,(-x.X*sy)+(x.Z*cy));return new float3((y.X*cz)-(y.Y*sz),(y.X*sz)+(y.Y*cz),y.Z);}
}
