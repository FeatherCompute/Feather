// ── Typed Shader IR Model ───────────────────────────────────────────────────
// Canonical typed intermediate representation for GPU shader code.
// Every expression, l-value, and statement carries a precise ShaderType.
// ──────────────────────────────────────────────────────────────────────────────

using Microsoft.CodeAnalysis;

namespace Feather.Generators.Model;

// ── Types ────────────────────────────────────────────────────────────────────

internal abstract record ShaderType
{
    public string CSharpTypeName { get; init; } = string.Empty;
    private protected ShaderType() { }
}

internal sealed record ShaderPrimitiveType(ShaderPrimitiveKind Kind) : ShaderType;
internal sealed record ShaderVectorType(ShaderPrimitiveType ElementType, int ComponentCount) : ShaderType;
internal sealed record ShaderMatrixType(ShaderPrimitiveType ElementType, int Rows, int Columns) : ShaderType;

internal sealed record ShaderStructType(
    string Name, string FullyQualifiedMetadataName,
    EquatableArray<ShaderStructField> Fields, int SizeInBytes, int Alignment) : ShaderType;

internal sealed record ShaderStructField(string Name, ShaderType Type, int Offset, int SizeInBytes, ShaderStructFieldFlags Flags = ShaderStructFieldFlags.None);

[Flags]
internal enum ShaderStructFieldFlags : uint
{
    None = 0,
    Position = 1,
    Color = 2,
    ColorIndexShift = 8
}

internal sealed record ShaderArrayType(ShaderType ElementType, int? Length) : ShaderType;

internal sealed record ShaderResourceWrapperType(
    ShaderResourceKind Kind, ShaderType ElementType, ShaderResourceAccess Access) : ShaderType;

internal sealed record ShaderVoidType : ShaderType;

internal enum ShaderPrimitiveKind : byte { Bool, Int, UInt, Float }
internal enum ShaderResourceKind : byte { Buffer, Texture2D, Texture3D, Sampler }
internal enum ShaderResourceAccess : byte { Read, Write, ReadWrite, Sample }

// ── Expressions ──────────────────────────────────────────────────────────────

internal abstract record ShaderExpression(ShaderType Type);

internal sealed record ShaderLiteralExpression(ShaderType Type, string ValueText) : ShaderExpression(Type);
internal sealed record ShaderLocalReferenceExpression(ShaderType Type, string Name, ISymbol Symbol) : ShaderExpression(Type);
internal sealed record ShaderParameterReferenceExpression(ShaderType Type, string Name, ISymbol Symbol) : ShaderExpression(Type);
internal sealed record ShaderFieldReferenceExpression(ShaderType Type, ShaderExpression Instance, IFieldSymbol Field) : ShaderExpression(Type);
internal sealed record ShaderResourceElementExpression(ShaderType Type, string ResourceName, ShaderExpression Index, ISymbol IndexSymbol) : ShaderExpression(Type);
internal sealed record ShaderUnaryExpression(ShaderType Type, ShaderUnaryOperator Operator, ShaderExpression Operand) : ShaderExpression(Type);
internal sealed record ShaderBinaryExpression(ShaderType Type, ShaderBinaryOperator Operator, ShaderExpression Left, ShaderExpression Right) : ShaderExpression(Type);
internal sealed record ShaderComparisonExpression(ShaderType Type, ShaderCompareOperator Operator, ShaderExpression Left, ShaderExpression Right) : ShaderExpression(Type);
internal sealed record ShaderLogicalExpression(ShaderType Type, ShaderLogicalOperator Operator, ShaderExpression Left, ShaderExpression Right) : ShaderExpression(Type);
internal sealed record ShaderConditionalExpression(ShaderType Type, ShaderExpression Condition, ShaderExpression WhenTrue, ShaderExpression WhenFalse) : ShaderExpression(Type);
internal sealed record ShaderConversionExpression(ShaderType Type, ShaderExpression Operand) : ShaderExpression(Type);
internal sealed record ShaderConstructorExpression(ShaderType Type, EquatableArray<ShaderExpression> Arguments) : ShaderExpression(Type);
internal sealed record ShaderIntrinsicCallExpression(ShaderType Type, string IntrinsicName, EquatableArray<ShaderExpression> Arguments) : ShaderExpression(Type);
internal sealed record ShaderCallableCallExpression(ShaderType Type, string CallableName, ISymbol TargetMethod, EquatableArray<ShaderExpression> Arguments) : ShaderExpression(Type);
internal sealed record ShaderAtomicExpression(ShaderType Type, ShaderAtomicOperation Operation, ShaderLValue Target, EquatableArray<ShaderExpression> Arguments) : ShaderExpression(Type);
internal sealed record ShaderTextureSampleExpression(
    ShaderType Type,
    ShaderTextureSampleOperation Operation,
    ShaderExpression Texture,
    ShaderExpression Sampler,
    ShaderExpression Uv,
    ShaderExpression? Lod,
    ShaderExpression? Ddx,
    ShaderExpression? Ddy) : ShaderExpression(Type);
internal sealed record ShaderSwizzleExpression(ShaderType Type, ShaderExpression Vector, string SwizzleComponents) : ShaderExpression(Type);
internal sealed record ShaderMemberAccessExpression(ShaderType Type, ShaderExpression Instance, ShaderStructField Field) : ShaderExpression(Type);
internal sealed record ShaderIndexAccessExpression(ShaderType Type, ShaderExpression Array, ShaderExpression Index) : ShaderExpression(Type);
internal sealed record ShaderBuiltinExpression(ShaderType Type, ShaderBuiltinKind Kind, int Component = 0) : ShaderExpression(Type);
internal sealed record ShaderPushConstantExpression(ShaderType Type, string ResourceName, uint Binding) : ShaderExpression(Type);
internal sealed record ShaderMatrixColumnExpression(ShaderType Type, ShaderExpression Matrix, ShaderExpression ColumnIndex) : ShaderExpression(Type);
internal sealed record ShaderSharedMemoryElementExpression(ShaderType Type, string Name, ShaderExpression Index) : ShaderExpression(Type);

internal enum ShaderUnaryOperator : byte { Negate, Not, BitwiseNot }
internal enum ShaderBinaryOperator : byte { Add, Subtract, Multiply, Divide, Modulo, BitwiseAnd, BitwiseOr, BitwiseXor, ShiftLeft, ShiftRight }
internal enum ShaderCompareOperator : byte { Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual }
internal enum ShaderLogicalOperator : byte { And, Or }
internal enum ShaderTextureSampleOperation : byte { Sample, SampleLevel, SampleGrad }

internal enum ShaderAtomicOperation : byte
{
    Add,
    Sub,
    Min,
    Max,
    And,
    Or,
    Xor,
    Exchange,
    CompareExchange
}

internal enum ShaderBuiltinKind : byte
{
    ThreadIndexX = 1, ThreadIndexY = 2, ThreadIndexZ = 3,
    LocalIndexX = 4, LocalIndexY = 5, LocalIndexZ = 6,
    GroupIdX = 7, GroupIdY = 8, GroupIdZ = 9,
    DispatchSizeX = 10, DispatchSizeY = 11, DispatchSizeZ = 12,
    GroupSizeX = 13, GroupSizeY = 14, GroupSizeZ = 15,
    VertexIndex = 16,
    InstanceIndex = 17,
    FragmentCoordX = 18, FragmentCoordY = 19, FragmentCoordZ = 20, FragmentCoordW = 21
}

// ── L-Values ─────────────────────────────────────────────────────────────────

internal abstract record ShaderLValue(ShaderType Type);
internal sealed record ShaderLocalLValue(ShaderType Type, string Name, ISymbol Symbol) : ShaderLValue(Type);
internal sealed record ShaderParameterLValue(ShaderType Type, string Name, ISymbol Symbol) : ShaderLValue(Type);
internal sealed record ShaderFieldLValue(ShaderType Type, ShaderLValue? Instance, IFieldSymbol Field) : ShaderLValue(Type);
internal sealed record ShaderResourceElementLValue(ShaderType Type, string ResourceName, ShaderExpression Index, ISymbol IndexSymbol) : ShaderLValue(Type);
internal sealed record ShaderSwizzleLValue(ShaderType Type, ShaderExpression Vector, string SwizzleComponents) : ShaderLValue(Type);
internal sealed record ShaderMemberAccessLValue(ShaderType Type, ShaderLValue Instance, ShaderStructField Field) : ShaderLValue(Type);
internal sealed record ShaderIndexAccessLValue(ShaderType Type, ShaderLValue Array, ShaderExpression Index) : ShaderLValue(Type);
internal sealed record ShaderMatrixColumnLValue(ShaderType Type, ShaderExpression Matrix, ShaderExpression ColumnIndex) : ShaderLValue(Type);
internal sealed record ShaderSharedMemoryElementLValue(ShaderType Type, string Name, ShaderExpression Index) : ShaderLValue(Type);

// ── Statements ───────────────────────────────────────────────────────────────

internal abstract record ShaderStatement;
internal sealed record ShaderBlockStatement(EquatableArray<ShaderStatement> Statements) : ShaderStatement;
internal sealed record ShaderLocalDeclarationStatement(string VariableName, ShaderType Type, ShaderExpression? Initializer, ISymbol Symbol) : ShaderStatement;
internal sealed record ShaderAssignmentStatement(ShaderLValue Target, ShaderExpression Value) : ShaderStatement;
internal sealed record ShaderCompoundAssignmentStatement(ShaderLValue Target, ShaderBinaryOperator Operator, ShaderExpression Value) : ShaderStatement;
internal sealed record ShaderIncrementDecrementStatement(ShaderLValue Target, bool IsIncrement, bool IsPrefix) : ShaderStatement;
internal sealed record ShaderIfStatement(ShaderExpression Condition, ShaderBlockStatement Then, ShaderBlockStatement? Else) : ShaderStatement;
internal sealed record ShaderForStatement(ShaderStatement? Init, ShaderExpression? Condition, ShaderStatement? Step, ShaderBlockStatement Body) : ShaderStatement;
internal sealed record ShaderWhileStatement(ShaderExpression Condition, ShaderBlockStatement Body) : ShaderStatement;
internal sealed record ShaderDoWhileStatement(ShaderBlockStatement Body, ShaderExpression Condition) : ShaderStatement;
internal sealed record ShaderBreakStatement : ShaderStatement;
internal sealed record ShaderContinueStatement : ShaderStatement;
internal sealed record ShaderReturnStatement(ShaderExpression? Value) : ShaderStatement;
internal sealed record ShaderExpressionStatement(ShaderExpression Expression) : ShaderStatement;
internal sealed record ShaderBarrierStatement(ShaderBarrierKind Kind) : ShaderStatement;
internal sealed record ShaderSharedMemoryDeclarationStatement(string VariableName, ShaderType ElementType, int Length, ISymbol Symbol) : ShaderStatement;

internal enum ShaderBarrierKind : byte { Workgroup, Memory, Full }

// ── Functions and Module ─────────────────────────────────────────────────────

internal enum ShaderFunctionKind : byte { Compute1D, Compute2D, Compute3D, Vertex, Fragment, Callable }

internal sealed record ShaderParameterModel(string Name, ShaderType Type, ShaderParameterDirection Direction);
internal enum ShaderParameterDirection : byte { In, Out, InOut }

internal sealed record ShaderFunctionModel(
    string Name, string MangledName, ShaderFunctionKind Kind, ShaderType ReturnType,
    EquatableArray<ShaderParameterModel> Parameters, ShaderBlockStatement Body);

internal sealed record ShaderModuleModel(
    ShaderFunctionModel EntryPoint,
    EquatableArray<ShaderFunctionModel> Callables,
    EquatableArray<ResourceModel> Resources,
    EquatableArray<ShaderStructType> Structs,
    ThreadGroupModel ThreadGroup,
    string Name,
    string Namespace);
