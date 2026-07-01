### New Rules

Rule ID | Category | Severity | Notes
--------|----------|----------|-------
FE0001 | Feather | Error | Shader type must be a readonly partial struct.
FE0002 | Feather | Error | Shader type must implement a supported shader interface.
FE0003 | Feather | Error | Shader entry point is invalid.
FE0004 | Feather | Error | Shader constructor parameter type is not supported.
FE0005 | Feather | Error | Captured field type is not supported.
FE0006 | Feather | Error | Unsupported statement in shader body.
FE0007 | Feather | Error | Unsupported expression in shader body.
FE0008 | Feather | Error | Unsupported method call in shader body.
FE0009 | Feather | Error | Unsupported control flow.
FE0010 | Feather | Error | Unsupported generic usage.
FE0011 | Feather | Error | Unsupported allocation in shader body.
FE0012 | Feather | Error | Unsupported exception handling in shader body.
FE0013 | Feather | Error | Unsupported async/await in shader body.
FE0014 | Feather | Error | Unsupported virtual/interface call in shader body.
FE0015 | Feather | Error | Recursive shader function is not supported.
FE0016 | Feather | Error | Resource access violates declared access mode.
FE0017 | Feather | Error | Buffer index type must be int-compatible.
FE0018 | Feather | Error | Texture index type must be int2 or int3.
FE0019 | Feather | Error | Struct layout is not std430-compatible.
FE0020 | Feather | Error | Matrix layout requires explicit Feather matrix type.
FE0021 | Feather | Error | Automatic differentiation marker is not supported here.
FE0022 | Feather | Error | Automatic differentiation source is not supported.
FE0023 | Feather | Error | Graphics shader varying type must be unmanaged and [GpuStruct].
FE0024 | Feather | Error | Fragment shader output type is not supported.
FE0025 | Feather | Error | ThreadGroupSize is invalid for kernel dimension.
FE0026 | Feather | Error | Elementwise expression intrinsic is not supported.
FE0027 | Feather | Error | Typed shader IR lowering failed.
FE0028 | Feather | Error | Top-level local is not supported in shader code.
