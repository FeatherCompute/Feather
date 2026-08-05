# LuisaCompute Metal Codegen Bugs

These failures reproduce with LuisaCompute commit
`0c84a1aa7ad4181373040fc018f20d735f73c626` on Apple M5. Feather translates the
kernel through `FEIR -> XIR -> XIR2AST`; selecting `FEATHER_LUISA_BACKEND=metal`
then makes LC generate and compile MSL. Each failure aborts at
`src/backends/metal/metal_compiler.cpp:402` after `newLibrary` returns an error.

Run one reproduction from the repository root after building the native runtime:

```sh
FEATHER_LUISA_BACKEND=metal dotnet test \
  tests/Feather.Integration.Tests/Feather.Integration.Tests.csproj \
  --filter 'FullyQualifiedName=<test name>'
```

## Sampled texture access

LC maps `TEXTURE2D_SAMPLE`, `TEXTURE2D_SAMPLE_LEVEL`, and
`TEXTURE2D_SAMPLE_GRAD` to `texture_sample*` in
`src/backends/metal/metal_codegen_ast.cpp:1159`. The helpers at
`src/backends/metal/metal_builtin/metal_device_lib.metal:261` accept
`texture2d<T, access a>` and call `.sample(...)`. The generated resource is
`texture2d<..., access::read>`, for which the Metal compiler reports
`no member named 'sample'`.

Three kernels expose distinct affected shapes:

- `TextureSamplingAndMixedResourceOrderMatchEasyGpu`: sample a 2D `float4`
  texture with both implicit and explicit LOD, alongside a writable buffer.
- `TextureSampleGradExecutesThroughLuisaXir`: sample a 2D `float4` texture with
  explicit UV gradients.
- `ShaderLibraryTextureAndSamplerCallablesMatchEasyGpu`: pass an `rgba8`
  sampled texture and sampler through an LC callable, sample it, and return the
  red channel to a buffer.

The likely LC fix is to emit sampled textures with Metal's sample-compatible
access mode (or separate sampled and storage texture parameter types) while
preserving read/write storage texture behavior. Feather has no semantic
lowering workaround: replacing filtered sampling with loads would change the
kernel contract.

## Swizzle reference binding

`VectorConstructionAndSwizzlesMatchEasyGpu` reads `float4` values, constructs
the multi-component swizzles `YX`, `ZXY`, `WZYX`, and `BGRA`, then reads scalar
components from those results. LC emits multi-component swizzles in
`src/backends/metal/metal_codegen_ast.cpp:788` and scalar vector access through
`vector_element_ref` at line 810. The non-const helper at
`src/backends/metal/metal_builtin/metal_device_lib.metal:190` requires an lvalue
reference, but the generated swizzle expression is a temporary. Metal reports
`non-const reference cannot bind to vector element`.

The likely LC fix is value-category-aware access generation: use the const/value
overload for temporary swizzles and retain the reference overload only for true
lvalues. Feather does not special-case this kernel; LC must preserve writable
component access elsewhere.

`LuisaBackendMetalTests` isolates all four aborting cases in subprocesses and
checks these exact compiler diagnostics. The other 19 Luisa parity cases pass
through Metal.
