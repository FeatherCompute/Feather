cmake_minimum_required(VERSION 3.26)

project(FeatherNative LANGUAGES C CXX)

if(APPLE)
    enable_language(OBJC)
    enable_language(OBJCXX)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Single-config generators leave CMAKE_BUILD_TYPE empty unless it is passed explicitly, and an empty
# build type means no -O flag at all. That is easy to miss: the library builds, links, and runs
# correctly, just several times slower -- a viewport frame measured 71ms unoptimized against 20ms at
# -O3, with the difference landing in per-pixel host-side work. Default to Release so a bare
# `cmake -S native -B native/build` cannot quietly produce a build nobody would want to profile.
get_property(_feather_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _feather_multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type for single-config generators" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE was unset; defaulting to Release.")
endif()

set(EASYGPU_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(EASYGPU_BUILD_TESTS OFF CACHE BOOL "" FORCE)
option(FEATHER_BUILD_WINDOW "Build Feather native window and texture presentation support" ON)
option(FEATHER_BUILD_OPTIMIZATION_BENCHMARK "Build the Vulkan shader optimization benchmark" OFF)
option(FEATHER_BUILD_LUISA_COMPUTE "Build LuisaCompute backends for the XIR path" ON)
option(FEATHER_LUISA_ENABLE_METAL "Build LuisaCompute's experimental Metal backend" ${APPLE})
option(FEATHER_LUISA_ENABLE_CUDA "Build LuisaCompute's experimental CUDA backend" OFF)
option(FEATHER_LUISA_ENABLE_HIP "Build LuisaCompute's experimental HIP backend" OFF)
set(FEATHER_LUISA_VULKAN_CODEGEN "XirSpirv" CACHE STRING
    "Luisa Vulkan compute codegen: XirSpirv, HlslSpirv, or LlvmSpirv")
set_property(CACHE FEATHER_LUISA_VULKAN_CODEGEN PROPERTY STRINGS
    XirSpirv HlslSpirv LlvmSpirv)
set(_feather_luisa_vulkan_codegen_options XirSpirv HlslSpirv LlvmSpirv)
if(NOT FEATHER_LUISA_VULKAN_CODEGEN IN_LIST _feather_luisa_vulkan_codegen_options)
    message(FATAL_ERROR
        "FEATHER_LUISA_VULKAN_CODEGEN must be one of: ${_feather_luisa_vulkan_codegen_options}")
endif()
set(FEATHER_SHADER_OPTIMIZATION_LEVEL "Ultra" CACHE STRING
    "SPIR-V optimization preset: None, Size, Aggressive, Ultra, or Extreme")
set_property(CACHE FEATHER_SHADER_OPTIMIZATION_LEVEL PROPERTY STRINGS
    None Size Aggressive Ultra Extreme)
set(_feather_shader_optimization_levels None Size Aggressive Ultra Extreme)
if(NOT FEATHER_SHADER_OPTIMIZATION_LEVEL IN_LIST _feather_shader_optimization_levels)
    message(FATAL_ERROR
        "FEATHER_SHADER_OPTIMIZATION_LEVEL must be one of: ${_feather_shader_optimization_levels}")
endif()
set(EASYGPU_BUILD_WINDOW ${FEATHER_BUILD_WINDOW} CACHE BOOL "" FORCE)
set(EASYGPU_BUILD_WINDOW_EXAMPLES OFF CACHE BOOL "" FORCE)
if(FEATHER_BUILD_OPTIMIZATION_BENCHMARK)
    set(EASYGPU_ENABLE_SPIRV_CROSS_GLSL ON CACHE BOOL "" FORCE)
endif()

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../EasyGPU" "${CMAKE_CURRENT_BINARY_DIR}/EasyGPU")

add_library(feather SHARED
    feather_c_api.cpp
    feather_ir_bridge.cpp
    feather_typed_ir.cpp
    feather_typed_ir_lowerer.cpp
)

if(FEATHER_BUILD_WINDOW)
    target_sources(feather PRIVATE feather_window_host.cpp)
endif()

target_compile_features(feather PRIVATE cxx_std_20)
target_compile_definitions(feather PRIVATE
    FEATHER_BUILDING_DLL=1
    FEATHER_SHADER_OPTIMIZATION_LEVEL=GPU::Backend::ShaderOptimizationLevel::${FEATHER_SHADER_OPTIMIZATION_LEVEL}
)
target_include_directories(feather PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(feather PRIVATE EasyGPU::EasyGPU)
if(FEATHER_BUILD_WINDOW AND TARGET EasyGPU::Window)
    target_link_libraries(feather PRIVATE EasyGPU::Window)
    target_compile_definitions(feather PRIVATE FEATHER_BUILD_WINDOW=1)
else()
    target_compile_definitions(feather PRIVATE FEATHER_BUILD_WINDOW=0)
endif()

set_target_properties(feather PROPERTIES
    OUTPUT_NAME feather
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

if(WIN32)
    set_target_properties(feather PROPERTIES
        OUTPUT_NAME feather_native
    )
endif()

option(FEATHER_BUILD_AD_NATIVE_COVERAGE_PROBE "Build the native AD coverage probe executable" OFF)
if(FEATHER_BUILD_AD_NATIVE_COVERAGE_PROBE)
    add_executable(feather_ad_native_coverage_probe
        ad_native_coverage_probe.cpp
    )
    target_compile_features(feather_ad_native_coverage_probe PRIVATE cxx_std_20)
    target_link_libraries(feather_ad_native_coverage_probe PRIVATE EasyGPU::EasyGPU)
endif()

if(FEATHER_BUILD_OPTIMIZATION_BENCHMARK)
    add_executable(feather_optimization_benchmark
        optimization_benchmark.cpp
    )
    target_compile_features(feather_optimization_benchmark PRIVATE cxx_std_20)
    target_link_libraries(feather_optimization_benchmark PRIVATE EasyGPU::EasyGPU)
endif()

# LuisaCompute changes global output-directory variables. Keep it after every
# Feather target has been created so the existing native asset layout remains
# stable while M1 verifies the Vulkan/XIR toolchain.
if(FEATHER_BUILD_LUISA_COMPUTE)
    # The opt-in compute rasterizer vertical slice uses the same Luisa C++ DSL
    # entry point as the runtime's own compute probes.
    set(LUISA_COMPUTE_ENABLE_DSL ON CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_TENSOR OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_DX OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_METAL ${FEATHER_LUISA_ENABLE_METAL} CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_HIP ${FEATHER_LUISA_ENABLE_HIP} CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_CUDA ${FEATHER_LUISA_ENABLE_CUDA} CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_CUDA_EXT_LCUB OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_VULKAN ON CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_CPU OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_FALLBACK OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_REMOTE OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_GUI OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_CLANG_CXX OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_RUST OFF CACHE BOOL "" FORCE)
    if(FEATHER_LUISA_VULKAN_CODEGEN STREQUAL "XirSpirv")
        set(LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV ON CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV OFF CACHE BOOL "" FORCE)
    elseif(FEATHER_LUISA_VULKAN_CODEGEN STREQUAL "HlslSpirv")
        set(LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV OFF CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV OFF CACHE BOOL "" FORCE)
    else()
        set(LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV OFF CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV ON CACHE BOOL "" FORCE)
    endif()
    set(LUISA_COMPUTE_ENABLE_SCCACHE OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_DOWNLOAD_OIDN OFF CACHE BOOL "" FORCE)
    # Use Luisa's portable STL mode. macOS-14's Apple Clang cannot compile
    # the bundled EASTL alias-template CTAD used by the Vulkan dependencies.
    set(LUISA_COMPUTE_USE_SYSTEM_STL ON CACHE BOOL "" FORCE)

    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../LuisaCompute" "${CMAKE_CURRENT_BINARY_DIR}/LuisaCompute")

    # LuisaCompute 0.9.0's Vulkan SPIR-V argument analysis dereferences
    # luisa::compute::Type while only its forward declaration is visible.
    # Supply its owning header for this one upstream target until the pinned
    # revision includes it directly. This keeps recursive checkouts
    # reproducible without carrying a modified LuisaCompute worktree.
    if(TARGET luisa-compute-spirv)
        if(MSVC)
            target_compile_options(luisa-compute-spirv PRIVATE "/FIluisa/ast/type.h")
        else()
            target_compile_options(luisa-compute-spirv PRIVATE "-include" "luisa/ast/type.h")
        endif()
    endif()

    target_sources(feather PRIVATE feather_luisa_backend.cpp feather_luisa_xir.cpp)
    target_link_libraries(feather PRIVATE luisa-compute-runtime luisa-compute-xir luisa-compute-dsl)
    target_compile_definitions(feather PRIVATE FEATHER_HAS_LUISA=1)
    set(_feather_luisa_backend_targets luisa-compute-backend-vk)
    foreach(_feather_backend IN ITEMS metal cuda hip)
        string(TOUPPER "${_feather_backend}" _feather_backend_upper)
        if(TARGET luisa-compute-backend-${_feather_backend})
            target_compile_definitions(feather PRIVATE FEATHER_HAS_LUISA_${_feather_backend_upper}=1)
            list(APPEND _feather_luisa_backend_targets luisa-compute-backend-${_feather_backend})
        else()
            target_compile_definitions(feather PRIVATE FEATHER_HAS_LUISA_${_feather_backend_upper}=0)
        endif()
    endforeach()
    if(APPLE)
        set(_feather_luisa_origin "@loader_path")
    elseif(UNIX)
        set(_feather_luisa_origin "$ORIGIN")
    endif()
    if(DEFINED _feather_luisa_origin)
        set_target_properties(feather luisa-compute-core luisa-compute-ast
            luisa-compute-runtime luisa-compute-xir luisa-compute-dsl ${_feather_luisa_backend_targets} PROPERTIES
            BUILD_RPATH "${_feather_luisa_origin}"
            INSTALL_RPATH "${_feather_luisa_origin}"
        )
    endif()
    add_dependencies(feather ${_feather_luisa_backend_targets})
else()
    target_compile_definitions(feather PRIVATE
        FEATHER_HAS_LUISA=0
        FEATHER_HAS_LUISA_METAL=0
        FEATHER_HAS_LUISA_CUDA=0
        FEATHER_HAS_LUISA_HIP=0)
endif()
