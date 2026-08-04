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
option(FEATHER_BUILD_LUISA_COMPUTE "Build LuisaCompute's Vulkan backend for the future XIR path" ON)
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
    set(LUISA_COMPUTE_ENABLE_DSL OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_TENSOR OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_DX OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_METAL OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_HIP OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_CUDA_EXT_LCUB OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_VULKAN ON CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_CPU OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_FALLBACK OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_REMOTE OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_GUI OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_CLANG_CXX OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_RUST OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV ON CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV OFF CACHE BOOL "" FORCE)
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
    if(MSVC)
        target_compile_options(luisa-compute-spirv PRIVATE "/FIluisa/ast/type.h")
    else()
        target_compile_options(luisa-compute-spirv PRIVATE "-include" "luisa/ast/type.h")
    endif()

    # Xcode 15.4 cannot construct C++20 aggregates through emplace. Generate
    # narrowly patched build-tree copies instead of modifying the pinned
    # LuisaCompute checkout. The replacements are also valid on GCC and MSVC.
    set(_feather_luisa_patch_dir "${CMAKE_CURRENT_BINARY_DIR}/LuisaCompute-patches")
    file(MAKE_DIRECTORY "${_feather_luisa_patch_dir}")

    set(_feather_luisa_hlsl_source
        "${CMAKE_CURRENT_SOURCE_DIR}/../LuisaCompute/src/backends/common/hlsl/hlsl_codegen.cpp")
    file(READ "${_feather_luisa_hlsl_source}" _feather_luisa_hlsl_contents)
    string(REPLACE
        "sharedVariables->emplace(f, v);"
        "sharedVariables->emplace(SharedVar{f, v});"
        _feather_luisa_hlsl_patched
        "${_feather_luisa_hlsl_contents}")
    if(_feather_luisa_hlsl_patched STREQUAL _feather_luisa_hlsl_contents)
        message(FATAL_ERROR "LuisaCompute hlsl_codegen.cpp no longer matches Feather's compatibility patch")
    endif()
    set(_feather_luisa_hlsl_patched_source
        "${_feather_luisa_patch_dir}/hlsl_codegen.cpp")
    file(WRITE "${_feather_luisa_hlsl_patched_source}" "${_feather_luisa_hlsl_patched}")
    get_target_property(_feather_luisa_hlsl_sources luisa-compute-hlsl-codegen SOURCES)
    list(REMOVE_ITEM _feather_luisa_hlsl_sources
        "${_feather_luisa_hlsl_source}" "hlsl_codegen.cpp")
    set_property(TARGET luisa-compute-hlsl-codegen PROPERTY SOURCES "${_feather_luisa_hlsl_sources}")
    target_sources(luisa-compute-hlsl-codegen PRIVATE "${_feather_luisa_hlsl_patched_source}")
    target_include_directories(luisa-compute-hlsl-codegen PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../LuisaCompute/src/backends/common/hlsl")

    set(_feather_luisa_restructure_source
        "${CMAKE_CURRENT_SOURCE_DIR}/../LuisaCompute/src/xir/passes/restructure_cfg.cpp")
    file(READ "${_feather_luisa_restructure_source}" _feather_luisa_restructure_contents)
    string(REPLACE
        "loops.emplace_back(loop->prepare_block(), loop->body_block(), loop->update_block(), loop->merge_block());"
        "loops.emplace_back(LoopSite{loop->prepare_block(), loop->body_block(), loop->update_block(), loop->merge_block()});"
        _feather_luisa_restructure_patched
        "${_feather_luisa_restructure_contents}")
    string(REPLACE
        "loops.emplace_back(loop->body_block(), loop->body_block(), loop->body_block(), loop->merge_block());"
        "loops.emplace_back(LoopSite{loop->body_block(), loop->body_block(), loop->body_block(), loop->merge_block()});"
        _feather_luisa_restructure_patched
        "${_feather_luisa_restructure_patched}")
    string(REPLACE
        "loops.emplace_back(\n                bb, loop->prepare_block(),\n                loop->body_block(),\n                loop->update_block(), loop->merge_block(),\n                loop->update_block(),\n                dom.contains(bb) ? dom_depth(dom, bb) : 0u);"
        "loops.emplace_back(\n                LoopSite{bb, loop->prepare_block(),\n                loop->body_block(),\n                loop->update_block(), loop->merge_block(),\n                loop->update_block(),\n                dom.contains(bb) ? dom_depth(dom, bb) : 0u});"
        _feather_luisa_restructure_patched
        "${_feather_luisa_restructure_patched}")
    string(REPLACE
        "loops.emplace_back(\n                bb, loop->body_block(),\n                loop->body_block(),\n                loop->body_block(), loop->merge_block(),\n                loop->merge_block(),\n                dom.contains(bb) ? dom_depth(dom, bb) : 0u);"
        "loops.emplace_back(\n                LoopSite{bb, loop->body_block(),\n                loop->body_block(),\n                loop->body_block(), loop->merge_block(),\n                loop->merge_block(),\n                dom.contains(bb) ? dom_depth(dom, bb) : 0u});"
        _feather_luisa_restructure_patched
        "${_feather_luisa_restructure_patched}")
    if(_feather_luisa_restructure_patched STREQUAL _feather_luisa_restructure_contents)
        message(FATAL_ERROR "LuisaCompute restructure_cfg.cpp no longer matches Feather's compatibility patch")
    endif()
    set(_feather_luisa_restructure_patched_source
        "${_feather_luisa_patch_dir}/restructure_cfg.cpp")
    file(WRITE "${_feather_luisa_restructure_patched_source}" "${_feather_luisa_restructure_patched}")
    get_target_property(_feather_luisa_xir_sources luisa-compute-xir SOURCES)
    list(REMOVE_ITEM _feather_luisa_xir_sources
        "${_feather_luisa_restructure_source}" "passes/restructure_cfg.cpp")
    set_property(TARGET luisa-compute-xir PROPERTY SOURCES "${_feather_luisa_xir_sources}")
    target_sources(luisa-compute-xir PRIVATE "${_feather_luisa_restructure_patched_source}")
    target_include_directories(luisa-compute-xir PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../LuisaCompute/src/xir"
        "${CMAKE_CURRENT_SOURCE_DIR}/../LuisaCompute/src/xir/passes")

    add_dependencies(feather luisa-compute-backend-vk)
endif()
