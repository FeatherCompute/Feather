cmake_minimum_required(VERSION 3.20)

project(FeatherNative LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(EASYGPU_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(EASYGPU_BUILD_TESTS OFF CACHE BOOL "" FORCE)
option(FEATHER_BUILD_WINDOW "Build Feather native window and texture presentation support" ON)
option(FEATHER_BUILD_OPTIMIZATION_BENCHMARK "Build the Vulkan shader optimization benchmark" OFF)
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
