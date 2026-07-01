cmake_minimum_required(VERSION 3.20)

project(FeatherNative LANGUAGES CXX)

set(EASYGPU_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(EASYGPU_BUILD_TESTS OFF CACHE BOOL "" FORCE)
option(FEATHER_BUILD_WINDOW "Build Feather native window and texture presentation support" ON)
set(EASYGPU_BUILD_WINDOW ${FEATHER_BUILD_WINDOW} CACHE BOOL "" FORCE)
set(EASYGPU_BUILD_WINDOW_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../EasyGPU" "${CMAKE_CURRENT_BINARY_DIR}/EasyGPU")

add_library(feather SHARED
    feather_c_api.cpp
    feather_ir_bridge.cpp
    feather_typed_ir.cpp
    feather_typed_ir_lowerer.cpp
)

target_compile_features(feather PRIVATE cxx_std_20)
target_compile_definitions(feather PRIVATE FEATHER_BUILDING_DLL=1)
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

option(FEATHER_BUILD_AD_NATIVE_COVERAGE_PROBE "Build the native AD coverage probe executable" OFF)
if(FEATHER_BUILD_AD_NATIVE_COVERAGE_PROBE)
    add_executable(feather_ad_native_coverage_probe
        ad_native_coverage_probe.cpp
    )
    target_compile_features(feather_ad_native_coverage_probe PRIVATE cxx_std_20)
    target_link_libraries(feather_ad_native_coverage_probe PRIVATE EasyGPU::EasyGPU)
endif()
