cmake_minimum_required(VERSION 3.16)
project(gemm VERSION 1.0.0)

set(LIB_NAME gemm_metal)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE SRC_FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/gemm/*.cpp"
)
file(GLOB_RECURSE INCLUDE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/gemm/*.h"
)
set(INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/gemm"
)

add_compile_options(-fsanitize=address -fsanitize=undefined)
link_libraries(-fsanitize=address -fsanitize=undefined)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/metal")
set(EXTRA_LIBS ${EXTRA_LIBS} METAL_CPP)

add_compile_options(-mcpu=native)

# Compiler Warnings
set(cxx_flags # Common and useful compiler warning flags.
    -Wall # Enable all common warnings.
    -Wextra # Enable extra warnings.
    -Wpedantic # Enable pedantic warnings (stricter standard conformance).
    -Wcast-qual # Warn about casting away qualifiers (const, volatile).
    -Wno-unused-function # Suppress warnings about unused functions (can be noisy in header-only libraries or during development).
) 
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>")

# Add Profiller
add_compile_options(-pg)

message(STATUS "Build shared library")
add_library(${LIB_NAME} SHARED ${SRC_FILES})

target_include_directories(${LIB_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
    $<INSTALL_INTERFACE:${INCLUDE_DIR}>
)
target_link_libraries(${LIB_NAME} PUBLIC 
                    ${EXTRA_LIBS})
target_compile_definitions(${LIB_NAME} PUBLIC KERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/")
target_compile_definitions(${LIB_NAME} PUBLIC OUTPUTS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/outputs/")
target_compile_features(${LIB_NAME} PRIVATE cxx_std_20)

set(debug_flags -g)
set(release_flags -O3)
target_compile_options(${LIB_NAME} PUBLIC "$<$<CONFIG:DEBUG>:${debug_flags}>")
target_compile_options(${LIB_NAME} PUBLIC "$<$<CONFIG:RELEASE>:${release_flags}>")

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${LIB_NAME})
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})