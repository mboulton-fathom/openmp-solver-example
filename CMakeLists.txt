cmake_minimum_required(VERSION 3.18...3.31)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(JACOBI_OFFLOAD "" CACHE STRING "OpenMP GPU offload target: empty, nvidia or amd")
set(JACOBI_GPU_ARCH "gfx90a" CACHE STRING "AMD GPU architecture for JACOBI_OFFLOAD=amd")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(nanobind CONFIG REQUIRED)

if(JACOBI_OFFLOAD STREQUAL "nvidia")
  set(OFFLOAD_FLAGS -fopenmp-targets=nvptx64-nvidia-cuda)
elseif(JACOBI_OFFLOAD STREQUAL "amd")
  set(OFFLOAD_FLAGS -fopenmp-targets=amdgcn-amd-amdhsa
                    -Xopenmp-target=amdgcn-amd-amdhsa -march=${JACOBI_GPU_ARCH})
elseif(NOT JACOBI_OFFLOAD STREQUAL "")
  message(FATAL_ERROR "JACOBI_OFFLOAD must be empty, nvidia or amd")
endif()
if(OFFLOAD_FLAGS AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR "JACOBI_OFFLOAD needs clang++. Set CXX=clang++.")
endif()

nanobind_add_module(_jacobi src/cpp/bindings.cpp src/cpp/jacobi.cpp)
target_link_libraries(_jacobi PRIVATE OpenMP::OpenMP_CXX)
# Offload flags go on the solver only, so that nanobind is not compiled for the device.
set_source_files_properties(src/cpp/jacobi.cpp PROPERTIES COMPILE_OPTIONS "${OFFLOAD_FLAGS}")
target_link_options(_jacobi PRIVATE ${OFFLOAD_FLAGS})

install(TARGETS _jacobi LIBRARY DESTINATION jacobi)
