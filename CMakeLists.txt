#
# This software contains source code provided by NVIDIA Corporation.

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(mdk-nvjp2k VERSION 0.28.0 DESCRIPTION "mdk nvJPEG2000 plugin")
set(PROJECT_VERSION_TWEAK 0)

find_package(CUDAToolkit)

find_library(NVJPEG2K_LIB
     NAMES nvjpeg2k
     PATHS ${NVJPEG2K_PATH}/lib64/${CUDAToolkit_VERSION_MAJOR}
           ${NVJPEG2K_PATH}/lib/${CUDAToolkit_VERSION_MAJOR})

if(NOT NVJPEG2K_LIB)
  message(FATAL_ERROR, " nvJPEG2000 library not found. Make sure the library is installed. If using the tar.xz/zip packages set -DNVJPEG2K_PATH")
else()
  message(INFO, " nvJPEG2000 library found here: ${NVJPEG2K_LIB}")
endif()


set(MDKSDK "${CMAKE_CURRENT_SOURCE_DIR}/mdk-sdk" CACHE STRING "libmdk SDK dir")

if(NOT CMAKE_PROJECT_NAME STREQUAL mdk) # not build in source tree
    list(APPEND CMAKE_MODULE_PATH ${MDKSDK}/lib/cmake)
endif()
find_package(MDK)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

setup_mdk_plugin(${PROJECT_NAME})

include_directories(
  ${NVJPEG2K_PATH}/include
  SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_sources(${PROJECT_NAME} PRIVATE
    nvJp2kDecoder.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    #VERSION ${PROJECT_VERSION} # -current_version can not be applied for MODULE
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME ${PROJECT_NAME}
    CLEAN_DIRECT_OUTPUT 1
  )

target_link_libraries(${PROJECT_NAME} PRIVATE ${NVJPEG2K_LIB} CUDA::cudart)
