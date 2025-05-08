# Minimum CMake version required
cmake_minimum_required(VERSION 3.18)
project(MatrixMultiplication LANGUAGES CXX C CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 11)

# CUDA settings
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 52 75)  # Support SM 5.2 and 7.5+

# OpenMP settings - only for CPU code
find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# AVX settings
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

# ISPC settings
find_program(ISPC_EXECUTABLE ispc)
if(NOT ISPC_EXECUTABLE)
    message(STATUS "ISPC compiler not found. Trying to download and use a pre-built binary...")
    
    # Try to download pre-compiled version if ISPC is not found
    if(WIN32)
        set(ISPC_VERSION "1.18.0")
        set(ISPC_URL "https://github.com/ispc/ispc/releases/download/v${ISPC_VERSION}/ispc-v${ISPC_VERSION}-windows.zip")
        set(ISPC_DIR "${CMAKE_CURRENT_BINARY_DIR}/ispc")
        
        file(DOWNLOAD ${ISPC_URL} "${CMAKE_CURRENT_BINARY_DIR}/ispc.zip" SHOW_PROGRESS)
        file(MAKE_DIRECTORY ${ISPC_DIR})
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_CURRENT_BINARY_DIR}/ispc.zip"
            WORKING_DIRECTORY ${ISPC_DIR}
        )
        
        file(GLOB ISPC_BIN "${ISPC_DIR}/bin/ispc.exe")
        if(ISPC_BIN)
            set(ISPC_EXECUTABLE ${ISPC_BIN})
            message(STATUS "Downloaded ISPC: ${ISPC_EXECUTABLE}")
        else()
            message(FATAL_ERROR "Failed to download and extract ISPC")
        endif()
    else()
        message(FATAL_ERROR "ISPC compiler not found and automatic download only supports Windows")
    endif()
endif()

# ISPC compilation command
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/matrix_mul_ispc.obj
    COMMAND ${ISPC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_mul.ispc 
            -o ${CMAKE_CURRENT_BINARY_DIR}/matrix_mul_ispc.obj 
            --target=avx2-i32x8
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_mul.ispc
)

# CUDA files
set(CUDA_SOURCE_FILES
    src/matrix_mul_cuda.cu
)

# Add CUDA library
add_library(matrix_mul_cuda STATIC ${CUDA_SOURCE_FILES})
target_compile_options(matrix_mul_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Set CUDA library properties
set_target_properties(matrix_mul_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Find CUDA runtime library
find_library(CUDART_LIBRARY cudart
    HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
)

# Add executable and link
add_executable(matrix_multiplication src/matrix_mul.cpp ${CMAKE_CURRENT_BINARY_DIR}/matrix_mul_ispc.obj)
target_link_libraries(matrix_multiplication matrix_mul_cuda ${CUDART_LIBRARY})

# Add runtime library consistency option (resolves LNK4098 warning)
if(MSVC)
    set_target_properties(matrix_multiplication PROPERTIES 
        LINK_FLAGS "/NODEFAULTLIB:libcmt.lib /NODEFAULTLIB:libcmtd.lib"
    )
endif()