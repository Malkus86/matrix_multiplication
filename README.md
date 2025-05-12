# High-Performance Matrix Multiplication

A comprehensive implementation and performance comparison of matrix multiplication optimization techniques for multi-core architectures.

## Project Overview

This project demonstrates various optimization techniques for matrix multiplication (1024×1024 integers) on modern hardware:

- **Sequential**: Baseline implementation
- **Cache Blocking**: Improved memory access patterns
- **OpenMP**: Multi-threaded parallel execution
- **AVX**: SIMD vectorization with AVX2 instructions
- **ISPC**: Intel SPMD Program Compiler implementation
- **CUDA**: GPU-accelerated computation (optional)

Each implementation is benchmarked and compared to demonstrate the performance benefits of different optimization strategies.

## Key Features

- Extensive performance comparisons with detailed timing measurements
- Memory access pattern optimizations (cache blocking, matrix transposition)
- SIMD vectorization with AVX2 instructions
- Multi-threaded processing with OpenMP
- ISPC automatic vectorization
- CUDA implementation with shared memory optimization
- Automated build system supporting CMake

## Project Structure

```
matrix_multiplication/
│
├── src/
│   ├── matrix_mul.cpp       // Main code with sequential, OpenMP and AVX implementations
│   ├── matrix_mul.ispc      // ISPC implementation
│   ├── matrix_mul_cuda.cu   // CUDA implementation
│   └── timing.h             // High-precision timing utilities
│
├── CMakeLists.txt           // CMake configuration
│
└── build.bat                // Windows build script
```

## Performance Results

On a system with an Intel Core i5-13500 CPU and NVIDIA RTX 3070 GPU:

| Implementation           | Speedup (vs. Sequential) |
|--------------------------|--------------------------|
| Sequential               | 1.00× (baseline)         |
| Sequential with Blocking | 3.55×                    |
| OpenMP                   | 22.20×                   |
| AVX                      | 230.89×                  |
| AVX with Blocking        | 241.13×                  |
| ISPC                     | 8.05×                    |
| CUDA (with data transfer)| 536.61×                  |
| CUDA (computation only)  | 1253.63×                 |

## Requirements

- C++11 compatible compiler
- CMake 3.18 or higher
- OpenMP support
- CPU with AVX2 instruction set
- ISPC compiler (auto-downloaded if not found)
- CUDA Toolkit (optional, for GPU implementation)

## Building the Project

### Windows

Simply run the provided build script:

```
build.bat
```

This will:
1. Check for required dependencies
2. Configure the project with CMake
3. Build the executable
4. Run the performance tests automatically

### Manual Build

```bash
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

## Technical Details

- Matrix size: 1024×1024 integers
- Block size for cache optimization: 32×32
- AVX implementation uses matrix transposition for optimal memory access
- CUDA implementation uses shared memory tiling
- Each implementation is verified for correctness against the sequential version