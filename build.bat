@echo off
echo === Building Matrix Multiplication Parallel Computing Project ===

REM Check if CMake is installed
where cmake >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: CMake not found. Please install CMake and ensure it's in PATH.
    exit /b 1
)

REM Check if CUDA Toolkit is installed
where nvcc >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Warning: NVCC not found. CUDA functionality may not be available.
    echo Please install NVIDIA CUDA Toolkit.
)

REM Clean old build directory
if exist "build" (
    echo Cleaning old build directory...
    rd /s /q build
)

REM Create build directory
mkdir build
cd build

REM Run CMake to generate Visual Studio solution
echo Generating Visual Studio solution...
cmake -G "Visual Studio 17 2022" -A x64 ..

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed, check errors above.
    cd ..
    pause
    exit /b 1
)

REM Build project
echo Building project...
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo Build failed, check errors above.
    cd ..
    pause
    exit /b 1
)

echo Build completed!

REM Test execution
if exist "Release\matrix_multiplication.exe" (
    echo Running performance test...
    Release\matrix_multiplication.exe
) else (
    echo Build appears successful, but executable not found. Check output path.
)

cd ..
pause