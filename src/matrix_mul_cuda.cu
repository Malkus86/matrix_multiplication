/**
 * Matrix Multiplication CUDA Implementation
 * Optimized for GPU parallel execution with detailed timing
 */

 #include <cuda_runtime.h>
 #include <device_launch_parameters.h>
 #include <stdio.h>
 
 #define BLOCK_SIZE 32  // Tile size for shared memory optimization
 
 /**
  * CUDA kernel function for matrix multiplication
  * This function runs on the GPU and computes matrix C = A*B
  * 
  * @param A Input matrix A
  * @param B Input matrix B
  * @param C Output matrix C
  * @param size Matrix dimension
  */
 __global__ void matrixMulKernel(int* A, int* B, int* C, int size) {
     // Calculate the row and column index of the element
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;
     
     // Shared memory for the sub-matrices of A and B
     __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
     __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];
     
     if (row < size && col < size) {
         int sum = 0;
         
         // Loop over all sub-matrices of A and B that are needed to compute C[row,col]
         for (int block = 0; block < (size + BLOCK_SIZE - 1) / BLOCK_SIZE; ++block) {
             // Load one tile of A and B into shared memory
             if (block * BLOCK_SIZE + threadIdx.x < size && row < size)
                 As[threadIdx.y][threadIdx.x] = A[row * size + block * BLOCK_SIZE + threadIdx.x];
             else
                 As[threadIdx.y][threadIdx.x] = 0;
                 
             if (block * BLOCK_SIZE + threadIdx.y < size && col < size)
                 Bs[threadIdx.y][threadIdx.x] = B[(block * BLOCK_SIZE + threadIdx.y) * size + col];
             else
                 Bs[threadIdx.y][threadIdx.x] = 0;
                 
             // Synchronize to ensure all threads have loaded their data before computation
             __syncthreads();
             
             // Multiply the two tiles together
             for (int k = 0; k < BLOCK_SIZE; ++k)
                 sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                 
             // Synchronize to ensure all threads are done with the computation
             // before loading new tiles in the next iteration
             __syncthreads();
         }
         
         // Write the result to global memory
         C[row * size + col] = sum;
     }
 }
 
 /**
  * C++ wrapper function for CUDA matrix multiplication
  * Handles memory transfers between host and device
  * 
  * @param h_A Host matrix A
  * @param h_B Host matrix B
  * @param h_C Host matrix C (result)
  * @param size Matrix dimension
  */
 extern "C" void matrix_multiply_cuda(int* h_A, int* h_B, int* h_C, int size) {
     int *d_A, *d_B, *d_C;  // Device pointers
     size_t matrix_size = size * size * sizeof(int);
     
     // Allocate device memory
     cudaMalloc((void**)&d_A, matrix_size);
     cudaMalloc((void**)&d_B, matrix_size);
     cudaMalloc((void**)&d_C, matrix_size);
     
     // Copy data from host to device
     cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
     cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);
     
     // Set up kernel launch parameters
     dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
     dim3 blocksPerGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                        (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
     
     // Launch the CUDA kernel
     matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);
     
     // Wait for kernel to complete
     cudaDeviceSynchronize();
     
     // Copy results back to host
     cudaMemcpy(h_C, d_C, matrix_size, cudaMemcpyDeviceToHost);
     
     // Free device memory
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);
 }
 
 /**
  * C++ wrapper function for CUDA matrix multiplication with detailed timing
  * Separates memory transfer and computation times
  * 
  * @param h_A Host matrix A
  * @param h_B Host matrix B
  * @param h_C Host matrix C (result)
  * @param size Matrix dimension
  * @param transfer_to_time Time for host-to-device transfer (output parameter)
  * @param compute_time Time for computation (output parameter)
  * @param transfer_from_time Time for device-to-host transfer (output parameter)
  */
 extern "C" void matrix_multiply_cuda_detailed(int* h_A, int* h_B, int* h_C, int size, 
                                              double* transfer_to_time, 
                                              double* compute_time, 
                                              double* transfer_from_time) {
     int *d_A, *d_B, *d_C;  // Device pointers
     size_t matrix_size = size * size * sizeof(int);
     
     // For measuring time
     cudaEvent_t start, stop;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
     float milliseconds = 0;
     
     // Allocate device memory
     cudaMalloc((void**)&d_A, matrix_size);
     cudaMalloc((void**)&d_B, matrix_size);
     cudaMalloc((void**)&d_C, matrix_size);
     
     // Measure host to device transfer time
     cudaEventRecord(start);
     cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
     cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);
     cudaEventRecord(stop);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&milliseconds, start, stop);
     *transfer_to_time = milliseconds / 1000.0;  // Convert to seconds
     
     // Set up kernel launch parameters
     dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
     dim3 blocksPerGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                        (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
     
     // Measure computation time
     cudaEventRecord(start);
     matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);
     cudaEventRecord(stop);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&milliseconds, start, stop);
     *compute_time = milliseconds / 1000.0;  // Convert to seconds
     
     // Measure device to host transfer time
     cudaEventRecord(start);
     cudaMemcpy(h_C, d_C, matrix_size, cudaMemcpyDeviceToHost);
     cudaEventRecord(stop);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&milliseconds, start, stop);
     *transfer_from_time = milliseconds / 1000.0;  // Convert to seconds
     
     // Clean up
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);
     cudaEventDestroy(start);
     cudaEventDestroy(stop);
 }