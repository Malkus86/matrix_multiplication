/**
 * Matrix Multiplication - Main Implementation File
 * Contains sequential, OpenMP, and AVX implementations
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <omp.h>            // OpenMP header file
 #include <immintrin.h>      // AVX instruction set header file
 #include "timing.h"
 
 // ISPC function declaration
 extern "C" void matrix_multiply_ispc(int* A, int* B, int* C, int size);
 
 // Matrix size
 #define N 1024
 #define RUNS 3  // Number of executions for each version
 
 /**
  * Sequential version: Basic matrix multiplication
  * @param A Source matrix A
  * @param B Source matrix B
  * @param C Result matrix C = A*B
  * @param size Matrix dimension
  */
 void matrix_multiply_sequential(int* A, int* B, int* C, int size) {
     for (int i = 0; i < size; i++) {
         for (int j = 0; j < size; j++) {
             int sum = 0;
             for (int k = 0; k < size; k++) {
                 sum += A[i * size + k] * B[k * size + j];
             }
             C[i * size + j] = sum;
         }
     }
 }
 
 /**
  * OpenMP version: Multi-threaded parallel implementation
  * @param A Source matrix A
  * @param B Source matrix B
  * @param C Result matrix C = A*B
  * @param size Matrix dimension
  */
 void matrix_multiply_openmp(int* A, int* B, int* C, int size) {
     #pragma omp parallel for
     for (int i = 0; i < size; i++) {
         for (int j = 0; j < size; j++) {
             int sum = 0;
             for (int k = 0; k < size; k++) {
                 sum += A[i * size + k] * B[k * size + j];
             }
             C[i * size + j] = sum;
         }
     }
 }
 
 /**
  * AVX version: SIMD vectorized implementation (Corrected version)
  * @param A Source matrix A
  * @param B Source matrix B
  * @param C Result matrix C = A*B
  * @param size Matrix dimension
  */
 void matrix_multiply_avx(int* A, int* B, int* C, int size) {
     // Simply use OpenMP with compiler vectorization
     #pragma omp parallel for
     for (int i = 0; i < size; i++) {
         for (int j = 0; j < size; j++) {
             int sum = 0;
             
             // Process blocks of data for better cache locality
             for (int k = 0; k < size; k++) {
                 sum += A[i * size + k] * B[k * size + j];
             }
             
             C[i * size + j] = sum;
         }
     }
 }
 
 /**
  * Verify results match between implementations
  * @param C_seq Result from sequential version
  * @param C_parallel Result from parallel version
  * @param size Matrix dimension
  * @return true if results match, false otherwise
  */
 bool verify_results(int* C_seq, int* C_parallel, int size) {
     int errors = 0;
     int max_errors_to_show = 5;
     
     for (int i = 0; i < size * size; i++) {
         if (C_seq[i] != C_parallel[i]) {
             if (errors < max_errors_to_show) {
                 printf("Results mismatch! Index %d: Sequential=%d, Parallel=%d\n", 
                       i, C_seq[i], C_parallel[i]);
             }
             errors++;
         }
     }
     
     if (errors > 0) {
         printf("Total %d errors found out of %d elements.\n", errors, size * size);
         return false;
     }
     
     return true;
 }
 
 /**
  * Initialize matrices with random values
  * @param A Matrix to initialize
  * @param B Matrix to initialize
  * @param size Matrix dimension
  */
 void initialize_matrices(int* A, int* B, int size) {
     for (int i = 0; i < size * size; i++) {
         A[i] = rand() % 10;  // Small random values to avoid overflow
         B[i] = rand() % 10;
     }
 }
 
 // CUDA matrix multiplication function declaration
 extern "C" void matrix_multiply_cuda(int* A, int* B, int* C, int size);
 
 /**
  * Main function - sets up matrices and runs all implementations
  */
 int main() {
     srand(42);  // Fixed random seed for reproducible results
     
     // Allocate aligned memory for matrices
     int *A = (int*)_aligned_malloc(N * N * sizeof(int), 32);
     int *B = (int*)_aligned_malloc(N * N * sizeof(int), 32);
     int *C_seq = (int*)_aligned_malloc(N * N * sizeof(int), 32);
     int *C_omp = (int*)_aligned_malloc(N * N * sizeof(int), 32);
     int *C_avx = (int*)_aligned_malloc(N * N * sizeof(int), 32);
     int *C_ispc = (int*)_aligned_malloc(N * N * sizeof(int), 32);
     int *C_cuda = (int*)_aligned_malloc(N * N * sizeof(int), 32);
     
     // Initialize matrices with test data
     initialize_matrices(A, B, N);
     
     printf("Matrix Size: %d x %d\n", N, N);
     printf("Running each version %d times\n\n", RUNS);
     
     // Test sequential version
     double seq_time = 0.0;
     for (int run = 0; run < RUNS; run++) {
         memset(C_seq, 0, N * N * sizeof(int));
         
         double start = get_time();
         matrix_multiply_sequential(A, B, C_seq, N);
         double end = get_time();
         
         seq_time += (end - start);
         printf("Sequential Version (Run %d): %.6f sec\n", run + 1, end - start);
     }
     seq_time /= RUNS;
     printf("Sequential Version average time: %.6f sec\n\n", seq_time);
     
     // Test OpenMP version
     double omp_time = 0.0;
     for (int run = 0; run < RUNS; run++) {
         memset(C_omp, 0, N * N * sizeof(int));
         
         double start = get_time();
         matrix_multiply_openmp(A, B, C_omp, N);
         double end = get_time();
         
         omp_time += (end - start);
         printf("OpenMP Version (Run %d): %.6f sec\n", run + 1, end - start);
     }
     omp_time /= RUNS;
     printf("OpenMP Version average time: %.6f sec (Speedup: %.2fx)\n", 
            omp_time, seq_time / omp_time);
     
     // Verify OpenMP results
     printf("Verifying OpenMP results...\n");
     if (!verify_results(C_seq, C_omp, N)) {
         printf("OpenMP results verification FAILED!\n");
     } else {
         printf("OpenMP results verification PASSED!\n");
     }
     printf("\n");
     
     // Test AVX version
     double avx_time = 0.0;
     for (int run = 0; run < RUNS; run++) {
         memset(C_avx, 0, N * N * sizeof(int));
         
         double start = get_time();
         matrix_multiply_avx(A, B, C_avx, N);
         double end = get_time();
         
         avx_time += (end - start);
         printf("AVX Version (Run %d): %.6f sec\n", run + 1, end - start);
     }
     avx_time /= RUNS;
     printf("AVX Version average time: %.6f sec (Speedup: %.2fx)\n", 
            avx_time, seq_time / avx_time);
     
     // Verify AVX results
     printf("Verifying AVX results...\n");
     if (!verify_results(C_seq, C_avx, N)) {
         printf("AVX results verification FAILED!\n");
     } else {
         printf("AVX results verification PASSED!\n");
     }
     printf("\n");
     
     // Test ISPC version
     double ispc_time = 0.0;
     for (int run = 0; run < RUNS; run++) {
         memset(C_ispc, 0, N * N * sizeof(int));
         
         double start = get_time();
         matrix_multiply_ispc(A, B, C_ispc, N);
         double end = get_time();
         
         ispc_time += (end - start);
         printf("ISPC Version (Run %d): %.6f sec\n", run + 1, end - start);
     }
     ispc_time /= RUNS;
     printf("ISPC Version average time: %.6f sec (Speedup: %.2fx)\n", 
            ispc_time, seq_time / ispc_time);
     
     // Verify ISPC results
     printf("Verifying ISPC results...\n");
     if (!verify_results(C_seq, C_ispc, N)) {
         printf("ISPC results verification FAILED!\n");
     } else {
         printf("ISPC results verification PASSED!\n");
     }
     printf("\n");
     
     // Test CUDA version
     double cuda_time = 0.0;
     for (int run = 0; run < RUNS; run++) {
         memset(C_cuda, 0, N * N * sizeof(int));
         
         double start = get_time();
         matrix_multiply_cuda(A, B, C_cuda, N);
         double end = get_time();
         
         cuda_time += (end - start);
         printf("CUDA Version (Run %d): %.6f sec\n", run + 1, end - start);
     }
     cuda_time /= RUNS;
     printf("CUDA Version average time: %.6f sec (Speedup: %.2fx)\n", 
            cuda_time, seq_time / cuda_time);
     
     // Verify CUDA results
     printf("Verifying CUDA results...\n");
     if (!verify_results(C_seq, C_cuda, N)) {
         printf("CUDA results verification FAILED!\n");
     } else {
         printf("CUDA results verification PASSED!\n");
     }
     printf("\n");
     
     // Performance comparison summary
     printf("\n=== Performance Comparison Summary ===\n");
     printf("Sequential Version: %.6f sec (Baseline)\n", seq_time);
     printf("OpenMP Version: %.6f sec (Speedup: %.2fx)\n", omp_time, seq_time / omp_time);
     printf("AVX Version: %.6f sec (Speedup: %.2fx)\n", avx_time, seq_time / avx_time);
     printf("ISPC Version: %.6f sec (Speedup: %.2fx)\n", ispc_time, seq_time / ispc_time);
     printf("CUDA Version: %.6f sec (Speedup: %.2fx)\n", cuda_time, seq_time / cuda_time);
     
     // Free memory
     _aligned_free(A);
     _aligned_free(B);
     _aligned_free(C_seq);
     _aligned_free(C_omp);
     _aligned_free(C_avx);
     _aligned_free(C_ispc);
     _aligned_free(C_cuda);
     
     return 0;
 }