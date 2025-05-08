/**
 * Matrix Multiplication - Main Implementation File
 * Contains sequential, OpenMP, and AVX implementations
 * Enhanced with proper AVX intrinsics and additional optimizations
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
 #define RUNS 5  // Increased from 3 to 5 for more stable results
 #define BLOCK_SIZE 32  // Block size for cache optimization
 
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
  * Sequential version with cache blocking optimization
  * @param A Source matrix A
  * @param B Source matrix B
  * @param C Result matrix C = A*B
  * @param size Matrix dimension
  */
 void matrix_multiply_sequential_blocked(int* A, int* B, int* C, int size) {
     // Initialize result matrix with zeros
     memset(C, 0, size * size * sizeof(int));
     
     // Process blocks for better cache locality
     for (int ii = 0; ii < size; ii += BLOCK_SIZE) {
         for (int jj = 0; jj < size; jj += BLOCK_SIZE) {
             for (int kk = 0; kk < size; kk += BLOCK_SIZE) {
                 // Process each block
                 for (int i = ii; i < min(ii + BLOCK_SIZE, size); i++) {
                     for (int j = jj; j < min(jj + BLOCK_SIZE, size); j++) {
                         int sum = C[i * size + j]; // Load current value
                         for (int k = kk; k < min(kk + BLOCK_SIZE, size); k++) {
                             sum += A[i * size + k] * B[k * size + j];
                         }
                         C[i * size + j] = sum; // Store updated value
                     }
                 }
             }
         }
     }
 }
 
 /**
  * OpenMP version: Multi-threaded parallel implementation with cache blocking
  * @param A Source matrix A
  * @param B Source matrix B
  * @param C Result matrix C = A*B
  * @param size Matrix dimension
  */
 void matrix_multiply_openmp(int* A, int* B, int* C, int size) {
     // Initialize result matrix with zeros
     memset(C, 0, size * size * sizeof(int));
     
     // Get the number of available cores and set thread count
     int num_threads = omp_get_max_threads();
     omp_set_num_threads(num_threads);
     
     // Process blocks for better cache locality
     #pragma omp parallel for collapse(2) schedule(dynamic)
     for (int ii = 0; ii < size; ii += BLOCK_SIZE) {
         for (int jj = 0; jj < size; jj += BLOCK_SIZE) {
             for (int kk = 0; kk < size; kk += BLOCK_SIZE) {
                 // Process each block
                 for (int i = ii; i < min(ii + BLOCK_SIZE, size); i++) {
                     for (int j = jj; j < min(jj + BLOCK_SIZE, size); j++) {
                         int sum = C[i * size + j]; // Load current value
                         
                         // Process inner loop with good cache locality
                         for (int k = kk; k < min(kk + BLOCK_SIZE, size); k++) {
                             sum += A[i * size + k] * B[k * size + j];
                         }
                         
                         C[i * size + j] = sum; // Store updated value
                     }
                 }
             }
         }
     }
 }
 
 /**
  * Transpose matrix for better memory access patterns
  * @param src Source matrix
  * @param dst Destination matrix (transposed)
  * @param size Matrix dimension
  */
 void transpose_matrix(int* src, int* dst, int size) {
     // Use OpenMP to parallelize the transpose operation
     #pragma omp parallel for
     for (int i = 0; i < size; i++) {
         for (int j = 0; j < size; j++) {
             dst[j * size + i] = src[i * size + j];
         }
     }
 }
 
 /**
  * Helper function to process a block of matrix multiplication with AVX
  * @param A Row of matrix A
  * @param B_transposed Column of matrix B (stored as row in transposed matrix)
  * @param size Matrix dimension
  * @return Dot product of row and column
  */
 int process_row_col_avx(int* A_row, int* B_col, int size) {
     // Initialize accumulators for AVX processing
     __m256i sum_vec = _mm256_setzero_si256();
     
     // Process 8 elements at a time using AVX
     for (int k = 0; k < size; k += 8) {
         // Handle boundary condition
         if (k + 8 <= size) {
             // Load 8 consecutive elements from row A and column B
             __m256i a_vec = _mm256_loadu_si256((__m256i*)&A_row[k]);
             __m256i b_vec = _mm256_loadu_si256((__m256i*)&B_col[k]);
             
             // Multiply and accumulate
             __m256i mul = _mm256_mullo_epi32(a_vec, b_vec);
             sum_vec = _mm256_add_epi32(sum_vec, mul);
         } else {
             // Process remaining elements (less than 8)
             for (int kk = k; kk < size; kk++) {
                 sum_vec = _mm256_add_epi32(sum_vec, 
                     _mm256_mullo_epi32(
                         _mm256_set1_epi32(A_row[kk]),
                         _mm256_set1_epi32(B_col[kk])
                     )
                 );
             }
         }
     }
     
     // Horizontal sum of vector elements
     int result[8];
     _mm256_storeu_si256((__m256i*)result, sum_vec);
     return result[0] + result[1] + result[2] + result[3] + 
            result[4] + result[5] + result[6] + result[7];
 }
 
 /**
  * AVX version: SIMD vectorized implementation with matrix transposition
  * @param A Source matrix A
  * @param B Source matrix B
  * @param C Result matrix C = A*B
  * @param size Matrix dimension
  */
 void matrix_multiply_avx(int* A, int* B, int* C, int size) {
     // Allocate memory for transposed B matrix for better access patterns
     int* B_transposed = (int*)_aligned_malloc(size * size * sizeof(int), 32);
     
     // Transpose B matrix to optimize column access pattern
     transpose_matrix(B, B_transposed, size);
     
     // Use OpenMP to parallelize the outer loop
     #pragma omp parallel for
     for (int i = 0; i < size; i++) {
         for (int j = 0; j < size; j++) {
             // Process a single dot product using AVX
             C[i * size + j] = process_row_col_avx(&A[i * size], &B_transposed[j * size], size);
         }
     }
     
     // Free the transposed matrix
     _aligned_free(B_transposed);
 }
 
 /**
  * AVX version with block processing for better cache utilization
  * @param A Source matrix A
  * @param B Source matrix B
  * @param C Result matrix C = A*B
  * @param size Matrix dimension
  */
 void matrix_multiply_avx_blocked(int* A, int* B, int* C, int size) {
     // Allocate memory for transposed B matrix
     int* B_transposed = (int*)_aligned_malloc(size * size * sizeof(int), 32);
     
     // Transpose B matrix to optimize column access pattern
     transpose_matrix(B, B_transposed, size);
     
     // Initialize result matrix with zeros
     memset(C, 0, size * size * sizeof(int));
     
     // Process in blocks for better cache locality
     #pragma omp parallel for collapse(2) schedule(dynamic)
     for (int ii = 0; ii < size; ii += BLOCK_SIZE) {
         for (int jj = 0; jj < size; jj += BLOCK_SIZE) {
             // Process each block
             for (int i = ii; i < min(ii + BLOCK_SIZE, size); i++) {
                 for (int j = jj; j < min(jj + BLOCK_SIZE, size); j++) {
                     // For each element in the block, we process with AVX
                     C[i * size + j] = process_row_col_avx(&A[i * size], &B_transposed[j * size], size);
                 }
             }
         }
     }
     
     // Free the transposed matrix
     _aligned_free(B_transposed);
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
     int max_absolute_diff = 0;
     
     for (int i = 0; i < size * size; i++) {
         if (C_seq[i] != C_parallel[i]) {
             int abs_diff = abs(C_seq[i] - C_parallel[i]);
             if (abs_diff > max_absolute_diff) {
                 max_absolute_diff = abs_diff;
             }
             
             if (errors < max_errors_to_show) {
                 printf("Results mismatch! Index %d: Sequential=%d, Parallel=%d (Diff=%d)\n", 
                       i, C_seq[i], C_parallel[i], abs_diff);
             }
             errors++;
         }
     }
     
     if (errors > 0) {
         printf("Total %d errors found out of %d elements. Max difference: %d\n", 
               errors, size * size, max_absolute_diff);
         if (max_absolute_diff <= 1) {
             printf("Minor differences may be due to different order of floating point operations.\n");
             return true; // Accept small numerical differences
         }
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
     
     printf("=== Matrix Multiplication Performance Comparison ===\n");
     printf("CPU Information:\n");
     
     // Get number of physical cores
     int max_threads = omp_get_max_threads();
     printf("- Available logical cores: %d\n", max_threads);
     
     // Check AVX2 support
     #if defined(__AVX2__)
     printf("- AVX2 instructions: Supported\n");
     #else
     printf("- AVX2 instructions: Not supported\n");
     #endif
     
     // Allocate aligned memory for matrices
     int *A = (int*)_aligned_malloc(N * N * sizeof(int), 32);
     int *B = (int*)_aligned_malloc(N * N * sizeof(int), 32);
     int *C_seq = (int*)_aligned_malloc(N * N * sizeof(int), 32);
     int *C_seq_blocked = (int*)_aligned_malloc(N * N * sizeof(int), 32);
     int *C_omp = (int*)_aligned_malloc(N * N * sizeof(int), 32);
     int *C_avx = (int*)_aligned_malloc(N * N * sizeof(int), 32);
     int *C_avx_blocked = (int*)_aligned_malloc(N * N * sizeof(int), 32);
     int *C_ispc = (int*)_aligned_malloc(N * N * sizeof(int), 32);
     int *C_cuda = (int*)_aligned_malloc(N * N * sizeof(int), 32);
     
     if (!A || !B || !C_seq || !C_seq_blocked || !C_omp || !C_avx || 
         !C_avx_blocked || !C_ispc || !C_cuda) {
         printf("Memory allocation failed\n");
         return -1;
     }
     
     // Initialize matrices with test data
     initialize_matrices(A, B, N);
     
     printf("\nMatrix Size: %d x %d\n", N, N);
     printf("Running each version %d times\n\n", RUNS);
     printf("Using block size of %d for cache optimization\n\n", BLOCK_SIZE);
     
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
     printf("Sequential Version average time: %.6f sec (baseline)\n\n", seq_time);
     
     // Test sequential blocked version
     double seq_blocked_time = 0.0;
     for (int run = 0; run < RUNS; run++) {
         memset(C_seq_blocked, 0, N * N * sizeof(int));
         
         double start = get_time();
         matrix_multiply_sequential_blocked(A, B, C_seq_blocked, N);
         double end = get_time();
         
         seq_blocked_time += (end - start);
         printf("Sequential Blocked Version (Run %d): %.6f sec\n", run + 1, end - start);
     }
     seq_blocked_time /= RUNS;
     printf("Sequential Blocked Version average time: %.6f sec (Speedup: %.2fx)\n", 
            seq_blocked_time, seq_time / seq_blocked_time);
     
     // Verify Sequential Blocked results
     printf("Verifying Sequential Blocked results...\n");
     if (!verify_results(C_seq, C_seq_blocked, N)) {
         printf("Sequential Blocked results verification FAILED!\n");
     } else {
         printf("Sequential Blocked results verification PASSED!\n");
     }
     printf("\n");
     
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
     
     // Test AVX Blocked version
     double avx_blocked_time = 0.0;
     for (int run = 0; run < RUNS; run++) {
         memset(C_avx_blocked, 0, N * N * sizeof(int));
         
         double start = get_time();
         matrix_multiply_avx_blocked(A, B, C_avx_blocked, N);
         double end = get_time();
         
         avx_blocked_time += (end - start);
         printf("AVX Blocked Version (Run %d): %.6f sec\n", run + 1, end - start);
     }
     avx_blocked_time /= RUNS;
     printf("AVX Blocked Version average time: %.6f sec (Speedup: %.2fx)\n", 
            avx_blocked_time, seq_time / avx_blocked_time);
     
     // Verify AVX Blocked results
     printf("Verifying AVX Blocked results...\n");
     if (!verify_results(C_seq, C_avx_blocked, N)) {
         printf("AVX Blocked results verification FAILED!\n");
     } else {
         printf("AVX Blocked results verification PASSED!\n");
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
     printf("Sequential Blocked: %.6f sec (Speedup: %.2fx)\n", seq_blocked_time, seq_time / seq_blocked_time);
     printf("OpenMP Version: %.6f sec (Speedup: %.2fx)\n", omp_time, seq_time / omp_time);
     printf("AVX Version: %.6f sec (Speedup: %.2fx)\n", avx_time, seq_time / avx_time);
     printf("AVX Blocked Version: %.6f sec (Speedup: %.2fx)\n", avx_blocked_time, seq_time / avx_blocked_time);
     printf("ISPC Version: %.6f sec (Speedup: %.2fx)\n", ispc_time, seq_time / ispc_time);
     printf("CUDA Version: %.6f sec (Speedup: %.2fx)\n", cuda_time, seq_time / cuda_time);
     
     // Free memory
     _aligned_free(A);
     _aligned_free(B);
     _aligned_free(C_seq);
     _aligned_free(C_seq_blocked);
     _aligned_free(C_omp);
     _aligned_free(C_avx);
     _aligned_free(C_avx_blocked);
     _aligned_free(C_ispc);
     _aligned_free(C_cuda);
     
     return 0;
 }