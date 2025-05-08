/**
 * Matrix multiplication implementation using ISPC
 * This file is compiled by the ISPC compiler to generate SIMD optimized code
 */

/**
 * Matrix multiplication function optimized with ISPC
 * @param A Source matrix A (1D array in row-major format)
 * @param B Source matrix B (1D array in row-major format)
 * @param C Result matrix C = A*B
 * @param size Matrix dimension
 */
export void matrix_multiply_ispc(uniform int* uniform A, 
                               uniform int* uniform B, 
                               uniform int* uniform C, 
                               uniform int size) {
    // Use blocks to improve cache locality
    uniform int BLOCK_SIZE = 32;
    
    // Parallelize across all elements using foreach
    foreach (i = 0 ... size, j = 0 ... size) {
        int sum = 0;
        
        // Process the matrix in blocks to improve cache utilization
        for (uniform int k_block = 0; k_block < size; k_block += BLOCK_SIZE) {
            uniform int k_end = min(k_block + BLOCK_SIZE, size);
            
            // Accumulate sum for the current block
            for (uniform int k = k_block; k < k_end; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
        }
        
        // Store the result
        C[i * size + j] = sum;
    }
}