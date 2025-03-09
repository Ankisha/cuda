#include <stdio.h>
#include <cuda_runtime.h>


// Naive matrix transpose kernel
__global__ void naiveMatrixTranspose(float *input, float *output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        // Input is in row-major order, so input[row * cols + col] accesses input[row][col]
        // For transpose, we write to output[col * rows + row] which is output[col][row]
        output[col * rows + row] = input[row * cols + col];
    }
}

int main() {
    // Matrix dimensions
    int rows = 1024;
    int cols = 1024;
    size_t matrix_size = rows * cols * sizeof(float);
    
    // Host matrices
    float *h_input = (float*)malloc(matrix_size);
    float *h_output = (float*)malloc(matrix_size);
    
    // Initialize input matrix with some values
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            h_input[i * cols + j] = i * cols + j; // Just a simple initialization pattern
        }
    }
    
    // Device matrices
    float *d_input, *d_output;
    cudaMalloc(&d_input, matrix_size);
    cudaMalloc(&d_output, matrix_size);
    
    // Copy input matrix from host to device
    cudaMemcpy(d_input, h_input, matrix_size, cudaMemcpyHostToDevice);
    
    // Set up grid and block dimensions
    dim3 blockDim(16, 16);  // 16x16 threads per block
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, 
                (rows + blockDim.y - 1) / blockDim.y);
    
    // Launch the transpose kernel
    naiveMatrixTranspose<<<gridDim, blockDim>>>(d_input, d_output, rows, cols);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Copy the result back to host
    cudaMemcpy(h_output, d_output, matrix_size, cudaMemcpyDeviceToHost);
    
    // Free memory
    free(h_input);
    free(h_output);    
    return 0;
}