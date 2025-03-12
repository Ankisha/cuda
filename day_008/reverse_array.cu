%%writefile array_reverse.cu
/ CUDA kernel to reverse an array
__global__ void reverseArrayKernel(int* input, int* output, int size) {
    // Calculate the thread's global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if the thread is within array bounds
    if (idx < size) {
        // Reverse the array by writing elements in reverse order
        output[size - 1 - idx] = input[idx];
    }
}

// Host function to handle the array reversal
int main(int* h_input, int* h_output, int size) {
    // Allocate device memory
    int *d_input, *d_output;
    size_t bytes = size * sizeof(int);
    // Initialize input array
    for (int i = 0; i < SIZE; i++) {
        h_input[i] = i;
    }
    
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Calculate grid size (number of blocks)
    // Use 256 threads per block as a common choice
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch the kernel
    reverseArrayKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Copy the result back to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
