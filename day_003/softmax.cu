#include "solve.h"
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float *d_input,float *d_output, int N){
    // step 1 : max trick
    // find the max row value and store in shared memory so that all threads in the block can access that
    __shared__ float max_val;
    // should one thread perform max calculation or multiple? multiple threads there is a pb 
    if(threadIdx.x == 0){
        max_val = -INFINITY;
        // loop over input to find max
        for(int i =0; i<N; ++i){
            max_val = fmaxf(max_val,d_input[i]);
        }
        __syncthreads(); // unless this is finished threads should wait because max val calculation should be finished before next step
        // can multiple threads compute the max in parallel if N is large
    }
    // step2 : compute the exponent at each index and simultaneously get it added to a sum
    __shared__ float sum_exp;
    if(threadIdx.x ==0){
        sum_exp = 0.0f;
    }
    __syncthreads(); // all threads see the initialized sum_exp value
    // now each thread performs exponent operation
    int i = threadIdx.x;
    float exp_val = expf(d_input[i]-max_val);
    // add exp_val calculated by each thread to a shared memory variable
    // each thread would try to calculate and update the sum so clashes possible between threads
    // So use atomic sum, only one thread updates at a time
    atomicAdd(&sum_exp,exp_val);
    __syncthreads(); // unless all threads have completed sum_exp won't be final so this is imp
    d_output[i] = exp_val/sum_exp;
}

void solve(const float* input, float* output, int N) {
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}