#include "solve.h"
#include <cuda_runtime.h>

// cuda kernel for vector addition
__global__ void vector_add(const float* d_A,const float* d_B,float* d_C,int N){
    // this code is run by each thread
    // 1. fetch the index of the thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){  // if there are more threads than elements to add then give work to only 
        d_C[i] = d_A[i] + d_B[i];
    }
}


void solve(const float* A, const float* B, float* C, int N){
    // declare device variables
    float* d_A;
    float* d_B;
    float* d_C;
    // allocate memory on device in global memory
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, N * sizeof(float));

    // copy input data from host to device
    cudaMemcpy(d_A,A,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B, N*sizeof(float),cudaMemcpyHostToDevice);

    // configure the kernel , calculate block and grid dimension
    int threadsPerBlock = 256; // usually a multiple of 32, 256 = 32 x 8
    int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock; // its N/threadsPerBlock but to get the integer value 

    // Launch the kernel
    vector_add<<<blocksPerGrid,threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // copy result back to host
    cudaMemcpy(C,d_C,N*sizeof(float),cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}