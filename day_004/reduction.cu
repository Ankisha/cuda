#define BLOCK_DIM 1024
#include <stdio.h>
#include <stdlib.h>

__global__ void reduce_kernel(float *input_d,float* partialSums, unsigned int N) {
  unsigned int segment = blockDim.x * blockIdx.x *2;
  unsigned int i = segment + threadIdx.x *2;
  for(int stride = 1; stride <=BLOCK_DIM ; stride *=2){
    if(threadIdx.x % stride ==0){
      input_d[i] += input_d[i+stride]
    }
    __syncthreads();
  }
  if(threadIdx.x==0){
    partialSums_d[..] = input_d[i]
  }
  
}


int main(float *input, int N) {
  // allocate memory
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  float *input_d;
  cudaMalloc((**void) &input_d, sizeof(float));

  // copy data to device
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaMemcpy(input_d,input,N*sizeof(float),cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  printf("Time taken for cudaMemcpy: %f ms\n", milliseconds);
  // Destroy events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);


  // kernel config
  const unsigned int numThreadsPerBlock = BLOCK_DIM;
  const unsigned int numElementsPerBlock = numThreadsPerBlock*2;
  const unsigned int numBlocks = (N+numThreadsPerBlock-1)/numThreadsPerBlock;

  // allocate partial sums
  float* partialSums = (float*)cudaMalloc(BLOCK_DIM*sizeof(float));
  float *partialSums_d;

  // call kernel
  reduce_kernel<<<numBlocks,numThreadsPerBlock>>>(input_d, partialSums_d, N);
  cudaDeviceSynchronize();

  // copy data from gpu
  cudaMemcpy(partialSums,partialSums_d,BLOCK_DIM*sizeof(float),cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Reduce parial sums on CPU
  float sum = 0.0f;
  for(int i = 0; i<BLOCK_DIM;++i){
    sum += partialSums[i];
  }
  printf(sum)

  // free memory
  cudaFree(input_d);
  cudaFree(partialSums_d);

}