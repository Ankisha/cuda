%%writefile warp_understand.cu
#include <stdio.h>
#include <cuda_runtime.h>
__global__ void warp_demonstrate(int* d_output){
  int tid = threadIdx.x + blockIdx.x*blockDim.x; // 128*2 threads have tids from 0 to 256
  int laneId = tid%32; // 0-31 for each thread in a warp
  int warpId = tid/32; // warp 0 to 3 within a block
  // Add warp ID information as an offset to distinguish between warps
  int warpOffset = (blockIdx.x * (blockDim.x / 32) + warpId) * 1000;
  d_output[tid] += warpOffset;
  // Demonstrate warp divergence with a simple if-else statement
  if (laneId < 16) {
      // First half of the warp executes this path
      d_output[tid] += 1;
  } else {
      // Second half of the warp executes this path
      d_output[tid] += 2;
  }


}

int main(){
  // threads configuration
  int numThreadsPerBlock = 128;
  int numBlocks = 2;
  int numElements = numThreadsPerBlock * numBlocks * 3; // multiplied with 3 to get some additional buffer space

  // allocate memory in host
  int* h_output = (int*)malloc(numElements*sizeof(int));
  // allocate device memory
  int* d_output;
  cudaMalloc((void**)&d_output,numElements*sizeof(int));

  warp_demonstrate<<<numBlocks, numThreadsPerBlock>>>(d_output);
  cudaDeviceSynchronize();

  cudaMemcpy(h_output, d_output, numElements * sizeof(int), cudaMemcpyDeviceToHost);


  // Print the first section (original values with warp offsets)
  printf("1. Original Values with Warp Offsets (after divergent execution):\n");
  printf("---------------------------------------------------------------\n");
  for (int b = 0; b < numBlocks; b++) {
      for (int w = 0; w < numThreadsPerBlock / 32; w++) {
          int warpBase = b * numThreadsPerBlock + w * 32;
          printf("Block %d, Warp %d (%d-%d): ", b, w, warpBase, warpBase + 31);
          
          for (int l = 0; l < 32; l++) {
              printf("%d ", h_output[warpBase + l]);
              
              // Print a separator in the middle of the warp to show the divergent paths
              if (l == 15) {
                  printf("| ");
              }
          }
          printf("\n");
      }
      printf("\n");
  }

}