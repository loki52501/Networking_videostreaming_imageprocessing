 #include <cuda_runtime.h>
 #include <iostream>

 __global__ void hello()
 {
    int tid=threadIdx.x+blockIdx.x*blockDim.x;
    printf("Hello world, tid: %d threadIdx: %d blockIdx: %d blockDim: %d\n",
           tid, threadIdx.x, blockIdx.x, blockDim.x);

 }
 int main()
 {
    hello<<<1,10>>>();
    cudaDeviceSynchronize();
    return 0;
 }