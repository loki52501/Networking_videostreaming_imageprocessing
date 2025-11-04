 #include <cuda_runtime.h>
 #include <iostream>
 using namespace std;
 __global__ void hello()
 {
    int tid=threadIdx.x+blockIdx.x*blockDim.x;
    cout<<"Hello world, tid: "<<tid<<" threadIdx: "<<threadIdx.x<<" blockIdx: "<<blockIdx.x<<" blockDim: "<<blockDim.x<<"\n";
 }
 int main()
 {
    hello<<<1,10>>>();
    cudaDeviceSynchronize();
    return 0;
 }