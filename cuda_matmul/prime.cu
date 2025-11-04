 #include <cuda_runtime.h>
 #include <iostream>
__device__ unsigned long long counter = 0;
 __global__ void isPrime(long long start, long long end)
 {
    int tid=threadIdx.x+blockIdx.x*blockDim.x;
    long long num=start+tid;
    if(num>end) return ;
    if(num<=0) return ;
    if(num==2)return ;
    if(num%2==0) return ;
    for(long long i=3;i*i<=num;i+=2){
        if(num%i==0){
            return ;
        }
    }
    printf("%lld ",num);
    atomicAdd(&counter,1);
    return ;
    
 }
 int main()
 {std::cout<<" hi there\n";
    int threads = 256;
 int odds    = (100000 - 1001) / 2 + 1;
 int blocks  = (odds + threads - 1) / threads;
 isPrime<<<999, 1000>>>(1000, 1000000);
    cudaDeviceSynchronize();
long long hostCounter = 0;
        cudaMemcpyFromSymbol(&hostCounter, counter, sizeof(hostCounter));
        std::cout << "\n" << hostCounter << "\n";
            return 0;
 }