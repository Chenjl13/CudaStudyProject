#include <stdio.h>

__global__ void hello_from_gpu_simpleDimension()
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int overall_id = threadIdx.x + blockDim.x * blockIdx.x;
    printf("hello world from GPU block %d,thread %d ,overall %d \n", bid, tid, overall_id);
}

int main(void)
{
    hello_from_gpu_simpleDimension<<<2,4>>>();
    cudaDeviceSynchronize();
    return 0;
}



