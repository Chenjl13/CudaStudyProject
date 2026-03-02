#include <stdio.h>

__global__ void hello_from_gpu()
{
    //grid+block
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int overall_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello World from GPU! Block %d, Thread %d, OverallID %d\n", bid, tid, overall_id);

}

int main(void)
{
    printf("Hello World from CPU\n");
    hello_from_gpu<<<2,2>>>();
    cudaDeviceSynchronize();
    return 0;
}