#include <stdio.h>
#include "setDevice.cuh"

/*
A problem: If ElementCount can't be divided by 32 
We need to define "dim3 grid((ElementCount + block.x - 1) / 32);", to ensure we have sufficient grid
Then, we need to judge if id >= N, to avoid the waste of id
*/

__device__ float add(float x, float y)  //device function, which can be called in kernal function!
{
    return x + y;
}

__global__ void AddFromGPU(float *A, float *B, float *C, const int N)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + bid * blockDim.x;
    if(id >= N) return;
    C[id] = add(A[id], B[id]);

}

void InitialData(float *addr, int ElemCount)
{ 
    for(int i = 0; i < ElemCount; i++)
    {
        addr[i] = (float)(rand() & 0xff) / 10.f;
    }    
}

int main(void)
{
    // Initialize GPU
    SetGPU();

    //allocate host memory and device memory
    int ElementCount = 513;
    size_t ByteCount = ElementCount * sizeof(float);

    /* host memory */
    float *fpHost_A, *fpHost_B, *fpHost_C;
    fpHost_A = (float*)malloc(ByteCount);
    fpHost_B = (float*)malloc(ByteCount);
    fpHost_C = (float*)malloc(ByteCount);
    if(fpHost_A != NULL && fpHost_B != NULL && fpHost_C != NULL)
    { 
        //Initialize Memory to 0. 
        memset(fpHost_A, 0, ByteCount);
        memset(fpHost_B, 0, ByteCount);
        memset(fpHost_C, 0, ByteCount);
    }
    else
    {
        printf("Fail to allocate Memory!\n");
        exit(-1);
    }

    /* device memory */
    float *fpDevice_A, *fpDevice_B, *fpDevice_C;
    cudaMalloc((float**)&fpDevice_A, ByteCount);
    cudaMalloc((float**)&fpDevice_B, ByteCount);
    cudaMalloc((float**)&fpDevice_C, ByteCount);
    if(fpDevice_A != NULL && fpDevice_B != NULL && fpDevice_C != NULL)
    {
        cudaMemset(fpDevice_A, 0, ByteCount);
        cudaMemset(fpDevice_B, 0, ByteCount);
        cudaMemset(fpDevice_C, 0, ByteCount);
    }
    else
    {
        printf("fail to allocate device memory!\n");
        free(fpDevice_A);
        free(fpDevice_B);
        free(fpDevice_C);
        exit(-1);
    }

    /* Initial Data in host*/
    srand(666);
    InitialData(fpHost_A, ElementCount);
    InitialData(fpHost_B, ElementCount);

    /* Copy Data from host to Device*/
    cudaMemcpy(fpDevice_A, fpHost_A, ByteCount, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_B, fpHost_B, ByteCount, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_C, fpHost_C, ByteCount, cudaMemcpyHostToDevice);

    /*Use Kernal Function*/
    dim3 block(32);
    //dim3 grid(ElementCount / 32);
    dim3 grid((ElementCount + block.x - 1) / 32);

    AddFromGPU <<<block, grid>>>(fpDevice_A, fpDevice_B, fpDevice_C, ByteCount);
    cudaDeviceSynchronize();

    /*Send result back to host*/
    cudaMemcpy(fpHost_C, fpDevice_C, ByteCount, cudaMemcpyDeviceToHost);

    /*Print first 10th result*/
    for(int i = 0; i < 10; i++)
    {
        printf("The idx %d: matric %.2f + matric %.2f = matric %.2f\n", i+1, fpHost_A[i], fpHost_B[i], fpHost_C[i]);
    }

    /*Free Memory*/
    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);
    cudaFree(fpDevice_A);
    cudaFree(fpDevice_B);
    cudaFree(fpDevice_C);
    cudaDeviceReset();

    return 0;
}

