#include <stdio.h>
#include "TimeRecord.cuh"

__device__ float add(const float x, const float y)
{
    return x + y;
}

__global__ void addFromGPU(float *A, float *B, float *C, const int N)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + bid * blockDim.x; 

    if (id >= N) return;
    C[id] = add(A[id], B[id]);
    
}

void initialData(float *addr, int elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        addr[i] = (float)(rand() & 0xFF) / 10.f;
    }
    return;
}

int main(void)
{
    SetGPU();

    int iElemCount = 512;                     
    size_t stBytesCount = iElemCount * sizeof(float); 
    
    float *fpHost_A, *fpHost_B, *fpHost_C;
    fpHost_A = (float *)malloc(stBytesCount);
    fpHost_B = (float *)malloc(stBytesCount);
    fpHost_C = (float *)malloc(stBytesCount);
    if (fpHost_A != NULL && fpHost_B != NULL && fpHost_C != NULL)
    {
        memset(fpHost_A, 0, stBytesCount);  
        memset(fpHost_B, 0, stBytesCount);
        memset(fpHost_C, 0, stBytesCount);
    
    }
    else
    {
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }

    float *fpDevice_A, *fpDevice_B, *fpDevice_C;
    cudaMalloc((float**)&fpDevice_A, stBytesCount);
    cudaMalloc((float**)&fpDevice_B, stBytesCount);
    cudaMalloc((float**)&fpDevice_C, stBytesCount);
    if (fpDevice_A != NULL && fpDevice_B != NULL && fpDevice_C != NULL)
    {
        cudaMemset(fpDevice_A, 0, stBytesCount);  
        cudaMemset(fpDevice_B, 0, stBytesCount);
        cudaMemset(fpDevice_C, 0, stBytesCount);
    }
    else
    {
        printf("fail to allocate memory\n");
        free(fpHost_A);
        free(fpHost_B);
        free(fpHost_C);
        exit(-1);
    }

    srand(666); 
    initialData(fpHost_A, iElemCount);
    initialData(fpHost_B, iElemCount);
    
    cudaMemcpy(fpDevice_A, fpHost_A, stBytesCount, cudaMemcpyHostToDevice); 
    cudaMemcpy(fpDevice_B, fpHost_B, stBytesCount, cudaMemcpyHostToDevice); 
    cudaMemcpy(fpDevice_C, fpHost_C, stBytesCount, cudaMemcpyHostToDevice);

    dim3 block(2048);  //Only support the maximum of 1024, 2048 is an error!!!
    dim3 grid((iElemCount + block.x - 1) / 2048); 

    /*Time Record*/
    cudaEvent_t start, end;
    ErrorCheck(cudaEventCreate(&start), __FILE__, __LINE__);
    ErrorCheck(cudaEventCreate(&end), __FILE__, __LINE__);
    ErrorCheck(cudaEventRecord(start), __FILE__, __LINE__);
    cudaEventQuery(start);

    addFromGPU<<<grid, block>>>(fpDevice_A, fpDevice_B, fpDevice_C, iElemCount);    // Time Record sentences!!!

    ErrorCheck(cudaEventRecord(end), __FILE__, __LINE__);
    ErrorCheck(cudaEventSynchronize(end), __FILE__, __LINE__);
    float elapsed_time;
    
    ErrorCheck(cudaEventElapsedTime(&elapsed_time, start, end), __FILE__, __LINE__);
    printf("Time Record is %g ms\n", elapsed_time);

    ErrorCheck(cudaEventDestroy(start), __FILE__, __LINE__);
    ErrorCheck(cudaEventDestroy(end), __FILE__, __LINE__);


    cudaMemcpy(fpHost_C, fpDevice_C, stBytesCount, cudaMemcpyDeviceToHost);
  
    // for (int i = 0; i < 10; i++)    
    // {
    //     printf("idx=%2d\tmatrix_A:%.2f\tmatrix_B:%.2f\tresult=%.2f\n", i+1, fpHost_A[i], fpHost_B[i], fpHost_C[i]);
    // }

    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);
    cudaFree(fpDevice_A);
    cudaFree(fpDevice_B);
    cudaFree(fpDevice_C);

    cudaDeviceReset();
    return 0;
}

