#include <stdio.h>
#include "ErrorCheck.cuh"

int main()
{
    //SetGPU();

    float *fpHost;
    fpHost = (float*)malloc(4);
    memset(fpHost, 0, 4); //Initial to 0.

    float *fpDevice;
    cudaError_t error = ErrorCheck(cudaMalloc((float**)&fpDevice, 4), __FILE__, __LINE__);
    cudaMemset(fpDevice, 0, 4);

    /*The Error!!! */
    ErrorCheck(cudaMemcpy(fpDevice, fpHost, 4, cudaMemcpyDeviceToHost), __FILE__, __LINE__); 

    free(fpHost);
    
    ErrorCheck(cudaFree(fpDevice), __FILE__, __LINE__);

    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);

    return 0;
}
