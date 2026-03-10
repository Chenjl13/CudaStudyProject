#include <stdio.h>

cudaError_t ErrorCheck(cudaError_t error_code, const char* file, int line)
{
    if(error_code != cudaSuccess)
    {
        printf("CUDA error:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, line%d\r\n", error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), 
        file, line);
        return error_code;
    }

    else
        return error_code;
}

void SetGPU()
{
    // Check the amount of GPU in the computer
    int iDeviceCount = 0;
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&iDeviceCount), __FILE__, __LINE__);

    // if(error != cudaSuccess || iDeviceCount ==0)
    // {
    //     printf("No Cuda Device Found");
    //     exit(-1);
    // }
    // else
    // {
    //     printf("The amount of GPU is %d\n", iDeviceCount);
    // }


    // Set to work
    int iDev = 0;
    error = ErrorCheck(cudaSetDevice(iDev), __FILE__, __LINE__);
    // if(error != cudaSuccess)
    // {
    //     printf("fail to set GPU computing!\n");
    //     exit(-1);
    // }
    // else
    // {
    //     printf("Set GPU 0 for computing!\n");
    // }
}



