#include <stdio.h>

void SetGPU()
{
    // Check the amount of GPU in the computer
    int iDeviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&iDeviceCount);

    if(error != cudaSuccess || iDeviceCount ==0)
    {
        printf("No Cuda Device Found");
        exit(-1);
    }
    else
    {
        printf("The amount of GPU is %d\n", iDeviceCount);
    }


    // Set to work
    int iDev = 0;
    error = cudaSetDevice(iDev);
    if(error != cudaSuccess)
    {
        printf("fail to set GPU computing!\n");
        exit(-1);
    }
    else
    {
        printf("Set GPU 0 for computing!\n");
    }
}



