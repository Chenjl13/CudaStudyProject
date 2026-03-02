# Computation of the global thread index
## Grid(Dimension 1) + Block(Dimension1)
Click [here](./thread_model_1.cu) to quickly go through the code
And the [result](./thread_model_1_result.png) of above

## Calculation
### Grid(Dimension 1) + Block(Dimension1)
```
int id = blockIdx.x * blockDim.x + threadIdx.x;
```
### Grid(Dimension 1) + Block(Dimension2)
```
int id = blockIdx.x * blockDim.x * blockDim.y
         + threadIdx.y * blockDim.x
         + threadIdx.x;
```
### Grid(Dimension 1) + Block(Dimension3)
```
int id = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
         + threadIdx.z * blockDim.y * blockDim.x
         + threadIdx.y * blockDim.x
         + threadIdx.x;
```
### Grid(Dimension 2) + Block(Dimension1)
```
int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x
         + threadIdx.x;
```
### Grid(Dimension 2) + Block(Dimension2)
```
int id = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y)
         + threadIdx.y * blockDim.x
         + threadIdx.x;
```
### Grid(Dimension 2) + Block(Dimension3)
```
int id = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y * blockDim.z)
         + threadIdx.z * (blockDim.x * blockDim.y)
         + threadIdx.y * blockDim.x
         + threadIdx.x;
```
### Grid(Dimension 3) + Block(Dimension1)
```
int id = (blockIdx.x
         + blockIdx.y * gridDim.x
         + blockIdx.z * gridDim.x * gridDim.y) * blockDim.x
         + threadIdx.x;
```
### Grid(Dimension 3) + Block(Dimension2)
```
int id = (blockIdx.x
         + blockIdx.y * gridDim.x
         + blockIdx.z * gridDim.x * gridDim.y) * (blockDim.x * blockDim.y)
         + threadIdx.y * blockDim.x
         + threadIdx.x;
```
### Grid(Dimension 3) + Block(Dimension3)
```
int id = (blockIdx.x
         + blockIdx.y * gridDim.x
         + blockIdx.z * gridDim.x * gridDim.y) * (blockDim.x * blockDim.y * blockDim.z)
         + threadIdx.z * (blockDim.x * blockDim.y)
         + threadIdx.y * blockDim.x
         + threadIdx.x;
```



