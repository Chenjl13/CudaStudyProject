# Virtual architecture capabilities

<img src="a_result.png">

# Real architecture capabilities(CD must bigger than AB)
```
nvcc a.cu -o a_ABCD -arch=compute_AB -code=sm_CD
```

# Compile under multiple GPU version
```
nvcc a.cu -o a_fat -gencode -arch=compute_AB -code=sm_AB -gencode -arch=compute_CD -code=sm_CD -gencode -arch=compute_EF -code=sm_EF
```
