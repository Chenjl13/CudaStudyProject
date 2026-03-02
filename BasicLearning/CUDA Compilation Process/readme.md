# CUDA Compilation Process (Two-Stage Model)

## Stage 1 – PTX Generation (Virtual Architecture)

The .cu source file is compiled by nvcc.

It generates PTX (Parallel Thread Execution) code.

PTX is a virtual intermediate representation.

It describes the required GPU functionality, independent of specific hardware.

Converting CUDA code to PTX improves portability.

## Stage 2 – Cubin Generation (Real Architecture)

The PTX code is compiled into CUBIN (binary code).

This binary is optimized for a specific GPU architecture (SM architecture).

The final binary is executed on the actual GPU hardware.
