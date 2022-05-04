# # computhon2022-1: Non-zero Fiber Counting

## Single-node, multiple-GPU vector addition

The CUDA code in `vecadd_mgpu.cu` will carry out vector addition on multipl GPUs. The slurm script `vecadd_mgpu.cu` dispatches a job to SLURM on TRUBA to compile and execute the code.
