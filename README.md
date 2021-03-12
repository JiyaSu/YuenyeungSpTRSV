# YuenyeungSpTRSV
A Thread-Level and Warp-Level Fusion Synchronization-Free Sparse Triangular Solve on GPUs

## Introduction

This is the source code of our paper "YuenyeungSpTRSV: A Thread-Level and Warp-Level Fusion Synchronization-Free Sparse Triangular Solve" by Feng Zhang, Jiya Su, Weifeng Liu, Bingsheng He, Ruofan Wu, Xiaoyong Du, Rujia Wang, 2021.
YuenyeungSpTRSV is an extension of our previous work "CapelliniSpTRSV: A Thread-Level Synchronization-Free Sparse Triangular Solve on GPUs" (https://github.com/JiyaSu/CapelliniSpTRSV).

Our paper can be downloaded from XXX (XXXXX).

There are two versions of YYSpYRSV, namely CUDA and openCL.

## Abstract

Sparse triangular solves (SpTRSVs) are widely used in linear algebra domains, and several GPU-based SpTRSV algorithms have been developed. Synchronization-free SpTRSVs, due to their short preprocessing time and high performance, are currently the most popular SpTRSV algorithms. However, we observe that the performance of those SpTRSV algorithms on different matrices can vary greatly by 845 times. Our further studies show that when the average number of components per level is high and the average number of nonzero elements per row is low, those SpTRSVs exhibit extremely low performance. The reason is that, they use a warp on the GPU to process a row in sparse matrices, and such warp-level designs have severe underutilization of the GPU. To solve this problem, we propose YuenyeungSpTRSV, a thread-level and wrap-level fusion synchronization-free SpTRSV algorithm, which handles the rows with a large number of nonzero elements at warp-level while the rows with a low number of nonzero elements at thread-level. Particularly, YuenyeungSpTRSV has three novel features. First, unlike the previous studies, YuenyeungSpTRSV does not need long preprocessing time to calculate levels. Second, YuenyeungSpTRSV exhibits high performance on matrices that previous SpTRSVs cannot handle efficiently. Third, YuenyeungSpTRSV’s optimization does not rely on the specific sparse matrix storage format. Instead, it can achieve very good performance on the most popular sparse matrix storage, compressed sparse row (CSR) format, and thus users do not need to conduct format conversion. We evaluate YuenyeungSpTRSV with 245 matrices from the Florida Sparse Matrix Collection on four GPU platforms, and experiments show that our YuenyeungSpTRSV exhibits 7.14 GFLOPS/s, which is 5.98x speedup over the state-of-the-art synchronization-free SpTRSV algorithm, and 4.83x speedup over the SpTRSV in cuSPARSE.


## Execution

1. Choose the language you want to run, and enter the folder.
2. Adjust the common.h file according to the your hardware, the repeated times, and the accuracy of the calculation (single or double precision).
3. Set library path in the Makefile.
4. Run ``make``.
5. Run ``./YYSpTRSV example.mtx``. (kernel is in the YYSpTRSV.h or YYSpTRSV_kernel.cl)
6. The result will be printed out.

## Tested environments

CUDA:
1. nvidia GTX 1080 (Pascal) GPU in a host with CUDA v8.0 and Ubuntu 16.04.4 Linux installed.
2. nvidia Tesla V100 (Volta) GPU in a host with CUDA v9.0 and Ubuntu 16.04.1 Linux installed.
3. nvidia GeForce RTX 2080 Ti (Turing) GPU in a host with CUDA v10.2 and Ubuntu 18.04.4 Linux installed.

OpenCL:
1. Radeon Vega 11 APU with ROCm compiler and Ubuntu 18.04.3 Linux installed.

## Acknowledgement

YuenyeungSpTRSV is developed by Renmin University of China, China University of Petroleum, National University of Singapore, and Illinois Institute of Technology.

Feng Zhang, Ruofan Wu and Xiaoyong Du are with the Key Laboratory of Data Engineering and Knowledge Engineering (MOE), and School of Information, Renmin University of China.

Weifeng Liu is with the Super Scientific Software Laboratory, Department of Computer Science and Technology, China University of Petroleum.

Bingsheng He is with the School of Computing, National University of Singapore.

Jiya Su and Rujia Wang is with the Computer Science Department, Illinois Institute of Technology.

If you have any questions, please contact us (Jiya_Su@ruc.edu.cn or jsu18@hawk.iit.edu).

## Citation

If you use our code, please cite our paper:
```
XXXX
```

