#ifndef _YYSpTRSV_
#define _YYSpTRSV_

#include "common.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>


/*  yySpTRSV kernel  */

__global__
void yySpTRSV_csr_kernel(const int* __restrict__        d_csrRowPtr,
                         const int* __restrict__        d_csrColIdx,
                         const VALUE_TYPE* __restrict__ d_csrVal,
                         int*                           d_get_value,
                         const int                      m,
                         const int                      nnz,
                         const VALUE_TYPE* __restrict__ d_b,
                         VALUE_TYPE*                    d_x,
                         const int                      begin,
                         const int* __restrict__        d_warp_num,
                         const int                      Len,
                         int*                           d_id_extractor)

{
    const int global_id =atomicAdd(d_id_extractor, 1);
//    const int global_id = (begin + blockIdx.x) * blockDim.x + threadIdx.x;

    const int warp_id = global_id/WARP_SIZE;
    int row;
    
    if(warp_id>=(Len-1))
        return;
    
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    
    /* Thread-level Syncfree SpTRSV */
    if(d_warp_num[warp_id+1]>(d_warp_num[warp_id]+1))
    {
        row =d_warp_num[warp_id]+lane_id;
        if(row>=m)
            return;
        
        int col,j,i;
        VALUE_TYPE xi;
        VALUE_TYPE left_sum=0;
        i=row;
        j=d_csrRowPtr[i];
        
        while(j<d_csrRowPtr[i+1])
        {
            col=d_csrColIdx[j];
            while(d_get_value[col]==1)
            //if(d_get_value[col]==1)
            {
                left_sum+=d_csrVal[j]*d_x[col];
                j++;
                col=d_csrColIdx[j];
            }
            if(i==col)
            {
                xi = (d_b[i] - left_sum) / d_csrVal[d_csrRowPtr[i+1]-1];
                d_x[i] = xi;
                __threadfence();
                d_get_value[i]=1;
                j++;
            }
        }
    }
    else  /* Warp-level Syncfree SpTRSV */
    {
        row = d_warp_num[warp_id];
        if(row>=m)
            return;
        
        int col,j;
        VALUE_TYPE xi,sum=0;
        for (j = d_csrRowPtr[row]  + lane_id; j < d_csrRowPtr[row+1]-1; j += WARP_SIZE)
        {
            
            col=d_csrColIdx[j];
            while(d_get_value[col]!=1)
            {
                __threadfence_block();
            }
            sum += d_x[col] * d_csrVal[j];
        }
        
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            sum += __shfl_down(sum, offset);
        }
        
        if (!lane_id)   /* thread 0 in each warp*/
        {
            xi = (d_b[row] - sum) / d_csrVal[d_csrRowPtr[row+1]-1];
            d_x[row]=xi;
            __threadfence();
            d_get_value[row]=1;
        }
        
    }
}
    
    
    
    
    
    
    

int YYSpTRSV_csr(const int            m,
                 const int            n,
                 const int            nnzL,
                 const int           *csrRowPtrL_tmp,
                 const int           *csrColIdxL_tmp,
                 const VALUE_TYPE    *csrValL_tmp,
                 const VALUE_TYPE    *b,
                 VALUE_TYPE          *x,
                 const int            border,
                 double              *solve_time_add,
                 double              *gflops_add,
                 double              *bandwith_add,
                 double              *pre_time_add,
                 double              *warp_occupy_add,
                 double              *element_occupy_add)
{
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }
    
    
    int i;
    
    /* preprocessing */
    int Len;
    int *warp_num=(int *)malloc((m+1)*sizeof(int));
    if (warp_num==NULL)
        printf("warp_num error\n");
    memset (warp_num, 0, sizeof(int)*(m+1));
    
    double warp_occupy=0,element_occupy=0;
    struct timeval t1, t2;
    double time_cuda_pre = 0;
    
    for(i=0;i<BENCH_REPEAT;i++)
    {
        gettimeofday(&t1, NULL);
        matrix_warp(m,n,nnzL,csrRowPtrL_tmp,csrColIdxL_tmp,csrValL_tmp,10,&Len,warp_num,&warp_occupy,&element_occupy);
        gettimeofday(&t2, NULL);
        time_cuda_pre += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }
    
    time_cuda_pre/=BENCH_REPEAT;
    *pre_time_add=time_cuda_pre;
    
    *warp_occupy_add=warp_occupy;
    *element_occupy_add=element_occupy;
    
    
    /* transfer host mem to device mem */
    int *d_csrRowPtrL;
    int *d_csrColIdx;
    
    VALUE_TYPE *d_csrValL;
    VALUE_TYPE *d_b;
    VALUE_TYPE *d_x;
    
    
    // Matrix L
    cudaMalloc((void **)&d_csrRowPtrL, (m+1) * sizeof(int));
    cudaMalloc((void **)&d_csrColIdx, nnzL  * sizeof(int));
    cudaMalloc((void **)&d_csrValL,    nnzL  * sizeof(VALUE_TYPE));
    
    cudaMemcpy(d_csrRowPtrL, csrRowPtrL_tmp, (m+1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdx, csrColIdxL_tmp, nnzL  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrValL,    csrValL_tmp,    nnzL  * sizeof(VALUE_TYPE),   cudaMemcpyHostToDevice);
    
    // Vector b
    cudaMalloc((void **)&d_b, m * sizeof(VALUE_TYPE));
    cudaMemcpy(d_b, b, m * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
    
    // Vector x
    cudaMalloc((void **)&d_x, n  * sizeof(VALUE_TYPE));
    cudaMemset(d_x, 0, n * sizeof(VALUE_TYPE));
    
    //get_value
    int *d_get_value;
    int *get_value = (int *)malloc(m * sizeof(int));
    memset(get_value, 0, m * sizeof(int));
    cudaMalloc((void **)&d_get_value, (m) * sizeof(int));
    cudaMemcpy(d_get_value, get_value, (m) * sizeof(int),   cudaMemcpyHostToDevice);
    
    //warp_num
    int *d_warp_num;
    cudaMalloc((void **)&d_warp_num, Len  * sizeof(int));
    cudaMemcpy(d_warp_num, warp_num, Len * sizeof(int), cudaMemcpyHostToDevice);
    
    
    /* solve Lx=b */
    double time_cuda_solve = 0;
    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int num_blocks = ceil ((double)((Len-1)*WARP_SIZE) / (double)(num_threads));
    
    int *d_id_extractor;
    cudaMalloc((void **)&d_id_extractor, sizeof(int));
    
    for (i = 0; i < BENCH_REPEAT; i++)
    {
        
        cudaMemset(d_get_value, 0, sizeof(int) * m);
        cudaMemset(d_x, 0, sizeof(VALUE_TYPE) * m);
        cudaMemset(d_id_extractor, 0, sizeof(int));
        
        gettimeofday(&t1, NULL);
        
        yySpTRSV_csr_kernel<<< num_blocks, num_threads >>> (d_csrRowPtrL, d_csrColIdx, d_csrValL, d_get_value,m, nnzL, d_b, d_x ,0 ,d_warp_num,Len,d_id_extractor);
        cudaDeviceSynchronize();
        
        gettimeofday(&t2, NULL);
        
        time_cuda_solve += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        
        cudaMemcpy(x, d_x, n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
        
    }
    
    
    time_cuda_solve /= BENCH_REPEAT;
    
    double dataSize = (double)((n+1)*sizeof(int) + (nnzL+m)*sizeof(int) + nnzL*sizeof(VALUE_TYPE) + 2*n*sizeof(VALUE_TYPE)+ Len * sizeof(int));
    
    *solve_time_add=time_cuda_solve;
    *gflops_add=2*nnzL/(1e6*time_cuda_solve);
    *bandwith_add=dataSize/(1e6*time_cuda_solve);
    
    

    cudaFree(d_csrRowPtrL);
    cudaFree(d_csrColIdx);
    cudaFree(d_csrValL);
    cudaFree(d_get_value);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_warp_num);
    cudaFree(d_id_extractor);
    
    free(get_value);
    free(warp_num);
    
    return 0;
}

#endif

