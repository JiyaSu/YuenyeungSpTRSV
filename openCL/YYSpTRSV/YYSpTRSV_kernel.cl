#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#define WARP_SIZE 64

#ifndef WARP_PER_BLOCK
#define WARP_PER_BLOCK   4
#endif



__kernel
void YYSpTRSV_csr_kernel(__global const int            *d_csrRowPtr,
                         __global const int            *d_csrColIdx,
                         __global const VALUE_TYPE     *d_csrVal,
                         __global volatile int         *d_get_value,
                         const int                      m,
                         __global const VALUE_TYPE     *d_b,
                         __global volatile VALUE_TYPE  *d_x,
                         __global const int            *d_warp_num,
                         const int                      Len)
{
    const int global_id = get_global_id(0);
    const int warp_id = global_id/WARP_SIZE;
    const int local_id = get_local_id(0);
    
    int row;
    
    if(warp_id>=(Len-1))
        return;
    
    const int lane_id = (WARP_SIZE - 1) & local_id;
    __local VALUE_TYPE s_left_sum[WARP_PER_BLOCK*WARP_SIZE];
    
    
    if(d_warp_num[warp_id+1]>(d_warp_num[warp_id]+1))
    {
        /* Thread-level Syncfree SpTRSV */
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
            if(atomic_load_explicit((atomic_int*)&d_get_value[col],memory_order_acquire, memory_scope_device)==1)
                //while(d_get_value[col]==1)
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
                mem_fence(CLK_GLOBAL_MEM_FENCE);
                d_get_value[i]=1;
                j++;
            }
        }
    }
    else
    {
        /* Warp-level Syncfree SpTRSV */
        
        row = d_warp_num[warp_id];
        if(row>=m)
            return;

        int col,j=d_csrRowPtr[row]  + lane_id;
        VALUE_TYPE xi,sum=0;
        while(j < (d_csrRowPtr[row+1]-1))
        {
            col=d_csrColIdx[j];
            //if(d_get_value[col]==1)
            if(atomic_load_explicit((atomic_int*)&d_get_value[col],memory_order_acquire, memory_scope_device)==1)
            {
                sum += d_x[col] * d_csrVal[j];
                j += WARP_SIZE;
            }
        }

        s_left_sum[local_id]=sum;

        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        {
            if(lane_id < offset)
            {
                s_left_sum[local_id] += s_left_sum[local_id+offset];
            }
        }

        /* thread 0 in each warp*/
        if (!lane_id)
        {
            xi = (d_b[row] - s_left_sum[local_id]) / d_csrVal[d_csrRowPtr[row+1]-1];
            d_x[row]=xi;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            d_get_value[row]=1;
        }
    }
    
}


