#ifndef _TRANS_
#define _TRANS_

#include "common.h"


int matrix_layer(const int         m,
                 const int         n,
                 const int         nnz,
                 const int        *csrRowPtr,
                 const int        *csrColIdx,
                 const VALUE_TYPE *csrVal,
                 int              *layer_add,
                 double           *parallelism_add
                 )

{
    int *layer=(int *)malloc(m*sizeof(int));
    if (layer==NULL)
        printf("layer error\n");
    memset (layer, 0, sizeof(int)*m);

    int *layer_num=(int *)malloc((m+1)*sizeof(int));
    if (layer_num==NULL)
        printf("layer_num error\n");
    memset (layer_num, 0, sizeof(int)*(m+1));
    
    int max_layer;
    int max_layer2=0;
    int max=0;
    unsigned int min=-1;

    // count layer
    int row,j;
    for (row = 0; row < m; row++)
    {
        max_layer=0;
        for (j = csrRowPtr[row]; j < csrRowPtr[row+1]; j++)
        {
            int col = csrColIdx[j];

            if((layer[col]+1)>max_layer)
                max_layer=layer[col]+1;

        }
        layer[row]=max_layer;
        layer_num[max_layer]++;
        if (max_layer>max_layer2)
            max_layer2=max_layer;
    }
    for(j=1;j<=max_layer2;j++)
    {
        if(max<layer_num[j])
            max=layer_num[j];
        if(min>layer_num[j])
            min=layer_num[j];
    }

    double avg=(double)m/max_layer2;
    free(layer);
    free(layer_num);

    //printf("matrix L's layer = %d, average numer of nodes in layer = %d\n",max_layer2,avg);
    int min2=min;
    //printf("the minimun parallelism is %d,the maximun parallelism is %d\n",min2,max);
    *layer_add=max_layer2;
    *parallelism_add=avg;
    //printf(",%d,%d,%d",nnz,max_layer2,avg);
    return max_layer2;

}


void matrix_warp     (const int         m,
                      const int         n,
                      const int         nnz,
                      const int        *csrRowPtr,
                      const int        *csrColIdx,
                      const VALUE_TYPE *csrVal,
                      const int         border,
                      int              *Len_add,
                      int              *warp_num,
                      double           *warp_occupy_add,
                      double           *element_occupy_add
                      )
{
    int end;
    int i;
    int element_n=0;
    double avg_element_n=0;
    int warp_greater=0,warp_lower=0;
    double element_warp=0,row_warp=0;
    int row=0;
    warp_num[0]=0;
    int k=1,j;
    for(i=0;i<m;i=i+WARP_SIZE)
    {
        end = i+WARP_SIZE;
        if(m<end)
            end=m;
        element_n=csrRowPtr[end]-csrRowPtr[i];
        avg_element_n = ((double)element_n)/(end-i);
        if(avg_element_n>=border)
        {
            warp_greater++;
            element_warp+=(double)element_n;
            row_warp+=(double)(end-i);
            for(j=0;j<(end-i);j++)
            {
                row++;
                warp_num[k]=row;
                k++;
            }
        }
        else
        {
            warp_lower++;
            row += (end-i);
            warp_num[k]=row;
            k++;
        }
    }
    
    int Len=k;
    *Len_add=Len;
    *warp_occupy_add = row_warp/m;
    *element_occupy_add = element_warp/nnz;
}


#endif

