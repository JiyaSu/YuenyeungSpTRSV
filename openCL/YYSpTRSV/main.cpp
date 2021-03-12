#include "common.h"
#include "mmio.h"
#include "tranpose.h"
#include "read_mtx.h"
#include "YYSpTRSV.h"


int main(int argc, char ** argv)
{

    // report precision of floating-point
    printf("---------------------------------------------------------------------------------------------\n");
    char  *precision;
    if (sizeof(VALUE_TYPE) == 4)
    {
        precision = (char *)"32-bit Single Precision";
    }
    else if (sizeof(VALUE_TYPE) == 8)
    {
        precision = (char *)"64-bit Double Precision";
    }
    else
    {
        printf("Wrong precision. Program exit!\n");
        return 0;
    }
    
    printf("PRECISION = %s\n", precision);
    printf("Benchmark REPEAT = %i\n", BENCH_REPEAT);
    printf("---------------------------------------------------------------------------------------------\n");
    
    int m, n, nnzA;
    int *csrRowPtrA;
    int *csrColIdxA;
    VALUE_TYPE *csrValA;
    
    //ex: ./YYSpTRSV webbase-1M.mtx
    int argi = 1;
    
    char  *filename;
    if(argc > argi)
    {
        filename = argv[argi];
        argi++;
    }
    printf("-------------- %s --------------\n", filename);
    
    read_mtx(filename, &m, &n, &nnzA, &csrRowPtrA, &csrColIdxA, &csrValA);

    //printf("read_mtx finish\n");
    
    /* extract Matrix L with the unit-lower triangular sparsity structure of input Matrix A */
    int nnzL = 0;
    int *csrRowPtrL_tmp ;
    int *csrColIdxL_tmp ;
    VALUE_TYPE *csrValL_tmp;
    if(m<=n)
        n=m;
    else
        m=n;
    if (m<1)
        return 0;

    change2tran(m, nnzA,csrRowPtrA, csrColIdxA, csrValA, &nnzL, &csrRowPtrL_tmp, &csrColIdxL_tmp, &csrValL_tmp);
    printf("A's unit-lower triangular L: ( %i, %i ) nnz = %i\n", m, n, nnzL);

    
    free(csrColIdxA);
    free(csrValA);
    free(csrRowPtrA);

    if(m==0 || nnzL==0)
        return -3;
    

    /* calculate the number of layer and parallelism of matrix L */
    int layer;
    double parallelism;

    matrix_layer(m,n,nnzL,csrRowPtrL_tmp,csrColIdxL_tmp,csrValL_tmp,&layer,&parallelism);

    /* get vector b and reference x for Lx=b */
    VALUE_TYPE *x_ref;
    VALUE_TYPE *b;
    get_x_b(m, n, csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp, &x_ref, &b);
    
    VALUE_TYPE *x = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n);
    
    /* The border between thread-level and warp-level algorithms, according to the number of non-zero elements in each row of the matrix L*/
    int border = 10;
    
    
    /* !!!!!! start computing SpTRSV !!!!!!!! */
    double solve_time,gflops,bandwith,pre_time,warp_occupy,element_occupy;
    int success = YYSpTRSV_csr(m,n,nnzL,csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp, b, x, border, &pre_time, &solve_time, &gflops, &bandwith, &warp_occupy, &element_occupy);
    
    
    /* check solution x */
    int err_counter = 0;
    for (int i = 0; i < n; i++)
    {
        if (abs(x_ref[i] - x[i]) > 0.01 * abs(x_ref[i]))
        {
            err_counter++;
        }
    }
    
    if (!err_counter)
        printf("YYSpTRSV on L passed!\n");

    
    printf("The unit-lower triangular L (%s): ( %i, %i ) nnz = %i, layer = %d, parallelism = %4.2f\n", filename, m, n, nnzL, layer, parallelism);
    printf("The preprocessing time = %4.2f ms, solving time =  %4.2f ms, throught = %4.2f gflops, bandwidth = %4.2f GB/s.\n", pre_time, solve_time, gflops, bandwith);
    


    free(x);
    free(x_ref);
    free(b);
    free(csrColIdxL_tmp);
    free(csrValL_tmp);
    free(csrRowPtrL_tmp);
    
    return 0;
}
    

//int main(int argc, char ** argv)
//{
//
//
//    FILE *fp = fopen("result.csv","a");
//    if(fp==NULL)
//        return -1;
//
//    // "Usage: ``./sptrsv A.mtx'' for LX=B on device 0"
//
//    if(argc!=2)
//    {
//        printf("Usage:/sptrsv example.mtx\n");
//        return -1;
//    }
//
//
//
//    // load matrix data from file
//    char  *filename;
//    filename = argv[1];
//
//    int m, n, nnzA;
//    int *csrRowPtrA;
//    int *csrColIdxA;
//    VALUE_TYPE *csrValA;
//
//    read_mtx(filename, &m, &n, &nnzA, &csrRowPtrA, &csrColIdxA, &csrValA);
//
//    // extract L with the unit-lower triangular sparsity structure of A
//    int nnzL = 0;
//    int *csrRowPtrL_tmp ;
//    int *csrColIdxL_tmp ;
//    VALUE_TYPE *csrValL_tmp;
//
//    if(m<=n)
//        n=m;
//    else
//        m=n;
//    if (m<=1)
//        return 0;
//
//    change2tran(m, nnzA,csrRowPtrA, csrColIdxA, csrValA, &nnzL, &csrRowPtrL_tmp, &csrColIdxL_tmp, &csrValL_tmp);
//
//    free(csrColIdxA);
//    free(csrValA);
//    free(csrRowPtrA);
//
//    if(m==0 || nnzL==0)
//        return -3;
//
//
//    int layer;
//    double parallelism,solve_time,gflops,bandwith,pre_time=0,warp_occupy,element_occupy;
//
//    matrix_layer(m,n,nnzL,csrRowPtrL_tmp,csrColIdxL_tmp,csrValL_tmp,&layer,&parallelism);
//
//    VALUE_TYPE *x_ref;
//    VALUE_TYPE *b ;
//    get_x_b(m, n, csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp, &x_ref, &b);
//    VALUE_TYPE *x = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * m);
//
//    sptrsv_syncfree_opencl(csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp, m, n, nnzL,x,b,x_ref,&pre_time,&solve_time,&gflops,&bandwith,&warp_occupy, &element_occupy);
//    sptrsv_syncfree_opencl(csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp, m, n, nnzL,x,b,x_ref,&pre_time,&solve_time,&gflops,&bandwith,&warp_occupy, &element_occupy);
//
//    fprintf(fp,"%s, %d, %d, %4.2f, %d, %4.2f, %4.2f, %4.2f, %4.2f, %4.2f, %4.2f, %4.2f\n",filename,m,nnzL,(double)nnzL/m,layer,parallelism,pre_time,solve_time,gflops,bandwith,warp_occupy,element_occupy);
//
//
//    // done!
//    free(csrColIdxL_tmp);
//    free(csrValL_tmp);
//    free(csrRowPtrL_tmp);
//    fclose(fp);
//
//
//    free(x_ref);
//    free(b);
//    free(x);
//
//    return 0;
//}
