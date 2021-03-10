
main: main.cu
	nvcc -O3 -w -m64 -Xptxas -dlcm=cg -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61 main.cu -o YYSpTRSV -I/usr/local/cuda-10.2/include -L/usr/local/cuda-10.2/lib64 -lcusparse -lcudart
clean:
	rm -f YYSpTRSV