#include "AddVectors.h"
#include<stdio.h>

__global__ void addVectorsMask(double *devPtrA, double *devPtrB, double *devPtrC, int size)
{
    int i = threadIdx.x ;//+ blockIdx.x * blockDim.x;
   // if(i!= size)
       // return;

    devPtrC[i] = devPtrA[i] + devPtrB[i];
__syncthreads();
}

void addVectors(double *A, double *B, double *C, int size)
{
    double *devPtrA,*devPtrB,*devPtrC;
    cudaMalloc(&devPtrA,sizeof(double)* size);
    cudaMalloc(&devPtrB,sizeof(double)* size);
    cudaMalloc(&devPtrC,sizeof(double)* size);

    cudaMemcpy(devPtrA,A, sizeof(double)* size, cudaMemcpyHostToDevice);
    cudaMemcpy(devPtrB,B, sizeof(double)* size, cudaMemcpyHostToDevice);
    addVectorsMask<<<1,size>>>(devPtrA,devPtrB, devPtrC, size);

    cudaMemcpy(C,devPtrC, sizeof(double)* size, cudaMemcpyDeviceToHost);
	
	double *d=(double *)malloc(sizeof(double)* size);
	cudaMemcpy(d,devPtrC, sizeof(double)* size, cudaMemcpyDeviceToHost);
    for(int i=0;i<size;i++)
	{
		printf("A=%f\n",A[i]);
	}

	free(d);

    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);

}