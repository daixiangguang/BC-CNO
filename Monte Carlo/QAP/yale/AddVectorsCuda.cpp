#include"mex.h"
#include"AddVectors.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

   int numRowsA = (int)mxGetM(prhs[0]);
   int numColsA = (int)mxGetN(prhs[0]);
   int numRowsB = (int)mxGetM(prhs[1]);
   int numColsB = (int)mxGetN(prhs[1]);
    int size=numRowsA*numColsA;

   double *A;
   double *B;
   A=mxGetPr(prhs[0]);
   B=mxGetPr(prhs[1]);

   plhs[0]=mxCreateDoubleMatrix(numRowsA,numColsA,mxREAL);
   double *C ;
   C=mxGetPr(plhs[0]); 
   for(int i=0;i<size;i++)
       printf("%lf ",A[i]);
   //addVectors(A, B, C, size);
} 