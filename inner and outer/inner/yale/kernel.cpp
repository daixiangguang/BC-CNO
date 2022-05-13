#include"mex.h"

#include"kernel.h"

// nlhs: 输出变量的个数(lefthand side,调用语句的左手面)

// plhs：输出的mxArray矩阵的头指针

// nrhs: 输入变量个数(righthand side,调用语句的右手面)

// prhs：输入的mxArray矩阵的头指针

// 如果有两个输入变量，那么prhs[0]指向第一个变量

//prhs[1]指向第二个变量

// Matlab的array使用mxArray类型来表示。

//plhs和hrhs都是指向mxArray类型的指针数组

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])

{
    double *initialx = (double*)mxGetData(prhs[0]);
//     int numRowsB = (int)mxGetM(prhs[1]);//prhs[1]指向第二个变量
//     int numColsB = (int)mxGetN(prhs[1]);

//mxGetData 获取数据阵列中的数据
//     double*A = (double*)mxGetData(prhs[0]);
//     double*B = (double*)mxGetData(prhs[1]);
// 生成输入参数的mxArray结构体
    plhs[0]= mxCreateDoubleMatrix(100000,1, mxREAL);
    // plhs[0]= mxCreateNumericMatrix(4000,1, mxSINGLE_CLASS, mxREAL);
    plhs[1]=mxCreateDoubleMatrix(15,165,  mxREAL);
// 获取输出参数的指针
    double*y = (double*)mxGetData(plhs[0]);
    double*gbestx = (double*)mxGetData(plhs[1]);
// 调用子程序
    
    Hopfield_syn_cuda(y,gbestx,initialx);
}