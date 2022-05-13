
clc;




system('nvcc -c --compiler-options -fPIC AddVectors.cu')

 mex AddVectorsCuda.cpp AddVectors.o -lcudart -lcufft ...
-L"/usr/local/cuda-10.1/lib64" ...
-v -I"/usr/local/cuda-10.1/include"

A=[3 2 3;7 8 9];
A=gpuArray(A);
B=[4 5 6];
 C = AddVectorsCuda(A',B);





