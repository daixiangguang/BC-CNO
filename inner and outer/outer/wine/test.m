
system('nvcc -c kernel.cu -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64"')
mex kernel.cpp kernel.obj  -lcublas -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64"



%  mex AddVectors.cpp AddVectors.obj -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64"
% 
% disp('C/C++compiling done!');
% 
% disp('3.TestAddVectorsCuda()...');
% 
% disp('Twoinput arrays:');
% 
% A = single([1 2 3 4]);
% 
% B = single([10 9 8 7]);
% 
% C = AddVectors(A,B);