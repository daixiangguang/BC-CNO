function nvmex(cuFileName)
CUDA_INC_Location = '/home/ubuntu/bin/Download/cuda/include';
CUDA_SAMPLES_Location = '/home/ubuntu/bin/Download/NVIDIA_CUDA-10.1_Samples/common/inc';
Host_Compiler_Location = ' ';
PIC_Option = ' --compiler-options -fPIC ';
machine_str = [];
CUDA_LIB_Location = '/home/ubuntu/bin/Download/cuda/lib64';

[~, filename] = fileparts(cuFileName);
nvccCommandLine = [ ...
'nvcc --compile ' Host_Compiler_Location ' ' ...
'-o ' filename '.o ' ...
machine_str PIC_Option ...
' -I' '"' matlabroot '/extern/include "' ...
' -I' CUDA_INC_Location ' -I' CUDA_SAMPLES_Location ...
' "' cuFileName '" '];
mexCommandLine = ['mex ' filename '.o' ' -L' CUDA_LIB_Location ' -lcudart'];
disp(nvccCommandLine);
warning off;
status = system(nvccCommandLine);  %system编译，编译成功则status >=0，封装失败，则status < 0，编译产生.o文件或者.Obj文件
warning on;
if status < 0
    error 'Error invoking nvcc';
end


disp(mexCommandLine);
eval(mexCommandLine); %mex执行，编译.o文件（在Windows系统下为.obj文件），并封装为MATLAB可调用的mexa64/32文件函数
%同时如果将CUDA和Cpp文件写开的话，可在mexCommandLine加入CPP文件和.o文件： mexCommandLine= ['mex '  filename  '.cpp ' filename  '.o'  ' -L' CUDA_LIB_Location  ' -lcudart'];   同时：filename  也可以是自己随意定义的名字



