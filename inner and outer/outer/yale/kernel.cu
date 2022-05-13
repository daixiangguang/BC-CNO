#include "kernel.h"
#include "curand_kernel.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "device_launch_parameters.h"
#include "E:\Matlab R2018a\extern\include\mex.h"


#include <stdio.h>


#define N   165 //样本个数
#define P   15  //样本类数
#define POP 32 //种群数量 
#define SIZE N*P*POP 
#define ALPHA 10 //惩罚参数1
#define BETA  10//惩罚参数2

//神经网络的BLOCKS和HREAD_NUM 
#define BLOCKS POP
#define THREAD_NUM P //THREAD_NUM 小于等于POP，且POP能除尽THREAD_NUM,THREAD_NUM最大1024，THREAD_NUM需除尽32,一般设置成P*POP,如果超过了1024，设置BLOCKS
//x转xt的HREAD_NUM，暂时不要BLOCKS


#define PSO_THREAD_NUM 32 //最大线程数，如果不能除尽POP，设置成POP
#define PSO_BLOCKS SIZE/PSO_THREAD_NUM 

#define RD_THREAD_NUM 32
#define RD_BLOCKS P*POP/RD_THREAD_NUM
#define W 1
#define C1 2
#define C2 2
#define STOPNUM 100

#define data "yale.txt"
#define U N/P
#define ITER 10
#define MAXITER 20000
//cublas矩阵乘法口诀表,A[m,k], B[k,n],C[n,m],alpha=1.0,beta=0.0
//C=A*B
//cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);

//备注
//从内存角度，c语言的矩阵是按行排列，cublas的矩阵是按列排列
//x[n,p*pop],其中n是样本个数，p是类别，pop是种群大小
//计算列和s1,需将x转成xt[n*pop,p],I1[p,1],用来计算s1=xt*I1,s1[n*pop,1]
//计算行和s2,I2[1,n],s2=I2*x,s2[1,p*pop]
//计算t=d*x,d[n,n],x[n,p*pop],t[p*pop,n]
//index[n,p*pop],批量更新矩阵
void generate_rd_num(int* a)
{
	int i, j;

	int b[N];
	// int* b = (int*)malloc(sizeof(int) * N);
	for (i = 0; i < N; i++)
		a[i] = i;
	srand(time(NULL));
	int temp;
	for (i = 0; i < N; i++)
	{
		//j = (int)((float)((N - i) * rand()) / (RAND_MAX + 1.0));
		j = rand() % N;
		temp = a[i];
		a[i] = a[j];
		a[j] = temp;
	}
}

//异步更新，产生batch，保存到矩阵index[n,p*pop]，index的每行中的神经元可以同步更新，每行只能异步更新
void generate_batch(int *index)
{
	int pos;
	int *block = (int*)malloc(sizeof(int) * N*P);
	int* rd = (int*)malloc(sizeof(int) * N);
	int i, j, loop;
	for (i = 0; i < N; i++)
		for (j = 0; j < P; j++)
			block[i*P + j] = (i*P + j * (P + 1)) % (N*P);

	//print_matrix_int(block, N, P);

	for (loop = 0; loop < POP; loop++)
	{
		generate_rd_num(rd);

		//for (int ii=0;ii<N;ii++)
		//	printf("%d ",rd[ii]);
		//	printf("\n");
		//	printf("\n");
		for (i = 0; i < N; i++)
			for (j = 0; j < P; j++)
			{
				index[i*P*POP + loop * P + j] = block[rd[i] * P + j];
			}
		//print_index(index, N, P,POP,loop);
	}
	free(rd);
	free(block);
}

__global__ static void tt()
{
}

//组内同步更新 
__global__ static void parallel_updated_GPU(float *obj, float *lbest, float *lbestx, float*initialx, float *x, float *d, float *s1, float *s2, float *t, int *index, int gen, curandState *globalState)
{
	curandGenerator_t gen1;  //生成随机数变量
	int tid, i, k, b, j;
	int loop;
	//curandState state;
	//curand_init(seed, tid, 0, &state);
	tid = blockIdx.x *blockDim.x + threadIdx.x; //获取线程号0~blocks*THREAD_NUM-1
	int rd[N], temp, ped;
	curandState localState = globalState[tid];
	//tt<<<1,20>>>();

	if (tid%P == 0)
	{

		//产生第一列初始序列
		int col = (curand(&localState)) % P;
		for (int i = 0; i < N; i++)
			rd[i] = i * P;
		//打乱第一列

		for (int i = 0; i < N; i++)
		{


			j = (curand(&localState)) % N;
			temp = rd[i];
			rd[i] = rd[j];
			rd[j] = temp;
		}
		//printf("%d,%d\n", tid, rd[0]);
		//产生随机间距，间距不能整除P，且大于P
		ped = P + 1;
		for (i = 0; i < N; i++)
			for (j = 0; j < P; j++)
				index[i*P*POP + tid + j] = (ped * j + rd[i]) % (N*P);
	}
	for (i = 0; i < N; i++)
	{
		x[i*P*POP + tid] = initialx[i*P*POP + tid];
	}
	__syncthreads();





	b = tid / P;
	float flag = 1.0;
	int it = 0;
	localState = globalState[tid / P];
	while (it < ITER)
	{

		flag = 0;
		//float total=0.0,const1=0.0,const2=0.0;
				//产生第一列初始序列
		for (int i = 0; i < N; i++)
			rd[i] = i;
		//打乱第一列

		for (int i = 0; i < N; i++)
		{


			j = (curand(&localState)) % N;
			temp = rd[i];
			rd[i] = rd[j];
			rd[j] = temp;
		}
		for (loop = 0; loop < N; loop++)
		{

			temp = index[rd[loop] * POP*P + tid];
			i = temp / P;
			k = temp % P + b * P;

			//loop取值0~N-1
			//int tid =  threadIdx.x;
			int pos = P * POP*i + k; //根据线程号计算出index坐标
			float x_pos = x[pos];//根据坐标取出x的数据，准备更新x(pos)
			float half = 0.5;
			//float s1_i = s1[b*N + i];//计算s1(i)
			float s1_i = s1[i*POP + b];//计算s1(i)
			float s2_k = s2[k];//计算s2(k)
			float dedx = 0.5*t[pos] + ALPHA * (s1_i - x_pos - 2 + P) + BETA * (s2_k - x_pos - 2 * U + N);
			//float dedx=t[k*N+i]+ALPHA*(s1_i-half-x_pos)+BETA*(s2_k-x_pos-U+half);
			//float dedx=ALPHA*(s1_i-half-x_pos)+BETA*(s2_k-x_pos-U+half);

			if (-dedx >= 0)
				x[pos] = 1.0;
			else
				x[pos] = -1.0;

			//这里可以优化，if x[i][k]==0 t[k][j]=0
			//for(int j=0;j<N;j++)
				//t[k*N+j]=d[j*N+i]*x[pos];
			if (x[pos] != x_pos)
			{
				s1[i*POP + b] = s1_i - x_pos + x[pos];
				s2[k] = s2_k - x_pos + x[pos];
				float deltx = x[pos] - x_pos;
				for (int j = 0; j < N; j++)
					t[j*P*POP + k] = t[j*P*POP + k] + d[j*N + i] * deltx;
			}
			__syncthreads();
			//flag = flag + (s1[i*POP + b] - 2 + P)*(s1[i*POP + b] - 2 + P) + (s2[k] - 2 * U + N)*(s2[k] - 2 * U + N);
		}
		//__syncthreads();
		it++;
		//if (flag == 0)
		//	break;
	}
	//__syncthreads();
	if (tid%P == 0)
	{
		float total = 0.0, const1 = 0.0, const2 = 0.0;
		int id;
		for (k = tid; k < P + tid; k++)
			for (i = 0; i < N; i++)
			{
				id = i * P*POP + k;
				if (x[id] == 1.0)
					total = total + t[id];
				else
					total = total - t[id];
			}
		for (i = 0; i < N; i++)
			const1 = const1 + ALPHA * (s1[i*POP + b] - 2 + P)*(s1[i*POP + b] - 2 + P);
		for (k = 0; k < P; k++)
			const2 = const2 + BETA * (s2[b*P + k] - 2 * U + N)*(s2[b*P + k] - 2 * U + N);
		total = 0.5*(total  + const1 + const2);
		obj[tid / P] = total;
		//如果是第一次迭代，那么各个种群得到的目标函数值与解肯定是最优的
		if (gen == 0)
		{
			lbest[tid / P] = total;
			for (k = tid; k < tid + P; k++)
				for (i = 0; i < N; i++)
				{
					id = i * P*POP + k;
					lbestx[id] = x[id];
				}
		}
		else
		{
			if (obj[tid / P] < lbest[tid / P])
			{

				lbest[tid / P] = obj[tid / P];
				for (k = tid; k < tid + P; k++)
					for (i = 0; i < N; i++)
					{
						id = i * P*POP + k;
						lbestx[id] = x[id];
					}
			}
		}
	}



}


//从文件读取d
void generate_d(float* d, char* str)
{
	FILE* fp;            /*文件指针*/
	errno_t error;
	float* TemporaryD = d;
	error = fopen_s(&fp, str, "r");
	if (error != 0)
	{
		perror("fail to read");
		exit(1);
	}
	int i = 0;
	while (!feof(fp))
	{
		fscanf_s(fp, "%f", &d[i++]);
		//fseek(fp, 1L, SEEK_CUR);   /*fp指针从当前位置向后移动*/
	}

	fclose(fp);                     //关闭文件
}
void generate_s1(cublasHandle_t handle, float* s1, float *x, float* I1)
{

	//将显存中的x转换成xt,一般THREAD_NUM等于POP，针对一般显卡，线程数量低于POP，采用for循环机制
	//for (int loop = 0; loop < REPEAT; loop++)
	//	trans_x_xt << <1, TRANS_THREAD_NUM >> > (xt, x, loop);


	//x[n,p*sizepop]
	//计算s1,s1=x*I1,x的列和保存到s1

	const float alpha = 1.0f;
	const float beta = 0.0f;
	//计算列和s1,s1=xt*I1,I1[p,1],相当于xt[n*pop,p]是A[m,k]，I1[p,1]是B[k,n]，
	//cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, N*POP, P, &alpha, I1, 1, xt, P, &beta, s1, 1);
	//计算列和s1,s1=x*I1,I1[p,1],相当于x[n,p*pop]是A[m,k]，I1[p*pop,pop]是B[k,n]，
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, POP, N, P*POP, &alpha, I1, POP, x, P*POP, &beta, s1, POP);
}

void generate_s2(cublasHandle_t handle, float* s2, float* x, float* I2)
{
	//x[n,p*sizepop]
	//计算s1,s1=x*I1,x的列和保存到s1


	const float alpha = 1.0f;
	const float beta = 0.0f;
	//计算s2,s2=I2*x,x的行和保存到s2
	//计算列和s1,s2=I2*x,I2[1,n],相当于I2[1,n]是A[m,k]，x[n,p*pop]是B[k,n]，
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, P*POP, 1, N, &alpha, x, P*POP, I2, N, &beta, s2, P*POP);

}

void generate_t(cublasHandle_t handle, float* t, float* d, float* x)
{
	const float alpha = 1.0f;
	const float beta = 0.0f;
	//计算t=d*x,d[n,n],x[n,p*pop],t[p*pop,n],相当于d[n,n]是A[m,k]，x[n,p*pop]是B[k,n]，
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, P*POP, N, N, &alpha, x, P*POP, d, N, &beta, t, P*POP);

}

void generate_I1(float* I)
{
	int i, j;
	for (i = 0; i < POP; i++)
	{
		for (j = 0; j < P*POP; j++)
			I[j*POP + i] = 0.0;
		for (j = i * P; j < (i + 1)*P; j++)
			I[j*POP + i] = 1.0;
	}
}
void generate_I2(float* I, int num)
{
	int i;
	for (i = 0; i < num; i++)
		I[i] = 1.0;
}

void cp_initialx_x(float *x,float *initialx)
{
	int i;
	float rd;
	srand((unsigned)time(NULL));
	for (i = 0; i < SIZE; i++) {

		rd = 2.0 * rand() / RAND_MAX - 1; //产生-1到1的随机数
		if (rd > 0)
			x[i] = 1.0;
		else
			x[i] = -1.0;
	}
}
void initial_x(float* x)
{
	int i;
	float rd;
	srand((unsigned)time(NULL));
	for (i = 0; i < SIZE; i++) {

		rd = 2.0 * rand() / RAND_MAX - 1; //产生-1到1的随机数
		if (rd > 0)
			x[i] = 1.0;
		else
			x[i] = -1.0;
	}
}

void initial_obj(float* x)
{
	int i;
	for (i = 0; i < POP; i++)
		x[i] = 0.0;
}

//计算全局最优目标函数值和最优解,
__global__ static void generate_global_best(float *y, float *gbest, float *gbestx, float * lbest, float *lbestx, int it)
{

	int i, j, k, flag = 0;
	//假设种群0的目标函数值最小且具有最优解
	float gb = lbest[0];
	int id = 0;
	//获取全局最优解
	for (i = 0; i < POP; i++)
		if (lbest[i] < gb)
		{
			gb = lbest[i];
			id = i;
		}
	if (it == 0 || gb < *gbest)
	{
		*gbest = gb;
		y[it] = gb;
		k = 0;
		int count = 0;
		for (j = 0; j < N; j++)
			for (i = id * P; i < (id + 1)*P; i++)
				gbestx[k++] = lbestx[j*P*POP + i];

		/*
			for (j = 0; j < N; j++)
			{
				for (i = id * P; i < (id + 1)*P; i++)
					printf("%f ", lbestx[j*P*POP + i]);
				printf("\n");
			}*/
	}
	else
	{
		y[it] = *gbest;
	}

	//printf("%d=%lf\n", it, y[it]);


}

//保存全局最优目标函数值和局部最优目标函数值,
__global__ static void restore_global_local_objectives(float *objectives,float * lbest,int it)
{
    int i;
    for(i=0;i<POP;i++)
        objectives[it*POP+i]=lbest[i];
}

__global__ static void pso(float *x, float *initialx, float *initialv, float *gbestx, float *lbestx, float *rd1, float *rd2, float *rd3)
{

	int i, k;
	int tid = blockIdx.x *blockDim.x + threadIdx.x; //获取线程号0~blocks*THREAD_NUM-1


	//printf("%f %f\n", rd1,rd2);
	//float rd1 = 0.7;
	//float rd2 = 0.5;
	/*
	if (tid == 0)
	{
		printf("\n");
		for (i = 0; i < N; i++)
		{
			for (k = 0; k < P; k++)
				printf("% f", initialx[i*P + k]);
			printf("\n");
		}
	}*/
	i = tid / (P*POP);
	k = tid % (P*POP) % P;

	/*
	   initial_v{j}=w1*initial_v{j}+beta1*rd1*(pbest_x{j}-initial_x{j})+beta2*rd2*(zbest_x-initial_x{j});
	   initial_x{j}=initial_x{j}+initial_v{j};
	   initial_x{j}=round(min(1,max(0,initial_x{j})));  %for zero one
	*/


	/*
	initialv[tid] = W * initialv[tid] + C1 * rd1[tid % (P*POP) / P] * (lbestx[tid] - initialx[tid]) + C2 * rd2[tid % (P*POP) / P] * (gbestx[i*P + k] - initialx[tid]);
	initialx[tid] = initialx[tid] + initialv[tid];
	if (initialx[tid] <= -1.0)
		initialx[tid] = -1.0;
	if (initialx[tid] >= 1.0)
		initialx[tid] = 1.0;
	if (initialx[tid] >= 0)
		initialx[tid] = 1.0;
	else
		initialx[tid] = -1.0;*/


		//initialv[tid] = W * initialv[tid] + C1 * rd1[tid % (P*POP) / P] * ((lbestx[tid] + 1) / 2.0 - (initialx[tid] + 1) / 2.0) + C2 * rd2[tid % (P*POP) / P] * ((gbestx[i*P + k] + 1) / 2.0 - (initialx[tid] + 1) / 2.0);
	initialv[tid] = W * initialv[tid] + C1 * rd1[tid] * ((lbestx[tid] + 1) / 2.0 - (initialx[tid] + 1) / 2.0) + C2 * rd1[tid] * ((gbestx[i*P + k] + 1) / 2.0 - (initialx[tid] + 1) / 2.0);
	float s;


	s = 1 / (1 + exp(-initialv[tid]));

	if (s > rd3[tid])
		initialx[tid] = 1.0;
	else
		initialx[tid] = -1.0;

	//if (5 / (N*P) > rd2[tid])
		//initialx[tid] = 1 - 2 * (1 - (initialx[tid] + 1) / 2.0);


}

void initial_v(float* x)
{
	int i;
	float rd;
	srand((unsigned)time(NULL));
	for (i = 0; i < SIZE; i++) {

		rd = 2.0 * rand() / RAND_MAX - 1; //产生-1到1的随机数
		x[i] = rd;
		/*
				if (rd > 0)
					x[i] =0;
				else
					x[i] = 0;
		*/
	}
}
__global__ void setup_kernel(curandState *state, unsigned long seed)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x; //获取线程号0~blocks*THREAD_NUM-1
	curand_init(seed, tid, 0, &state[tid]);// initialize the state
}
__global__ void generate_rd1_rd2(float *rd1, float *rd2, float *rd3, curandState *globalState)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x; //获取线程号0~blocks*THREAD_NUM-1
	int i;
	curandState localState;
	if (tid%P == 0)
	{
		//localState = globalState[tid / P];
		//rd1[tid / P] = curand_uniform(&localState);
		//rd2[tid / P] = curand_uniform(&localState);
	}
	localState = globalState[tid];
	for (i = 0; i < N; i++)
	{
		rd1[i*P*POP + tid] = curand_uniform(&localState);
		rd2[i*P*POP + tid] = curand_uniform(&localState);
		rd3[i*P*POP + tid] = curand_uniform(&localState);
		//printf("%d %lf", i*P*POP + tid, rd3[i*P*POP + tid]);
	}
}
__global__ void stop(float *y, int *is_stop, int it)
{
	*is_stop = 0;
	if (y[it] == y[it - STOPNUM])
		*is_stop = 1;
}
int Hopfield_syn_cuda(float *y,float *gbestx,float *initialx)
{
	
	//选择显卡,ubuntu用户有3块2080ti显卡，西南大学有2块rtx8000显卡
	cudaSetDevice(1);


	float *xh, *xth, *xd, *xtd;
	xh = (float*)malloc(sizeof(float) * SIZE); //x的内存变量，申请内存空间
	xth = (float*)malloc(sizeof(float)*SIZE); //xt的内存变量，申请内存空间
	cp_initialx_x(xh,initialx);//printf("\n");print_matrix(xh, N, P*POP);//初始化x
	cudaMalloc((void**)&xd, sizeof(float) * SIZE); //x的显存变量，申请显存空间
	cudaMalloc((void**)&xtd, sizeof(float) * SIZE); //xtd的显存变量，申请显存空间
	cudaMemcpy(xd, xh, sizeof(int) * SIZE, cudaMemcpyHostToDevice); //内存中的x复制到显存

	//多显卡计算，暂时不用#pragma omp parallel for num_threads(3)


	//测试x与xt是否正确，暂时不用
	/*print_matrix(xh, N, P*POP);
	cudaMemcpy(xth, xtd, sizeof(int)*SIZE, cudaMemcpyDeviceToHost); //显存数据拷贝回内存
	print_matrix(xth, N* POP, P);*/

	//申请obj,index,I1、I2、s1、s2、t与d的内存与显存变量 
	int *indexd, *indexh;
	float *sd1, *sd2, *sh1, *sh2, *Id1, *Id2, *Ih1, *Ih2, *td, *th, *dd, *dh;
	float *objd, *objh;
	float *initialxh, *initialxd, *initialvh, *initialvd;
	initialxh = (float*)malloc(sizeof(float)*SIZE);
	initialvh = (float*)malloc(sizeof(float)*SIZE); initial_v(initialvh);
	cudaMalloc((void**)&initialvd, sizeof(float) * SIZE);
	indexh = (int*)malloc(sizeof(int) *SIZE); generate_batch(indexh);//index的内存变量，申请内存空间
	Ih1 = (float*)malloc(sizeof(float)  * POP*P*POP); generate_I1(Ih1);//I1的内存变量，申请内存空间
	Ih2 = (float*)malloc(sizeof(float) * N); generate_I2(Ih2, N);//I2的内存变量，申请内存空间
	sh1 = (float*)malloc(sizeof(float) * N*POP);//s1的内存变量，申请内存空间
	sh2 = (float*)malloc(sizeof(float) * P*POP);//s2的内存变量，申请内存空间
	th = (float*)malloc(sizeof(float) * SIZE);//t的内存变量，申请内存空间
	dh = (float*)malloc(sizeof(float) * N*N); generate_d(dh, data);//d的内存变量，申请内存空间
	objh = (float*)malloc(sizeof(float) * POP); initial_obj(objh);//obj的内存变量，申请内存空间

	cudaMalloc((void**)&indexd, sizeof(int) * SIZE);//I1的内存变量，申请显存空间
	cudaMalloc((void**)&Id1, sizeof(float) * POP*P*POP);//I1的内存变量，申请显存空间
	cudaMalloc((void**)&Id2, sizeof(float) * N);//I2的显存变量，申请显存空间
	cudaMalloc((void**)&sd1, sizeof(float) * N*POP);//I1的内存变量，申请显存空间
	cudaMalloc((void**)&sd2, sizeof(float) * P*POP);//I2的显存变量，申请显存空间
	cudaMalloc((void**)&td, sizeof(float) * SIZE);//t的显存变量，申请显存空间
	cudaMalloc((void**)&dd, sizeof(float) * N*N);//d的显存变量，申请显存空间
	cudaMalloc((void**)&objd, sizeof(float) * POP);//obj的显存变量，申请显存空间
	cudaMalloc((void**)&initialxd, sizeof(float) * SIZE);
	cudaMemcpy(initialxd, xh, sizeof(float) * SIZE, cudaMemcpyHostToDevice); //内存中的x复制到显存
	cudaMemcpy(initialvd, initialvh, sizeof(float) * SIZE, cudaMemcpyHostToDevice); //内存中的x复制到显存

	cudaMemcpy(indexd, indexh, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(Id1, Ih1, sizeof(float) * POP*P*POP, cudaMemcpyHostToDevice);
	cudaMemcpy(Id2, Ih2, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(td, th, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dd, dh, sizeof(float) * N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(objd, objh, sizeof(float) * POP, cudaMemcpyHostToDevice);



	float *yh, *yd, *gbestd, *gbesth, *gbestxd, *gbestxh, *lbesth, *lbestxh, *lbestd, *lbestxd;
	yh = (float*)malloc(sizeof(float) *MAXITER);
	gbesth = (float*)malloc(sizeof(float));
	gbestxh = (float*)malloc(sizeof(float)*N*P);
	lbestxh = (float*)malloc(sizeof(float)*SIZE);
	lbesth = (float*)malloc(sizeof(float)*POP);
	cudaMalloc((void**)&yd, sizeof(float) * MAXITER);
	cudaMalloc((void**)&gbestd, sizeof(float));
	cudaMalloc((void**)&gbestxd, sizeof(float)*N*P);
	cudaMalloc((void**)&lbestd, sizeof(float)*POP);
	cudaMalloc((void**)&lbestxd, sizeof(float)*SIZE);

	float *rdh1, *rdh2, *rdh3;
	float *rdd1, *rdd2, *rdd3;
	rdh1 = (float*)malloc(sizeof(float) *SIZE);
	rdh2 = (float*)malloc(sizeof(float) *SIZE);
	rdh3 = (float*)malloc(sizeof(float) *SIZE);
	cudaMalloc((void**)&rdd1, sizeof(float) * SIZE);
	cudaMalloc((void**)&rdd2, sizeof(float) *SIZE);
	cudaMalloc((void**)&rdd3, sizeof(float) * SIZE);
	cublasHandle_t handle;
	cublasCreate(&handle);
	srand((unsigned int)time(NULL));
	curandState* devStates;

	cudaMalloc(&devStates, P*POP * sizeof(curandState));

	srand(time(0));
	int gen;
	int is_stoph, *is_stopd, stop_number = STOPNUM;
	cudaMalloc((void**)&is_stopd, sizeof(int));

    float *objectivesh,*objectivesd;
    objectivesh = (float*)malloc(sizeof(float) * MAXITER*POP); //x的内存变量，申请内存空间
    cudaMalloc((void**)&objectivesd, sizeof(float)* MAXITER*POP);

	for (gen = 0; gen < MAXITER; gen++)
	{



		generate_s1(handle, sd1, initialxd, Id1);
		generate_s2(handle, sd2, initialxd, Id2);
		generate_t(handle, td, dd, initialxd);
		//这里种子过多，可以优化哈
		setup_kernel << <BLOCKS, THREAD_NUM >> > (devStates, rand());
		parallel_updated_GPU << <BLOCKS, THREAD_NUM >> > (objd, lbestd, lbestxd, initialxd, xd, dd, sd1, sd2, td, indexd, gen, devStates);//计算每行
		generate_global_best << <1, 1, 0 >> > (yd, gbestd, gbestxd, lbestd, lbestxd, gen);
        restore_global_local_objectives<< <1, 1, 0 >> >(objectivesd,lbestd,gen);
		generate_rd1_rd2 << <RD_BLOCKS, RD_THREAD_NUM >> > (rdd1, rdd2, rdd3, devStates);
		//cudaMemcpy(yh, yd, sizeof(float)*MAXITER, cudaMemcpyDeviceToHost);
		pso << <PSO_BLOCKS, PSO_THREAD_NUM >> > (xd, initialxd, initialvd, gbestxd, lbestxd, rdd1, rdd2, rdd3);//计算每行

		//cudaMemcpy(rdh3, rdd3, sizeof(float)*SIZE, cudaMemcpyDeviceToHost);
		//cudaMemcpy(sh1, sd1, sizeof(int)*N*POP, cudaMemcpyDeviceToHost);

		//cudaMemcpy(sh2, sd2, sizeof(int)*P*POP, cudaMemcpyDeviceToHost);

		if (gen > stop_number)
		{
			stop << <1, 1, 0 >> > (yd, is_stopd, gen);
			cudaMemcpy(&is_stoph, is_stopd, sizeof(int), cudaMemcpyDeviceToHost);
			if (is_stoph == 1)
				break;
			else
				stop_number = stop_number + STOPNUM;
		}

	}

	cudaMemcpy(gbestx, gbestxd, sizeof(float) * N*P, cudaMemcpyDeviceToHost);
	cudaMemcpy(y, objectivesd, sizeof(float)*MAXITER*POP, cudaMemcpyDeviceToHost);





	free(xh);
	free(indexh);
	free(sh1);
	free(sh2);
	free(Ih1);
	free(Ih2);
	free(th);
	free(dh);
	free(objh);
	free(initialxh);
	free(initialvh);
	free(yh);
	free(gbesth);
	free(gbestxh);
	free(lbesth);
	free(lbestxh);
	free(rdh1);
	free(rdh2);
    free(objectivesh);

    cudaFree(objectivesd);
	cudaFree(xd);
	cudaFree(indexd);
	cudaFree(sd1);
	cudaFree(sd2);
	cudaFree(Id1);
	cudaFree(Id2);
	cudaFree(td);
	cudaFree(dd);
	cudaFree(objd);
	cudaFree(initialxd);
	cudaFree(initialvd);
	cudaFree(yd);
	cudaFree(gbestd);
	cudaFree(gbestxd);
	cudaFree(lbestd);
	cudaFree(lbestxd);
	cudaFree(rdd1);
	cudaFree(rdd2);
	cudaFree(is_stopd);
	cublasDestroy(handle);
    return 0;
}