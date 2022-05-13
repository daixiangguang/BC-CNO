#include "kernel.h"
#include "curand_kernel.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "device_launch_parameters.h"
#include "E:\Matlab R2018a\extern\include\mex.h"

#define N   144 //��������
#define P   3  //��������
#define POP 512 //��Ⱥ���� 
#define SIZE N*P*POP 
#define ALPHA 100 //�ͷ�����1
#define BETA 100  //�ͷ�����2

//�������BLOCKS��HREAD_NUM 
#define BLOCKS POP
#define THREAD_NUM P //THREAD_NUM С�ڵ���POP����POP�ܳ���THREAD_NUM,THREAD_NUM���1024��THREAD_NUM�����32,һ�����ó�P*POP,���������1024������BLOCKS
//xתxt��HREAD_NUM����ʱ��ҪBLOCKS


#define PSO_THREAD_NUM 512 //����߳�����������ܳ���POP�����ó�POP
#define PSO_BLOCKS SIZE/PSO_THREAD_NUM 

#define RD_THREAD_NUM 512
#define RD_BLOCKS POP/RD_THREAD_NUM
#define W 1
#define C1 2
#define C2 2
#define STOPNUM 50

#define data "wine.txt"
#define U N/P
#define ITER 30
#define MAXITER 500
//cublas����˷��ھ���,A[m,k], B[k,n],C[n,m],alpha=1.0,beta=0.0
//C=A*B
//cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);

//��ע
//���ڴ�Ƕȣ�c���Եľ����ǰ������У�cublas�ľ����ǰ�������
//x[n,p*pop],����n������������p�����pop����Ⱥ��С
//�����к�s1,�轫xת��xt[n*pop,p],I1[p,1],��������s1=xt*I1,s1[n*pop,1]
//�����к�s2,I2[1,n],s2=I2*x,s2[1,p*pop]
//����t=d*x,d[n,n],x[n,p*pop],t[p*pop,n]
//index[n,p*pop],�������¾���

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

//�첽���£�����batch�����浽����index[n,p*pop]��index��ÿ���е���Ԫ����ͬ�����£�ÿ��ֻ���첽����
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



//����ͬ������ 
__global__ static void parallel_updated_GPU(float *obj, float *lbest, float *lbestx, float*initialx, float *x, float *d, float *s1, float *s2, float *t, int *index, int gen, curandState *globalState)
{
	curandGenerator_t gen1;  //�������������
	int tid, i, k, b, j;
	int loop;
	//curandState state;
	//curand_init(seed, tid, 0, &state);
	tid = blockIdx.x *blockDim.x + threadIdx.x; //��ȡ�̺߳�0~blocks*THREAD_NUM-1
	int rd[N], temp, ped;
	curandState localState = globalState[tid];
	if (tid%P == 0)
	{

		//������һ�г�ʼ����
		for (int i = 0; i < N; i++)
			rd[i] = i * P;
		//���ҵ�һ��

		for (int i = 0; i < N; i++)
		{


			j = (curand(&localState)) % N;
			temp = rd[i];
			rd[i] = rd[j];
			rd[j] = temp;
		}
		//���������࣬��಻������P���Ҵ���P
		while (1)
		{
			ped = (curand(&localState)) % N;
			if (ped%P != 0 && ped > P)
				break;
		}
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
	while (it < ITER && (flag != 0.0))
	{
		flag = 0;
		//float total=0.0,const1=0.0,const2=0.0;
		int tidtemp = (curand(&localState)) % (POP*P);
		for (loop = 0; loop < N; loop++)
		{

			temp = index[loop*POP*P + tid];
			i = temp / P;
			k = temp % P + b * P;

			//loopȡֵ0~N-1
			//int tid =  threadIdx.x;
			int pos = P * POP*i + k; //�����̺߳ż����index����
			float x_pos = x[pos];//��������ȡ��x�����ݣ�׼������x(pos)
			float half = 0.5;
			//float s1_i = s1[b*N + i];//����s1(i)
			float s1_i = s1[i*POP + b];//����s1(i)
			float s2_k = s2[k];//����s2(k)
			float dedx = t[pos] + ALPHA * (s1_i - x_pos - 2 + P) + BETA * (s2_k - x_pos - 2 * U + N);
			//float dedx=t[k*N+i]+ALPHA*(s1_i-half-x_pos)+BETA*(s2_k-x_pos-U+half);
			//float dedx=ALPHA*(s1_i-half-x_pos)+BETA*(s2_k-x_pos-U+half);

			if (-dedx > 0)
				x[pos] = 1.0;
			else if (-dedx < 0)
				x[pos] = -1.0;

			//��������Ż���if x[i][k]==0 t[k][j]=0
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
			//__syncthreads();
			flag = flag + (s1[i*POP + b] - 2 + P)*(s1[i*POP + b] - 2 + P) + (s2[k] - 2 * U + N)*(s2[k] - 2 * U + N);
		}
		it++;
		//if (flag == 0)
		//	break;
	}
	__syncthreads();
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
		total = total * 0.5 + const1 + const2;
		obj[tid / P] = total;
		//����ǵ�һ�ε�������ô������Ⱥ�õ���Ŀ�꺯��ֵ���϶������ŵ�
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


//���ļ���ȡd
void generate_d(float* d, char* str)
{
	FILE* fp;            /*�ļ�ָ��*/
	errno_t error;
	float* TemporaryD = d;
	error = fopen_s(&fp, str, "r");
	if (error != 0)
	{
		perror("fail to read");
		exit(1);
	}
	for (size_t i = 0; i < N; i++) {
		for (size_t j = 0; j < N; j++)
		{
			fscanf_s(fp, "%f", TemporaryD);
			TemporaryD++;
			fseek(fp, 1L, SEEK_CUR);   /*fpָ��ӵ�ǰλ������ƶ�*/
		}
	}
	fclose(fp);                     //�ر��ļ�
}
void generate_s1(cublasHandle_t handle, float* s1, float *x, float* I1)
{

	//���Դ��е�xת����xt,һ��THREAD_NUM����POP�����һ���Կ����߳���������POP������forѭ������
	//for (int loop = 0; loop < REPEAT; loop++)
	//	trans_x_xt << <1, TRANS_THREAD_NUM >> > (xt, x, loop);


	//x[n,p*sizepop]
	//����s1,s1=x*I1,x���кͱ��浽s1

	const float alpha = 1.0f;
	const float beta = 0.0f;
	//�����к�s1,s1=xt*I1,I1[p,1],�൱��xt[n*pop,p]��A[m,k]��I1[p,1]��B[k,n]��
	//cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, N*POP, P, &alpha, I1, 1, xt, P, &beta, s1, 1);
	//�����к�s1,s1=x*I1,I1[p,1],�൱��x[n,p*pop]��A[m,k]��I1[p*pop,pop]��B[k,n]��
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, POP, N, P*POP, &alpha, I1, POP, x, P*POP, &beta, s1, POP);
}

void generate_s2(cublasHandle_t handle, float* s2, float* x, float* I2)
{
	//x[n,p*sizepop]
	//����s1,s1=x*I1,x���кͱ��浽s1


	const float alpha = 1.0f;
	const float beta = 0.0f;
	//����s2,s2=I2*x,x���кͱ��浽s2
	//�����к�s1,s2=I2*x,I2[1,n],�൱��I2[1,n]��A[m,k]��x[n,p*pop]��B[k,n]��
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, P*POP, 1, N, &alpha, x, P*POP, I2, N, &beta, s2, P*POP);

}

void generate_t(cublasHandle_t handle, float* t, float* d, float* x)
{
	const float alpha = 1.0f;
	const float beta = 0.0f;
	//����t=d*x,d[n,n],x[n,p*pop],t[p*pop,n],�൱��d[n,n]��A[m,k]��x[n,p*pop]��B[k,n]��
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


void initial_x(float* x)
{
	int i;
	float rd;
	srand((unsigned)time(NULL));
	for (i = 0; i < SIZE; i++) {

		rd = 2.0 * rand() / RAND_MAX - 1; //����-1��1�������
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

//����ȫ������Ŀ�꺯��ֵ�����Ž�,
__global__ static void generate_global_best(float *y, float *gbest, float *gbestx, float * lbest, float *lbestx, int it)
{

	int i, j, k, flag = 0;
	//������Ⱥ0��Ŀ�꺯��ֵ��С�Ҿ������Ž�
	float gb = lbest[0];
	int id = 0;
	//��ȡȫ�����Ž�
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

	//printf("%f\n", y[it]);


}
__global__ static void pso(float *x, float *initialx, float *initialv, float *gbestx, float *lbestx, float *rd1, float *rd2,unsigned long seed)
{

	int i, k;
	int tid = blockIdx.x *blockDim.x + threadIdx.x; //��ȡ�̺߳�0~blocks*THREAD_NUM-1


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

	initialv[tid] = W * initialv[tid] + C1 * rd1[tid % (P*POP) / P] * ((lbestx[tid]+1)/2.0 - (initialx[tid]+1)/2.0) + C2 * rd2[tid % (P*POP) / P] * ((gbestx[i*P + k]+1)/2.0 - (initialx[tid]+1)/2.0);
	/*
	initialx[tid] = initialx[tid] + initialv[tid];
	if (initialx[tid] < 0.0)
		initialx[tid] = -1.0;
	if (initialx[tid] >= 1.0)
		initialx[tid] = 1.0;
	if (initialx[tid] >= 0.5)
		initialx[tid] = 1.0;
	else
		initialx[tid] = -1.0;
		*/
	initialv[tid] = W * initialv[tid] + C1 * rd1[tid % (P*POP) / P] * (lbestx[tid] - initialx[tid]) + C2 * rd2[tid % (P*POP) / P] * (gbestx[i*P + k] - initialx[tid]);
	float s, rd;
	s = 1 / (1 + exp(-initialv[tid]));
	curandState state;
	curand_init(seed, tid, 0, &state);
	rd = curand_uniform(&state);
	if (s > rd)
		initialx[tid] = 1.0;
	else
		initialx[tid] = -1.0;

	/*
if(initialv[tid]>0.0)
	initialv[tid]=1.0;
else
	initialv[tid]=-1.0;
	*/
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


}

void initial_v(float* x)
{
	int i;
	float rd;
	srand((unsigned)time(NULL));
	for (i = 0; i < SIZE; i++) {

		rd = 2.0 * rand() / RAND_MAX - 1; //����-1��1�������
		if (rd > 0)
			x[i] = 1.0;
		else
			x[i] = -1.0;
	}
}
__global__ void setup_kernel(curandState *state, unsigned long seed)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x; //��ȡ�̺߳�0~blocks*THREAD_NUM-1
	curand_init(seed, tid, 0, &state[tid]);// initialize the state
}
__global__ void generate_rd1_rd2(float *rd1, float *rd2, curandState *globalState)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x; //��ȡ�̺߳�0~blocks*THREAD_NUM-1
	curandState localState = globalState[tid];
	rd1[tid] = curand_uniform(&localState);
	rd2[tid] = curand_uniform(&localState);
}
__global__ void stop(float *y, int *is_stop, int it)
{
	*is_stop = 0;
	if (y[it] == y[it - STOPNUM])
		*is_stop = 1;
}

void Hopfield_syn_cuda(float *y,float *gbestx)
{
	//ѡ���Կ�,ubuntu�û���3��2080ti�Կ������ϴ�ѧ��2��rtx8000�Կ�
	cudaSetDevice(1);

	//�������ʱ�䣬��ʱ����
	/*clock_t start, finish;
	double  duration;
	start = clock();
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("%f seconds\n", duration);
	*/

	//��ʼ��x
	//����x��xt���ڴ����Դ���� 
	float *xh, *xth, *xd, *xtd;
	xh = (float*)malloc(sizeof(float) * SIZE); //x���ڴ�����������ڴ�ռ�
	xth = (float*)malloc(sizeof(float)*SIZE); //xt���ڴ�����������ڴ�ռ�
	initial_x(xh);//printf("\n");print_matrix(xh, N, P*POP);//��ʼ��x
	cudaMalloc((void**)&xd, sizeof(float) * SIZE); //x���Դ�����������Դ�ռ�
	cudaMalloc((void**)&xtd, sizeof(float) * SIZE); //xtd���Դ�����������Դ�ռ�
	cudaMemcpy(xd, xh, sizeof(int) * SIZE, cudaMemcpyHostToDevice); //�ڴ��е�x���Ƶ��Դ�

	//���Կ����㣬��ʱ����#pragma omp parallel for num_threads(3)


	//����x��xt�Ƿ���ȷ����ʱ����
	/*print_matrix(xh, N, P*POP);
	cudaMemcpy(xth, xtd, sizeof(int)*SIZE, cudaMemcpyDeviceToHost); //�Դ����ݿ������ڴ�
	print_matrix(xth, N* POP, P);*/

	//����obj,index,I1��I2��s1��s2��t��d���ڴ����Դ���� 
	int *indexd, *indexh;
	float *sd1, *sd2, *sh1, *sh2, *Id1, *Id2, *Ih1, *Ih2, *td, *th, *dd, *dh;
	float *objd, *objh;
	float *initialxh, *initialxd, *initialvh, *initialvd;
	initialxh = (float*)malloc(sizeof(float)*SIZE);
	initialvh = (float*)malloc(sizeof(float)*SIZE); initial_v(initialvh);
	cudaMalloc((void**)&initialvd, sizeof(float) * SIZE);
	indexh = (int*)malloc(sizeof(int) *SIZE); generate_batch(indexh);//index���ڴ�����������ڴ�ռ�
	Ih1 = (float*)malloc(sizeof(float)  * POP*P*POP); generate_I1(Ih1);//I1���ڴ�����������ڴ�ռ�
	Ih2 = (float*)malloc(sizeof(float) * N); generate_I2(Ih2, N);//I2���ڴ�����������ڴ�ռ�
	sh1 = (float*)malloc(sizeof(float) * N*POP);//s1���ڴ�����������ڴ�ռ�
	sh2 = (float*)malloc(sizeof(float) * P*POP);//s2���ڴ�����������ڴ�ռ�
	th = (float*)malloc(sizeof(float) * SIZE);//t���ڴ�����������ڴ�ռ�
	dh = (float*)malloc(sizeof(float) * N*N); generate_d(dh, data);//d���ڴ�����������ڴ�ռ�
	objh = (float*)malloc(sizeof(float) * POP); initial_obj(objh);//obj���ڴ�����������ڴ�ռ�

	cudaMalloc((void**)&indexd, sizeof(int) * SIZE);//I1���ڴ�����������Դ�ռ�
	cudaMalloc((void**)&Id1, sizeof(float) * POP*P*POP);//I1���ڴ�����������Դ�ռ�
	cudaMalloc((void**)&Id2, sizeof(float) * N);//I2���Դ�����������Դ�ռ�
	cudaMalloc((void**)&sd1, sizeof(float) * N*POP);//I1���ڴ�����������Դ�ռ�
	cudaMalloc((void**)&sd2, sizeof(float) * P*POP);//I2���Դ�����������Դ�ռ�
	cudaMalloc((void**)&td, sizeof(float) * SIZE);//t���Դ�����������Դ�ռ�
	cudaMalloc((void**)&dd, sizeof(float) * N*N);//d���Դ�����������Դ�ռ�
	cudaMalloc((void**)&objd, sizeof(float) * POP);//obj���Դ�����������Դ�ռ�
	cudaMalloc((void**)&initialxd, sizeof(float) * SIZE);
	cudaMemcpy(initialxd, xh, sizeof(float) * SIZE, cudaMemcpyHostToDevice); //�ڴ��е�x���Ƶ��Դ�
	cudaMemcpy(initialvd, initialvh, sizeof(float) * SIZE, cudaMemcpyHostToDevice); //�ڴ��е�x���Ƶ��Դ�

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

	float *rdh1, *rdh2;
	float *rdd1, *rdd2;
	rdh1 = (float*)malloc(sizeof(float) *POP);
	rdh2 = (float*)malloc(sizeof(float) *POP);
	cudaMalloc((void**)&rdd1, sizeof(float) * POP);
	cudaMalloc((void**)&rdd2, sizeof(float) * POP);
	cublasHandle_t handle;
	cublasCreate(&handle);
	srand((unsigned int)time(NULL));
	curandState* devStates;

	cudaMalloc(&devStates, POP * sizeof(curandState));

	srand(time(0));
	int gen;
	int is_stoph, *is_stopd, stop_number = STOPNUM;
	cudaMalloc((void**)&is_stopd, sizeof(int));
	for (gen = 0; gen < 500; gen++)
	{



		generate_s1(handle, sd1, initialxd, Id1);
		generate_s2(handle, sd2, initialxd, Id2);
		generate_t(handle, td, dd, initialxd);
		//�������ӹ��࣬�����Ż���
		setup_kernel << <BLOCKS, THREAD_NUM >> > (devStates, rand());
		parallel_updated_GPU << <BLOCKS, THREAD_NUM >> > (objd, lbestd, lbestxd, initialxd, xd, dd, sd1, sd2, td, indexd, gen, devStates);//����ÿ��
		generate_global_best << <1, 1, 0 >> > (yd, gbestd, gbestxd, lbestd, lbestxd, gen);
		generate_rd1_rd2 << <RD_BLOCKS, RD_THREAD_NUM >> > (rdd1, rdd2, devStates);
		//cudaMemcpy(yh, yd, sizeof(float)*MAXITER, cudaMemcpyDeviceToHost);
		pso << <PSO_BLOCKS, PSO_THREAD_NUM >> > (xd, initialxd, initialvd, gbestxd, lbestxd, rdd1, rdd2, rand());//����ÿ��
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

	cudaMemcpy(gbestxh, gbestxd, sizeof(float) * N*P, cudaMemcpyDeviceToHost);
	cudaMemcpy(yh, yd, sizeof(float)*MAXITER, cudaMemcpyDeviceToHost);





	//�ͷ��ڴ�
	free(indexh);
	free(xh); free(xth);
	free(sh1); free(sh2);
	free(Ih1); free(Ih2);
	free(th); free(dh);
	//�ͷ��Դ�
	cudaFree(indexd);
	cudaFree(xd); cudaFree(xtd);
	cudaFree(sd1); cudaFree(sd2);
	cudaFree(Id1); cudaFree(Id2);
	cudaFree(td); cudaFree(dd);
	//�ͷž��
	cublasDestroy(handle);

}