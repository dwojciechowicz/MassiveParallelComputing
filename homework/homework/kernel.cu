#include <cstdio>
#include <Windows.h>
#include <WinBase.h>
#include <stdlib.h>

#if defined(NDEBUG)
#define CUDA_CHECK(x)	(x)
#else
#define CUDA_CHECK(x)	do {\
		(x); \
		cudaError_t e = cudaGetLastError(); \
		if (cudaSuccess != e) { \
			printf("cuda failure \"%s\" at %s:%d\n", \
			       cudaGetErrorString(e), \
			       __FILE__, __LINE__); \
			exit(1); \
		} \
	} while (0)
#endif
//data generator
void generateData(float* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (float)(rand() % 1000) / 1000.0F;
	}
}

// kernel program
__global__ void mulKernel(float* p, const float* m, const float* n, const int WIDTH) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = y * WIDTH + x;
	float sum = 0.0f;
	for (int k = 0; k < WIDTH; ++k) {
		sum += m[y * WIDTH + k] * n[k * WIDTH + x];
	}
	p[i] = sum;
}


int main(void) {
	// host-side data
	long long start, end, f;
	QueryPerformanceFrequency((LARGE_INTEGER*)(&f));
	const int WIDTH = 4096;
	const int TILE_WIDTH = 32;
	float* M=NULL;
	float* N=NULL;
	//float* P=NULL;
	float (*P)[WIDTH];

	M = (float*)malloc(WIDTH * WIDTH * sizeof(float));
	N = (float*)malloc(WIDTH * WIDTH * sizeof(float));
	P = (float(*)[WIDTH])malloc(WIDTH * WIDTH * sizeof(float));
	//P = (float*)malloc(WIDTH * WIDTH * sizeof(float));

	generateData(M, WIDTH * WIDTH);
	generateData(N, WIDTH * WIDTH);

	// device-side data
	float* dev_m = 0;
	float* dev_n = 0;
	float* dev_p = 0;
	
	CUDA_CHECK(cudaMalloc((void**)&dev_m, WIDTH * WIDTH * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&dev_n, WIDTH * WIDTH * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&dev_p, WIDTH * WIDTH * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(dev_m, M, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_n, N, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice));
	
	dim3 dimGrid(WIDTH/TILE_WIDTH, WIDTH/TILE_WIDTH, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	QueryPerformanceCounter((LARGE_INTEGER*)(&start));//I am interested only in the time of CUDA kernel function execution
	mulKernel <<< dimGrid, dimBlock >>> (dev_p, dev_m, dev_n, WIDTH);
	QueryPerformanceCounter((LARGE_INTEGER*)(&end));
	CUDA_CHECK(cudaPeekAtLastError());
	CUDA_CHECK(cudaMemcpy(P, dev_p, WIDTH * WIDTH * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(dev_m));
	CUDA_CHECK(cudaFree(dev_n));
	CUDA_CHECK(cudaFree(dev_p));

	//printf("%5f\n", P[0]);
	//printf("%5f\n", P[123 * WIDTH + 456]);
	//printf("%5f\n", P[WIDTH * WIDTH - 1]);
	printf("%5f\n", P[0][0]);
	printf("%5f\n", P[123][456]);
	printf("%5f\n", P[WIDTH-1][WIDTH-1]);
	printf("\nelapsed time = %f usec\n", (double)(end - start) * 1000000.0 / (double)(f));
	free(M);
	free(N);
	free(P);
	return 0;
}

