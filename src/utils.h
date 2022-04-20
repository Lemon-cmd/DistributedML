#ifndef __UTILS__
#define __UTILS__

#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 1024

#define cudaAssert(ans)                       \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__); \
	}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

#define cublasAssert(ans)                      \
	{                                          \
		blasAssert((ans), __FILE__, __LINE__); \
	}
inline void blasAssert(cublasStatus_t code, const char *file, int line, bool abort = true)
{
	if (code != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "cuBlas-assert: %s %s %d\n", code, file, line);
		if (abort)
			exit(code);
	}
}

void cublas_transpose(const float *A, float *B,
					  int m, int n, cublasHandle_t &handle)
{
	static const float alpha = 1.0, beta = 0.0;

	cublasAssert(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
							 n, m, &alpha, A,
							 m, &beta, A,
							 n, B, n));
}

void cublas_mat_mult(const float *A, const float *B, float *C,
					 int m, int k, int n, cublasHandle_t &handle)
{

	int lda = m, ldb = k, ldc = m;
	static const float alpha = 1, beta = 0;

	cublasAssert(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
							 m, n, k,
							 &alpha, A, lda,
							 B, ldb, &beta,
							 C, ldc));
}

template <typename T>
__global__ void exp_arr(T *A, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		A[j] = exp(A[j]);
	}
}

template <typename T>
__global__ void log_arr(T *A, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		if (A[j] > 0.0)
			A[j] = log(A[j]);
		else
			A[j] = 1e-8;
	}
}

template <typename T>
__global__ void sigmoid_arr(T *A, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		A[j] = 1.0 / (1.0 + exp(-A[j]));
	}
}

template <typename T>
__global__ void tanh_arr(T *A, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		A[j] = tanh(A[j]);
	}
}

template <typename T>
__global__ void relu_arr(T *A, const T alph, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		if (A[j] < 0.0)
			A[j] = alph * A[j];
	}
}

template <typename T>
__global__ void sign_arr(T *A, const T alph, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		if (A[j] < 0.0)
			A[j] = alph;
		else
			A[j] = 1.0;
	}
}

template <typename T>
__global__ void elu_arr(T *A, const T alph, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		if (A[j] < 0.0)
			A[j] = alph * (exp(A[j] - 1.0));
	}
}

template <typename T>
__global__ void bin_arr(T *arr, const float threshold, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		if (arr[j] >= threshold)
			arr[j] = 1.0;
		else
			arr[j] = 0.0;
	}
}

template <typename T>
__global__ void pow_arr(T *arr, const T power, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		arr[j] = pow(arr[j], power);
	}
}

template <typename T>
__global__ void fill_arr(T *arr, const T val, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		arr[j] = val;
	}
}

template <typename T>
__global__ void mult_arr(T *A, const T *B, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		A[j] = A[j] * B[j];
	}
}

template <typename T>
__global__ void mult_arr_val(T *A, const T val, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		A[j] = A[j] * val;
	}
}

template <typename T>
__global__ void add_arr(T *A, const T *B, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		A[j] = A[j] + B[j];
	}
}

template <typename T>
__global__ void add_arr_val(T *A, const T val, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		A[j] = A[j] + val;
	}
}

template <typename T>
__global__ void minus_arr(T *A, const T *B, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		A[j] = A[j] - B[j];
	}
}

template <typename T>
__global__ void minus_arr_val(T *A, const T val, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		A[j] = A[j] - val;
	}
}

template <typename T>
__global__ void div_arr(T *A, const T *B, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		A[j] = A[j] / B[j];
	}
}

template <typename T>
__global__ void div_arr_val(T *A, const T val, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		A[j] = A[j] / val;
	}
}

#endif
