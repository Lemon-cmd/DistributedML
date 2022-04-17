#ifndef __UTILS__
#define __UTILS__

#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 1024

template <typename T>
using func_t = T (*)(T);

template <typename T>
using func_alph = T (*)(T, T);

template <typename T>
__device__ T cudaLog(T x)
{
	if (x > 0.0)
		return log(x);
	return 0.0;
}

template <typename T>
__device__ T cudaExp(T x)
{
	return exp(x);
}

template <typename T>
__device__ T cudaSigmoid(T x)
{
	return 1.0 / (1.0 + exp(-x));
}

template <typename T>
__device__ T cudaTanh(T x)
{
	return tanh(x);
}

template <typename T>
__device__ T cudaReLU(T x, T y = 0.0)
{
	if (x < 0.0)
		return y;
	return x;
}

template <typename T>
__device__ T cudaSign(T x, T y = 0.0)
{
	if (x < 0.0)
		return y;
	return 1.0;
}

template <typename T>
__device__ T cudaELU(T x, T y = 1.0)
{
	if (x < 0.0)
		return y * (exp(x) - 1.0);
	return x;
}

/* Methods with no alpha variable */
template <typename T>
__device__ func_t<T> p_log = cudaLog<T>;

template <typename T>
__device__ func_t<T> p_exp = cudaExp<T>;

template <typename T>
__device__ func_t<T> p_tanh = cudaTanh<T>;

template <typename T>
__device__ func_t<T> p_sigmoid = cudaSigmoid<T>;

/* Methods with alpha variable */
template <typename T>
__device__ func_alph<T> p_elu = cudaELU<T>;

template <typename T>
__device__ func_alph<T> p_relu = cudaReLU<T>;

template <typename T>
__device__ func_alph<T> p_sign = cudaSign<T>;

template <typename T>
__global__ void apply_non_alph(T *arr, func_t<T> op, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		arr[j] = (*op)(arr[j]);
	}
}

template <typename T>
__global__ void apply_alph(T *arr, func_alph<T> op, const T alph, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		arr[j] = (*op)(arr[j], alph);
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
