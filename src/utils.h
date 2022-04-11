#ifndef __UTILS__
#define __UTILS__

#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 1024

template <typename T>
__global__ void rand_arr(T *arr, const size_t size, curandState *states)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	curand_init(seed, idx, 0, &states[id]);

	for (uint j = idx; j < size; j += stride)
	{
		arr[j] = curand_uniform (&states[id]);
	}
}

template <typename T>
__global__ void mult_arr(T *A, const T* B, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		A[j] = A[j] * B[j];
	}
}

template <typename T>
__global__ void mult_arr_val(T *A, const T* val, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		A[j] = A[j] * val;
	}
}

template <typename T>
__global__ void add_arr(T *A, const T* B, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		A[j] = A[j] + B[j];
	}
}

template <typename T>
__global__ void add_arr_val(T *A, const T* val, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		A[j] = A[j] + val;
	}
}

template <typename T>
__global__ void minus_arr(T *A, const T* B, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		A[j] = A[j] - B[j];
	}
}

template <typename T>
__global__ void minus_arr_val(T *A, const T* val, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		A[j] = A[j] - val;
	}
}


template <typename T>
__global__ void div_arr(T *A, const T* B, const size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (uint j = idx; j < size; j += stride)
	{
		A[j] = A[j] / B[j];
	}
}

#endif
