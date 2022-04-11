#include "modules.h"
#include "dense.h"

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

void gpu_blas_mmul(const float *A, const float *B, float *C,
			   const int m, const int k, const int n, cublasHandle_t& handle)
{

	int lda = m, ldb = k, ldc = m;
	const float alpha = 1, beta = 0;

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
				m, n, k,
				&alpha, A, lda,
				B, ldb, &beta,
				C, ldc);
}

__global__ void add_arrs(const float *A, const float *B, float *C, size_t size)
{
	const uint stride = blockDim.x * gridDim.x;
    const uint tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	for (uint j = tid; j < size; j += stride) 
	{
		C[j] = A[j] + B[j];
	}
}



int main()
{
	cublasHandle_t handle;
	cublasCreate(&handle);


	Eigen::MatrixXf X = Eigen::MatrixXf::Random(2, 2), Y = Eigen::MatrixXf::Random(2, 1);

	std::cout << X << "\n\n" << Y << std::endl;

	float *x, *y, *r, *o;

	cudaMalloc(&x, X.size() * sizeof(float));
	cudaMalloc(&y, Y.size() * sizeof(float));
	
	cudaMemcpy(x, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(y, Y.data(), Y.size() * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&r, X.rows() * Y.cols() * sizeof(float));

	//add_arrs <<<1, 1>>> (x, y, r, X.size());
	//cudaDeviceSynchronize();

	gpu_blas_mmul(x, y, r, X.rows(), X.cols(), Y.cols(), handle);
	cudaDeviceSynchronize();
	
	o = (float*) calloc(X.rows() * Y.cols(), sizeof(float));
	cudaMemcpy(o, r, X.rows() * Y.cols() * sizeof(float), cudaMemcpyDeviceToHost);

	Eigen::MatrixXf R = Eigen::Map <Eigen::MatrixXf> (o, X.rows(), Y.cols());
	std::cout << '\n' <<  R << '\n';

	cudaFree(x);
	cudaFree(y);
	cudaFree(r);
	free(o);
	cublasDestroy(handle);	
}
