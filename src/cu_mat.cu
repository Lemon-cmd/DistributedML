#ifndef __CUDA_MATRIX__
#define __CUDA_MATRIX__

#include "utils.h"
#include "modules.h"

typedef Eigen::MatrixXf Tensor2d;

class CuMatrix
{
	public:
		CuMatrix(size_t r); 
		CuMatrix(size_t r, size_t c);
 		
		~CuMatrix() 
		{ 
			cudaFree(dev_mat); 
			cudaFree(cuStates);
		}
		
		size_t size() { return rows * cols; }

		void cuda();

		void Random();
		void Constant(float val);
		void Uniform(float min, float max);

		void ToHost();
		void ToDevice();
		
		void dot(const Matrix &val) const;
		
		void operator*=(const Matrix &val) const;
		void operator+=(const Matrix &val) const;
		void operator-=(const Matrix &val) const;
		void operator/=(const Matrix &val) const;

	private:
		bool cuda = false;

		float *dev_mat;
		size_t rows, cols;

		Tensor2d mat; 
		curandState *cuStates;

		void allocDevice()
		{
			if (cuda) cudaFree(dev_mat);
			
			cudaMalloc(&dev_mat, size() * sizeof(float));
			cudaMemcpy(dev_mat, mat.data(), size() * sizeof(float), 
					   cudaMemcpyHostToDevice);
		}
};

CuMatrix::CuMatrix(size_t r) { CuMatrix(r, 1); }

CuMatrix::CuMatrix(size_t r, size_t c) : rows(r), cols(c)
{
	mat = Tensor2d::Zero(rows, cols);
}

void CuMatrix::Random()
{
	if (!cuda) 
	{
		mat = Tensor2d::Random(rows, cols);
	} else {
		rand_arr<float> <<<rows * cols / BLOCK_SIZE, BLOCK_SIZE>>> (dev_mat, this->size(), cuStates);	
	}
}

#endif
