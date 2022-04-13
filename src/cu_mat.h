#ifndef __CUDA_MATRIX__
#define __CUDA_MATRIX__

#include "utils.h"
#include "modules.h"

typedef Eigen::MatrixXf Tensor2d;
typedef std::pair<size_t, size_t> Shape;

class Matrix
{
public:
	/* Constructors */
	Matrix();
	Matrix(size_t r);
	Matrix(const Matrix &val);
	Matrix(size_t r, size_t c);

	Matrix(const std::vector<float> &arr);
	Matrix(size_t r, size_t c, const std::vector<float> &arr);

	/* Destructor */
	~Matrix()
	{
		deallocDevMat();
		if (cuda)
			cublasDestroy(handle);
	}

	/* Disable GPU and move back to host completely */
	void cpu();

	/* GPU/CPU utils */
	void ToHost();
	void ToDevice();

	/* Filling Methods */
	void Random();
	void Constant(float val);
	void Uniform(float min, float max);

	/* Transpose */
	void T();			// transpose in-place
	Matrix transpose(); // transpose and return

	/* Sum */
	float sum() const;

	/* Power Function */
	void pow(float val);
	Matrix power(float val) const;

	/* Matrix Multiplication */
	void dot(const Matrix &val);			   // matrix mult in-place
	Matrix operator%(const Matrix &val) const; // matrix mult and return

	/*
	 *
	 * Special Math Functions
	 *
	 *
	 */

	void Exp();
	void Tanh();
	void Sigmoid();
	void Softmax();
	void Elu(float alph = 1.0);
	void Sign(float alph = 0.0);
	void Relu(float alph = 0.0);

	/*
	 *
	 * These methods
	 * Modified the current object
	 * Ex: X += 1.0; X += Y;
	 *
	 * */
	void operator*=(float val);
	void operator+=(float val);
	void operator-=(float val);
	void operator/=(float val);

	void operator=(const Matrix &val);
	void operator*=(const Matrix &val);
	void operator+=(const Matrix &val);
	void operator-=(const Matrix &val);
	void operator/=(const Matrix &val);

	/*
	 *
	 * These methods
	 * Return a newly created result
	 * Ex: X = X + X; X = 1 + X;
	 *
	 * */

	Matrix operator-(float val) const;
	Matrix operator+(float val) const;
	Matrix operator/(float val) const;
	Matrix operator*(float val) const;

	Matrix operator+(const Matrix &val) const;
	Matrix operator-(const Matrix &val) const;
	Matrix operator/(const Matrix &val) const;
	Matrix operator*(const Matrix &val) const;

	/* Return array on the gpu */
	float *DevData() const { return dev_mat; }

	/* Return Eigen::MatrixXf */
	Tensor2d HostData() const { return mat; }

	/* Return size of matrix */
	size_t size() const { return rows * cols; }

	/* Return shape of matrix */
	Shape shape() const { return std::make_pair(rows, cols); }

	/* Return bytes w.r.t mat size */
	size_t bytes() const
	{
		return this->size() * sizeof(float);
	}

private:
	bool cuda = false;

	Tensor2d mat;
	float *dev_mat;
	size_t rows, cols;
	cublasHandle_t handle;

	func_t<float> cu_exp, cu_tanh, cu_sigmoid;
	func_alph<float> cu_elu, cu_sign, cu_relu;

	void deallocDevMat()
	{
		if (dev_mat != nullptr)
		{
			cudaFree(dev_mat);
		}
	}

	void allocDevice(const float *val)
	{
		deallocDevMat();
		cudaMalloc(&dev_mat, bytes());
		cudaMemcpy(dev_mat, val, bytes(), cudaMemcpyHostToDevice);
	}

	void allocDevFuncs()
	{
		cudaMemcpyFromSymbol(&cu_elu, p_elu<float>, sizeof(func_alph<float>));
		cudaMemcpyFromSymbol(&cu_relu, p_relu<float>, sizeof(func_alph<float>));
		cudaMemcpyFromSymbol(&cu_sign, p_sign<float>, sizeof(func_alph<float>));

		cudaMemcpyFromSymbol(&cu_exp, p_exp<float>, sizeof(func_t<float>));
		cudaMemcpyFromSymbol(&cu_tanh, p_tanh<float>, sizeof(func_t<float>));
		cudaMemcpyFromSymbol(&cu_sigmoid, p_sigmoid<float>, sizeof(func_t<float>));
	}

	float randint(float min, float max) const
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dist(min, max);
		return dist(gen);
	}
};

/*
 *
 *
 *
 * -------------- Non-class Methods --------------
 *
 *
 *
 *  */

std::ostream &operator<<(std::ostream &stream,
						 const Shape &dim)
{
	stream << '(' << dim.first << ' ' << dim.second << ')' << std::endl;
	return stream;
}

std::ostream &operator<<(std::ostream &stream, const Matrix &matrix)
{
	return stream << matrix.HostData() << std::endl;
}

Matrix operator/(float val, const Matrix &mat)
{
	Matrix item(mat);
	item.pow(-1.0);

	return item * val;
}

Matrix operator-(float val, const Matrix &mat)
{
	return (mat * -1.0) + val;
}

/*
 *
 *
 *
 * -------------- Constructors --------------
 *
 *
 *
 *  */

Matrix::Matrix() { dev_mat = nullptr; }

Matrix::Matrix(size_t r) { Matrix(r, 1); }

Matrix::Matrix(const Matrix &val)
{
	mat = val.mat;
	rows = val.rows, cols = val.cols;

	if (val.cuda)
	{
		cuda = true;
		cublasCreate(&handle);
		cudaMalloc(&dev_mat, bytes());
		cudaMemcpy(dev_mat, val.dev_mat, bytes(), cudaMemcpyDeviceToDevice);
	}
}

Matrix::Matrix(size_t r, size_t c) : rows(r), cols(c)
{
	Matrix();
	mat = Tensor2d::Zero(rows, cols);
}

Matrix::Matrix(const std::vector<float> &arr)
{
	Matrix();
	rows = 1, cols = arr.size();
	mat = Eigen::Map<const Tensor2d>(arr.data(), rows, cols);
}

Matrix::Matrix(size_t r, size_t c, const std::vector<float> &arr)
{
	Matrix();
	rows = r, cols = c;
	mat = Eigen::Map<const Tensor2d>(arr.data(), rows, cols);
}

/*
 *
 *
 *
 * -------------- Device/Host Move Methods --------------
 *
 *
 *
 *  */

void Matrix::cpu()
{
	ToHost();
	cuda = false;
	deallocDevMat();
}

void Matrix::ToHost()
{
	if (cuda)
	{
		mat = Tensor2d::Zero(rows, cols);
		cudaMemcpy(mat.data(), dev_mat, bytes(),
				   cudaMemcpyDeviceToHost);
	}
}

void Matrix::ToDevice()
{
	allocDevice(mat.data());

	if (!cuda)
	{
		cublasCreate(&handle);
	}

	cuda = true;
	allocDevFuncs();
}

/*
 *
 *
 *
 * -------------- Initialize Methods --------------
 *
 *
 *
 *  */

void Matrix::Random()
{
	mat = Tensor2d::Random(rows, cols);
	allocDevice(mat.data());
}

void Matrix::Constant(float val)
{
	if (!cuda)
	{
		mat = Tensor2d::Constant(rows, cols, val);
	}
	else
	{
		fill_arr<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val, this->size());
		cudaDeviceSynchronize();
	}
}

void Matrix::Uniform(float min, float max)
{
	mat = Tensor2d::NullaryExpr(rows, cols, [&, this]()
								{ return this->randint(min, max); });
	allocDevice(mat.data());
}

/*
 *
 *
 * -------------- Transpose Methods --------------
 *
 *
 *  */

/* Transpose In Place */
void Matrix::T()
{

	if (!cuda)
	{
		mat.transposeInPlace();
	}
	else
	{
		float *new_mat;
		cudaMalloc(&new_mat, bytes());
		cudaMemset(new_mat, 0.0, bytes());

		cublas_transpose(dev_mat, new_mat, rows, cols, handle);
		cudaDeviceSynchronize();

		deallocDevMat();
		dev_mat = new_mat;
	}

	std::swap(rows, cols);
}

/* Transpose */
Matrix Matrix::transpose()
{
	Matrix item(cols, rows);

	if (!cuda)
	{
		item.mat = mat.transpose();
	}
	else
	{
		item.cuda = true;
		cublasCreate(&item.handle);
		cudaMalloc(&item.dev_mat, bytes());
		cudaMemset(item.dev_mat, 0.0, bytes());

		cublas_transpose(dev_mat, item.dev_mat, rows, cols, handle);
		cudaDeviceSynchronize();
	}

	return item;
}

/*
 *
 *
 * -------------- Math Methods --------------
 *
 *
 *  */

float Matrix::sum() const
{
	if (!cuda)
	{
		return mat.sum();
	}
	else
	{
		float h_sum;
		cublasAssert(cublasSasum(handle, rows * cols, dev_mat, 1, h_sum));
		return h_sum;
	}
}

void Matrix::pow(float val)
{
	if (!cuda)
	{
		mat = mat.array().pow(val);
	}
	else
	{
		pow_arr<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val, this->size());
		cudaDeviceSynchronize();
	}
}

void Matrix::dot(const Matrix &val)
{
	assert(cols == val.rows);

	cols = val.cols;

	if (!cuda)
	{
		mat = mat * val.mat;
	}
	else
	{
		float *new_mat;
		cudaMalloc(&new_mat, bytes());
		cudaMemset(new_mat, 0.0, bytes());

		cublas_mat_mult(dev_mat, val.dev_mat, new_mat, rows, val.rows, val.cols, handle);
		cudaDeviceSynchronize();

		deallocDevMat();
		dev_mat = new_mat;
	}
}

Matrix Matrix::power(float val) const
{
	Matrix item(rows, cols);

	if (!cuda)
	{
		item.mat = mat.array().pow(val);
	}
	else
	{
		item.ToDevice();
		pow_arr<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, val, this->size());
		cudaDeviceSynchronize();
	}

	return item;
}

Matrix Matrix::operator%(const Matrix &val) const
{
	assert(cols == val.rows);

	Matrix item(rows, val.cols);

	if (!cuda)
	{
		item.mat = mat * val.mat;
	}
	else
	{
		item.ToDevice();
		cublas_mat_mult(dev_mat, val.dev_mat, item.dev_mat, rows, val.rows, val.cols, item.handle);
		cudaDeviceSynchronize();
	}

	return item;
}

/*
 *
 *
 * -------------- Equal Operator --------------
 *
 *
 *  */

void Matrix::operator=(const Matrix &val)
{
	mat = val.mat;
	rows = val.rows;
	cols = val.cols;
	deallocDevMat();

	if (val.cuda)
	{
		cuda = true;
		cudaMalloc(&dev_mat, bytes());
		cudaMemcpy(dev_mat, val.dev_mat, bytes(), cudaMemcpyDeviceToDevice);
	}
}

/*
 *
 *
 *
 * -------------- Single Value Matrix Operators --------------
 *
 *
 *
 *  */

void Matrix::operator+=(float val)
{
	if (!cuda)
	{
		mat = mat.array() + val;
	}
	else
	{
		add_arr_val<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val, this->size());
		cudaDeviceSynchronize();
	}
}

void Matrix::operator-=(float val)
{
	if (!cuda)
	{
		mat = mat.array() - val;
	}
	else
	{
		minus_arr_val<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val, this->size());
		cudaDeviceSynchronize();
	}
}

void Matrix::operator*=(float val)
{
	if (!cuda)
	{
		mat = mat.array() * val;
	}
	else
	{
		mult_arr_val<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val, this->size());
		cudaDeviceSynchronize();
	}
}

void Matrix::operator/=(float val)
{
	if (!cuda)
	{
		mat = mat.array() / val;
	}
	else
	{
		div_arr_val<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val, this->size());
		cudaDeviceSynchronize();
	}
}

Matrix Matrix::operator+(float val) const
{
	Matrix item(rows, cols);

	if (!cuda)
	{
		item.mat = mat.array() + val;
	}
	else
	{
		item.cuda = true;
		cublasCreate(&item.handle);
		cudaMalloc(&item.dev_mat, bytes());
		cudaMemcpy(item.dev_mat, dev_mat, bytes(), cudaMemcpyDeviceToDevice);
		add_arr_val<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, val, this->size());
		cudaDeviceSynchronize();
	}

	return item;
}

Matrix Matrix::operator-(float val) const
{
	Matrix item(rows, cols);

	if (!cuda)
	{
		item.mat = mat.array() - val;
	}
	else
	{
		item.cuda = true;
		cublasCreate(&item.handle);
		cudaMalloc(&item.dev_mat, bytes());
		cudaMemcpy(item.dev_mat, dev_mat, bytes(), cudaMemcpyDeviceToDevice);

		minus_arr_val<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, val, this->size());
		cudaDeviceSynchronize();
	}

	return item;
}

Matrix Matrix::operator*(float val) const
{
	Matrix item(rows, cols);

	if (!cuda)
	{
		item.mat = mat.array() * val;
		return item;
	}
	else
	{
		item.cuda = true;
		cublasCreate(&item.handle);
		cudaMalloc(&item.dev_mat, bytes());
		cudaMemcpy(item.dev_mat, dev_mat, bytes(), cudaMemcpyDeviceToDevice);
		mult_arr_val<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, val, this->size());
		cudaDeviceSynchronize();

		return item;
	}
}

Matrix Matrix::operator/(float val) const
{
	Matrix item(rows, cols);

	if (!cuda)
	{
		item.mat = mat.array() / val;
	}
	else
	{
		item.cuda = true;
		cudaMalloc(&item.dev_mat, bytes());
		cudaMemcpy(item.dev_mat, dev_mat, bytes(), cudaMemcpyDeviceToDevice);
		add_arr_val<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, val, this->size());
		cudaDeviceSynchronize();
	}

	return item;
}

/*
 *
 *
 *
 * -------------- Matrix Operators --------------
 *
 *
 *
 *  */

void Matrix::operator+=(const Matrix &val)
{
	assert(val.rows == rows && val.cols == cols);

	if (!cuda)
	{
		mat = mat.array() + val.mat.array();
	}
	else
	{
		assert(val.cuda);
		add_arr<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val.dev_mat, this->size());
		cudaDeviceSynchronize();
	}
}

void Matrix::operator-=(const Matrix &val)
{
	assert(val.rows == rows && val.cols == cols);

	if (!cuda)
	{
		mat = mat.array() - val.mat.array();
	}
	else
	{
		assert(val.cuda);
		minus_arr<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val.dev_mat, this->size());
		cudaDeviceSynchronize();
	}
}

void Matrix::operator*=(const Matrix &val)
{
	assert(val.rows == rows && val.cols == cols);

	if (!cuda)
	{
		mat = mat.array() * val.mat.array();
	}
	else
	{
		assert(val.cuda);
		mult_arr<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val.dev_mat, this->size());
		cudaDeviceSynchronize();
	}
}

void Matrix::operator/=(const Matrix &val)
{
	assert(val.rows == rows && val.cols == cols);

	if (!cuda)
	{
		mat = mat.array() / val.mat.array();
	}
	else
	{
		assert(val.cuda);
		div_arr<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val.dev_mat, this->size());
		cudaDeviceSynchronize();
	}
}

Matrix Matrix::operator+(const Matrix &val) const
{
	assert(val.rows == rows && val.cols == cols);
	Matrix item(rows, cols);

	if (!cuda)
	{
		item.mat = mat.array() + val.mat.array();
	}
	else
	{
		item.cuda = true;
		cublasCreate(&item.handle);
		cudaMalloc(&item.dev_mat, bytes());
		cudaMemcpy(item.dev_mat, dev_mat, bytes(), cudaMemcpyDeviceToDevice);
		add_arr<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, val.dev_mat, this->size());
		cudaDeviceSynchronize();
	}

	return item;
}

Matrix Matrix::operator-(const Matrix &val) const
{
	assert(val.rows == rows && val.cols == cols);
	Matrix item(rows, cols);

	if (!cuda)
	{
		item.mat = mat.array() - val.mat.array();
	}
	else
	{
		item.cuda = true;
		cublasCreate(&item.handle);
		cudaMalloc(&item.dev_mat, bytes());
		cudaMemcpy(item.dev_mat, dev_mat, bytes(), cudaMemcpyDeviceToDevice);
		minus_arr<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, val.dev_mat, this->size());
		cudaDeviceSynchronize();
	}

	return item;
}

Matrix Matrix::operator*(const Matrix &val) const
{
	assert(val.rows == rows && val.cols == cols);
	Matrix item(rows, cols);

	if (!cuda)
	{
		item.mat = mat.array() * val.mat.array();
	}
	else
	{
		item.cuda = true;
		cublasCreate(&item.handle);
		cudaMalloc(&item.dev_mat, bytes());
		cudaMemcpy(item.dev_mat, dev_mat, bytes(), cudaMemcpyDeviceToDevice);
		mult_arr<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, val.dev_mat, this->size());
		cudaDeviceSynchronize();
	}

	return item;
}

Matrix Matrix::operator/(const Matrix &val) const
{
	assert(val.rows == rows && val.cols == cols);

	Matrix item(rows, cols);

	if (!cuda)
	{
		item.mat = mat.array() - val.mat.array();
	}
	else
	{
		item.cuda = true;
		cublasCreate(&item.handle);
		cudaMalloc(&item.dev_mat, bytes());
		cudaMemcpy(item.dev_mat, dev_mat, bytes(), cudaMemcpyDeviceToDevice);
		div_arr<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, val.dev_mat, this->size());
		cudaDeviceSynchronize();
	}

	return item;
}

/*
 *
 *
 *
 * -------------- Math Methods --------------
 *
 *
 *
 *  */

void Matrix::Exp()
{
	if (!cuda)
	{
		mat = mat.array().exp();
	}
	else
	{
		apply_non_alph<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, cu_exp, this->size());
	}
}

void Matrix::Tanh()
{
	if (!cuda)
	{
		mat = mat.array().tanh();
	}
	else
	{
		apply_non_alph<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, cu_tanh, this->size());
	}
}

void Matrix::Sigmoid()
{
	if (!cuda)
	{
		mat = 1.0 / (1.0 + (-mat).array().exp());
	}
	else
	{
		apply_non_alph<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, cu_sigmoid, this->size());
	}
}

void Matrix::Elu(float alph)
{
	if (!cuda)
	{
		mat = Tensor2d::NullaryExpr([&alph, this](float x)
									{ if (x < 0.0f) return alph * (exp(x) - 1.0f); 
									  return x; });
	}
	else
	{
		apply_alph<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, cu_elu, alph, this->size());
	}
}

void Matrix::Relu(float alph)
{
	if (!cuda)
	{
		mat = Tensor2d::NullaryExpr([&alph, this](float x)
									{ if (x < 0.0) return alph;
									  return x; });
	}
	else
	{
		apply_alph<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, cu_relu, alph, this->size());
	}
}

void Matrix::Sign(float alph)
{
	if (!cuda)
	{
		mat = Tensor2d::NullaryExpr([&alph, this](float x)
									{ if (x < 0.0f) return alph;
									  return 1.0f; });
	}
	else
	{
		apply_alph<float><<<(rows * cols - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, cu_sign, alph, this->size());
	}
}

#endif
