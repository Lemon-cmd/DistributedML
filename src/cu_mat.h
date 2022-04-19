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

	Matrix(const Shape &shape);

	Matrix(size_t r, size_t c,
		   const float *arr);

	Matrix(size_t r, size_t c,
		   const std::vector<float> &arr);

	/* Destructor */
	~Matrix()
	{
		cudaFree(dev_mat);
		cublasDestroy(handle);
	}

	/* GPU/CPU utils */
	void ToHost();

	// set points > 0.5 to 1 else 0;
	Matrix bin() const;

	/* Filling Methods */
	void Random();
	void Constant(float val);
	void Uniform(float min, float max);

	/* Transpose */
	void T_();	// transpose in-place
	Matrix T(); // transpose and return

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

	void Log();
	void Exp();
	void Tanh();
	void Sigmoid();
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
	const Tensor2d &HostData() const
	{
		return mat;
	}

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

	func_t<float> cu_log, cu_exp, cu_tanh,
		cu_sigmoid;

	func_alph<float> cu_elu, cu_sign, cu_relu;

	void allocDevice(const float *val)
	{
		cudaFree(dev_mat);
		cudaMalloc(&dev_mat, bytes());
		cudaMemcpy(dev_mat, val, bytes(), cudaMemcpyHostToDevice);
	}

	void ToDevice()
	{
		if (!cuda)
		{
			cuda = true;
			allocDevFuncs();
			cublasCreate(&handle);
			cudaMalloc(&dev_mat, bytes());
			cudaMemcpy(dev_mat, mata.data(), bytes(), cudaMemcpyHostToDevice);
		}
		else
		{
			allocDevice(mat.data());
		}
	}

	void allocDevFuncs()
	{
		cudaMemcpyFromSymbol(&cu_elu, p_elu<float>, sizeof(func_alph<float>));
		cudaMemcpyFromSymbol(&cu_relu, p_relu<float>, sizeof(func_alph<float>));
		cudaMemcpyFromSymbol(&cu_sign, p_sign<float>, sizeof(func_alph<float>));

		cudaMemcpyFromSymbol(&cu_log, p_log<float>, sizeof(func_t<float>));
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
	item *= val;

	return item;
}

Matrix operator-(float val, const Matrix &mat)
{
	Matrix out = mat * (-1.0);
	out += val;

	return out;
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

Matrix::Matrix()
{
	rows = !(rows > 0) ? 1 : rows;
	cols = !(cols > 0) ? 1 : cols;

	mat = Tensor2d::Zero(rows, cols);
	ToDevice();
}

Matrix::Matrix(size_t r) : rows(r)
{
	Matrix();
}

Matrix::Matrix(size_t r, size_t c) : rows(r), cols(c)
{
	Matrix();
}

Matrix::Matrix(const Shape &shape)
{
	rows = shape.first;
	cols = shape.second;

	Matrix();
}

Matrix::Matrix(const Matrix &val)
{
	rows = val.rows;
	cols = val.cols;

	Matrix();
	mat = val.mat;
	allocDevice(val.dev_mat);
}

Matrix::Matrix(size_t r, size_t c,
			   const float *arr)
{
	rows = r;
	cols = c;
	mat = Eigen::Map<const Tensor2d>(arr, rows, cols);
	ToDevice();
}

Matrix::Matrix(size_t r, size_t c,
			   const std::vector<float> &arr)
{
	rows = r;
	cols = c;
	mat = Eigen::Map<const Tensor2d>(arr.data(), rows, cols);
	ToDevice();
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

void Matrix::ToHost()
{
	assert(cuda);

	mat = Tensor2d::Zero(rows, cols);

	cudaMemcpy(mat.data(), dev_mat, bytes(),
			   cudaMemcpyDeviceToHost);
}

#endif
