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

	/* Filling Methods */
	void Random();
	void Constant(float val);
	void Uniform(float min, float max);

	/* Transpose */
	void T_();	// transpose in-place
	Matrix T(); // transpose and return

	// set points > 0.5 to 1 else 0;
	void bin_();
	Matrix bin() const;

	/* Sum */
	float sum() const;

	/* Power Function */
	void pow_(float val);
	Matrix pow(float val) const;

	/* Matrix Multiplication */
	void dot_(const Matrix &val);		 // matrix mult in-place
	Matrix dot(const Matrix &val) const; // matrix mult and return

	/*
	 *
	 * Special Math Functions
	 *
	 *
	 */

	/*
	void Log();
	void Exp();
	void Tanh();
	void Sigmoid();
	void Elu(float alph = 1.0);
	void Sign(float alph = 0.0);
	void Relu(float alph = 0.0); */

	/*
	 *
	 * These methods
	 * Modified the current object
	 * Ex: X += 1.0; X += Y;
	 *
	 * */
	/*
	void operator*=(float val);
	void operator+=(float val);
	void operator-=(float val);
	void operator/=(float val);

	void operator=(const Matrix &val);
	void operator*=(const Matrix &val);
	void operator+=(const Matrix &val);
	void operator-=(const Matrix &val);
	void operator/=(const Matrix &val);*/

	/*
	 *
	 * These methods
	 * Return a newly created result
	 * Ex: X = X + X; X = 1 + X;
	 *
	 * */
	/*
	Matrix operator-(float val) const;
	Matrix operator+(float val) const;
	Matrix operator/(float val) const;
	Matrix operator*(float val) const;

	Matrix operator+(const Matrix &val) const;
	Matrix operator-(const Matrix &val) const;
	Matrix operator/(const Matrix &val) const;
	Matrix operator*(const Matrix &val) const; */

	/* Return array on the gpu */
	float *DevData() const { return dev_mat; }

	/* Return Eigen::MatrixXf */
	const Tensor2d &HostData() const { return host_mat; }

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

	Tensor2d host_mat;
	float *dev_mat;
	size_t rows = 1, cols = 1;
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
			cudaMemcpy(dev_mat, host_mat.data(), bytes(), cudaMemcpyHostToDevice);
		}
		else
		{
			allocDevice(host_mat.data());
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

/*
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
*/

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
	host_mat = Tensor2d::Zero(rows, cols);
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
	host_mat = val.host_mat;
	allocDevice(val.dev_mat);
}

Matrix::Matrix(size_t r, size_t c,
			   const float *arr)
{
	rows = r;
	cols = c;
	host_mat = Eigen::Map<const Tensor2d>(arr, rows, cols);
	ToDevice();
}

Matrix::Matrix(size_t r, size_t c,
			   const std::vector<float> &arr)
{
	rows = r;
	cols = c;
	host_mat = Eigen::Map<const Tensor2d>(arr.data(), rows, cols);
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
	assert(!cuda);

	host_mat = Tensor2d::Zero(rows, cols);

	cudaMemcpy(host_mat.data(), dev_mat, bytes(),
			   cudaMemcpyDeviceToHost);
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
	host_mat = Tensor2d::Random(rows, cols);
	allocDevice(host_mat.data());
}

void Matrix::Constant(float val)
{
	fill_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val, this->size());
	cudaDeviceSynchronize();
}

void Matrix::Uniform(float min, float max)
{
	host_mat = Tensor2d::NullaryExpr(rows, cols, [&, this]()
									 { return this->randint(min, max); });
	allocDevice(host_mat.data());
}

/*
 *
 *
 * -------------- Transpose Methods --------------
 *
 *
 *  */

/* Transpose In Place */
void Matrix::T_()
{

	float *new_mat;
	cudaMalloc(&new_mat, bytes());
	cudaMemset(new_mat, 0.0, bytes());

	cublas_transpose(dev_mat, new_mat, rows, cols, handle);
	cudaDeviceSynchronize();

	cudaFree(dev_mat);
	dev_mat = new_mat;

	std::swap(rows, cols);
}

/* Transpose */
Matrix Matrix::T()
{
	Matrix item(cols, rows);

	cublas_transpose(dev_mat, item.dev_mat, rows, cols, item.handle);
	cudaDeviceSynchronize();

	return item;
}

/*
 *
 *
 * -------------- Basic Math Methods --------------
 *
 *
 *  */

float Matrix::sum() const
{
	float mat_sum = 0, *d_ones;
	Tensor2d ones = Tensor2d::Constant(rows, cols, 1.0);

	cudaMalloc(&d_ones, bytes());
	cudaMemcpy(d_ones, ones.data(), bytes(), cudaMemcpyHostToDevice);

	cublasAssert(cublasSdot(handle, size(), dev_mat, 1, d_ones, 1, &mat_sum));

	cudaFree(d_ones);
	return mat_sum;
}

void Matrix::bin_()
{
	ToHost();
	host_mat = host_mat.NullaryExpr([this](float x)
									{ if (x < 0.5f) return 0.0f;
								     return 1.0f; });
	ToDevice();
}

Matrix Matrix::bin() const
{
	Matrix item(rows, cols);

	float threshold = this->sum() / size() + 0.07f;

	cudaFree(item.dev_mat);
	cudaMalloc(&item.dev_mat, bytes());
	cudaMemcpy(item.dev_mat, dev_mat, bytes(), cudaMemcpyDeviceToDevice);

	bin_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, threshold, this->size());
	cudaDeviceSynchronize();

	return item;
}

void Matrix::pow_(float val)
{
	pow_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val, this->size());
	cudaDeviceSynchronize();
}

Matrix Matrix::pow(float val) const
{
	Matrix item(rows, cols);

	cudaFree(item.dev_mat);
	cudaMalloc(&item.dev_mat, bytes());
	cudaMemcpy(item.dev_mat, dev_mat, bytes(), cudaMemcpyDeviceToDevice);
	pow_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, val, this->size());
	cudaDeviceSynchronize();

	return item;
}

void Matrix::dot_(const Matrix &val)
{
	assert(cols == val.rows);

	cols = val.cols;

	float *new_mat;
	cudaMalloc(&new_mat, bytes());
	cudaMemset(new_mat, 0, bytes());

	cublas_mat_mult(dev_mat, val.dev_mat, new_mat, rows, val.rows, val.cols, handle);
	cudaDeviceSynchronize();

	cudaFree(dev_mat);
	dev_mat = new_mat;
}

Matrix Matrix::dot(const Matrix &val) const
{
	assert(cols == val.rows);

	Matrix item(rows, val.cols);

	cublas_mat_mult(dev_mat, val.dev_mat, item.dev_mat, rows, val.rows, val.cols, item.handle);
	cudaDeviceSynchronize();

	return item;
}

#endif
