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
        cudaAssert(cudaFree(dev_mat));
        cublasAssert(cublasDestroy(handle));
    }

    /* GPU/CPU utils */
    void ToHost();

    /* Filling Methods */
    void Random();
    void Constant(float val);
    void Uniform(float min, float max);

private:
    bool cuda = false;

    float *dev_mat;
    Tensor2d host_mat;

    cublasHandle_t handle;
    size_t rows = 1, cols = 1;

    func_t<float> cu_log, cu_exp, cu_tanh,
        cu_sigmoid;

    func_alph<float> cu_elu, cu_sign, cu_relu;

    float randint(float min, float max) const
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(min, max);
        return dist(gen);
    }

    void allocDevice(const float *val, bool dev_transfer = false)
    {
        cudaAssert(cudaFree(dev_mat));
        cudaAssert(cudaMalloc((void **)&dev_mat, bytes()));
        cudaAssert(cudaMemset(dev_mat, 0, bytes()));

        if (!dev_transfer)
            cudaMemcpy(dev_mat, val, bytes(), cudaMemcpyHostToDevice);
        else
            cudaMemcpy(dev_mat, val, bytes(), cudaMemcpyDeviceToDevice);
    }

    void ToDevice()
    {
        if (!cuda)
        {
            cuda = true;
            allocDevFuncs();
            cublasAssert(cublasCreate(&handle));
            cudaAssert(cudaMalloc((void **)&dev_mat, bytes()));
            cudaMemcpy(dev_mat, host_mat.data(), bytes(), cudaMemcpyHostToDevice);
        }
        else
        {
            allocDevice(host_mat.data());
        }
    }

    void allocDevFuncs()
    {
        cudaAssert(cudaMemcpyFromSymbol(&cu_elu, p_elu<float>, sizeof(func_alph<float>)));
        cudaAssert(cudaMemcpyFromSymbol(&cu_relu, p_relu<float>, sizeof(func_alph<float>)));
        cudaAssert(cudaMemcpyFromSymbol(&cu_sign, p_sign<float>, sizeof(func_alph<float>)));

        cudaAssert(cudaMemcpyFromSymbol(&cu_log, p_log<float>, sizeof(func_t<float>)));
        cudaAssert(cudaMemcpyFromSymbol(&cu_exp, p_exp<float>, sizeof(func_t<float>)));
        cudaAssert(cudaMemcpyFromSymbol(&cu_tanh, p_tanh<float>, sizeof(func_t<float>)));
        cudaAssert(cudaMemcpyFromSymbol(&cu_sigmoid, p_sigmoid<float>, sizeof(func_t<float>)));
    }
};

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
    host_mat = Tensor2d::Constant(rows, cols, 0.0);
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
    allocDevice(val.dev_mat, true);
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
    cudaAssert(cudaDeviceSynchronize());
}

void Matrix::Uniform(float min, float max)
{
    host_mat = Tensor2d::NullaryExpr(rows, cols, [&, this]()
                                     { return this->randint(min, max); });
    allocDevice(host_mat.data());
}

#endif
