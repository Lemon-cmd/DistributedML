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
    Matrix(const Matrix &val);
    Matrix(size_t r, size_t c);
    Matrix(const Shape &shape);
    Matrix(size_t r, size_t c, const float *arr);
    Matrix(size_t r, size_t c, const std::vector<float> &arr);

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

    /* Return array on the gpu */
    float *DevData() const { return dev_mat; }

    /* Return Eigen::MatrixXf */
    const Tensor2d &HostData() const
    {
        return host_mat;
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

    float *dev_mat;
    Tensor2d host_mat;

    cublasHandle_t handle;
    size_t rows = 1, cols = 1;

    func_t<float> cu_log, cu_exp, cu_tanh,
        cu_sigmoid;

    func_alph<float> cu_elu, cu_sign, cu_relu;

    void init_mat()
    {
        host_mat = Tensor2d::Zero(rows, cols);
        ToDevice();
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

    void ModifyDevMat(const float *val, uint type = 0)
    {
        // free Dev Mat
        cudaAssert(cudaFree(dev_mat));

        // host to dev
        if (type == 0)
        {
            cudaAssert(cudaMalloc(&dev_mat, bytes()));
            cudaAssert(cudaMemcpy(dev_mat, val, bytes(), cudaMemcpyHostToDevice));
        }

        // dev to dev
        else if (type == 1)
        {
            cudaAssert(cudaMalloc(&dev_mat, bytes()));
            cudaAssert(cudaMemcpy(dev_mat, val, bytes(), cudaMemcpyDeviceToDevice));
        }
        else
        {
            assert(type != 0 || type != 1);
        }
    }

    void ToDevice()
    {
        if (!cuda)
        {
            cuda = true;
            cublasAssert(cublasCreate(&handle));
            cudaAssert(cudaMalloc(&dev_mat, bytes()));
            cudaAssert(cudaMemcpy(dev_mat, host_mat.data(), bytes(), cudaMemcpyHostToDevice));
        }

        else
        {
            ModifyDevMat(host_mat.data(), 0);
        }
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
    init_mat();
}

Matrix::Matrix(size_t r, size_t c) : rows(r), cols(c)
{
    init_mat();
}

Matrix::Matrix(const Shape &shape)
{
    rows = shape.first, cols = shape.second;
    init_mat();
}

Matrix::Matrix(const Matrix &val)
{
    host_mat = val.host_mat;
    rows = val.rows, cols = val.cols;

    ToDevice();
    ModifyDevMat(val.dev_mat, 1);
}

Matrix::Matrix(size_t r, size_t c, const float *arr)
{
    rows = r, cols = c;
    host_mat = Eigen::Map<const Tensor2d>(arr, rows, cols);
    ToDevice();
}

Matrix::Matrix(size_t r, size_t c, const std::vector<float> &arr)
{
    rows = r, cols = c;
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
    ToDevice();
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
    ToDevice();
}

#endif