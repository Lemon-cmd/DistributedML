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

    /* GPU/CPU utils */
    void ToHost();
    void ToDevice();

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

    float *dev_mat = NULL;
    Tensor2d host_mat;

    cublasHandle_t handle;
    size_t rows = 1, cols = 1;

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
    host_mat = Tensor2d::Zero(rows, cols);

    if (dev_mat == NULL)
        std::cout << "hi\n";
}

Matrix::Matrix(const Matrix &val) {}
Matrix::Matrix(size_t r, size_t c) {}
Matrix::Matrix(const Shape &shape) {}
Matrix::Matrix(size_t r, size_t c, const float *arr) {}
Matrix::Matrix(size_t r, size_t c, const std::vector<float> &arr) {}

/* GPU/CPU utils */
void Matrix::ToHost() {}
void Matrix::ToDevice() {}

/* Filling Methods */
void Matrix::Random() {}
void Matrix::Constant(float val) {}
void Matrix::Uniform(float min, float max) {}

#endif