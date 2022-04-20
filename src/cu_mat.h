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

    // set points > threshold to 1 else 0;
    void bin_();
    Matrix bin() const;

    /* Transpose */
    void T_();        // transpose in-place
    Matrix T() const; // transpose and return

    /* Sum */
    float sum() const;
    float compare(const Matrix &val) const;

    /* Power Function */
    void pow_(float val);
    Matrix pow(float val) const;

    /* Matrix Multiplication */
    void dot_(const Matrix &val);        // matrix mult in-place
    Matrix dot(const Matrix &val) const; // matrix mult and return

    /*
     *
     * Special Math Functions
     *
     *
     */

    void log_();
    void exp_();
    void tanh_();
    void sigmoid_();
    void elu_(float alph = 1.0);
    void sign_(float alph = 0.0);
    void relu_(float alph = 0.0);

    Matrix log() const;
    Matrix exp() const;
    Matrix tanh() const;
    Matrix sigmoid() const;
    Matrix elu(float alph = 1.0) const;
    Matrix sign(float alph = 0.0) const;
    Matrix relu(float alph = 0.0) const;

    /*
     *
     * In-place Operators
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
     * Non In-place Operators
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

    float *dev_mat;
    Tensor2d host_mat;

    cublasHandle_t handle;
    size_t rows = 1, cols = 1;

    void ModifyDevMat(const float *val, uint type = 0)
    {
        // free Dev Mat
        cudaAssert(cudaFree(dev_mat));
        cudaAssert(cudaMalloc(&dev_mat, bytes()));
        cudaAssert(cudaMemset(dev_mat, 0, bytes()));

        // host to dev
        if (type == 0)
        {
            cudaAssert(cudaMemcpy(dev_mat, val, bytes(), cudaMemcpyHostToDevice));
        }

        // dev to dev
        else if (type == 1)
        {
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
            cudaAssert(cudaMemset(dev_mat, 0, bytes()));
            cudaAssert(cudaMemcpy(dev_mat, host_mat.data(), bytes(), cudaMemcpyHostToDevice));
        }

        else
        {
            ModifyDevMat(host_mat.data(), 0);
        }
    }

    void init_mat()
    {
        host_mat = Tensor2d::Zero(rows, cols);
        ToDevice();
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
    stream << matrix.HostData() << std::endl;
    return stream;
}

Matrix operator/(float val, const Matrix &mat)
{
    Matrix item(mat);
    item.pow_(-1.0);

    return item * val;
}

Matrix operator*(float val, const Matrix &mat)
{
    return mat * val;
}

Matrix operator-(float val, const Matrix &mat)
{
    return (mat * -1.0) + val;
}

Matrix operator+(float val, const Matrix &mat)
{
    return mat + val;
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
    rows = val.rows, cols = val.cols;

    init_mat();
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
    cudaAssert(cudaDeviceSynchronize());
}

void Matrix::Uniform(float min, float max)
{
    host_mat = Tensor2d::NullaryExpr(rows, cols, [&, this]()
                                     { return this->randint(min, max); });
    ToDevice();
}

/*
 *
 *
 *
 * -------------- Math Functions --------------
 *
 *
 *
 *  */

/* Sum */
float Matrix::sum() const
{
    float *d_ones, mat_sum = 0;
    Tensor2d ones = Tensor2d::Constant(rows, cols, 1.0);

    cudaAssert(cudaMalloc(&d_ones, bytes()));

    cudaAssert(cudaMemcpy(d_ones, ones.data(), bytes(),
                          cudaMemcpyHostToDevice));

    cublasAssert(cublasSdot(handle, size(), dev_mat,
                            1, d_ones, 1, &mat_sum));

    cudaAssert(cudaFree(d_ones));

    return mat_sum;
}

/* Return the number of matched items */
float Matrix::compare(const Matrix &val) const
{
    float h_acc = 0, *d_acc;
    cudaAssert(cudaMalloc(&d_acc, sizeof(float)));
    cudaAssert(cudaMemcpy(d_acc, &h_acc, sizeof(float), cudaMemcpyHostToDevice));

    match_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, 1>>>(dev_mat, val.dev_mat, d_acc, this->size());
    cudaAssert(cudaDeviceSynchronize());

    cudaAssert(cudaMemcpy(&h_acc, d_acc, sizeof(float), cudaMemcpyDeviceToHost));
    cudaAssert(cudaFree(d_acc));

    return h_acc;
}

// set points > threshold to 1 else 0;
void Matrix::bin_()
{
    float threshold = this->sum() / size() + 0.07f;
    bin_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, threshold, this->size());
    cudaAssert(cudaDeviceSynchronize());
}

Matrix Matrix::bin() const
{
    Matrix item(*this);

    float threshold = this->sum() / size() + 0.07f;
    bin_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, threshold, this->size());

    return item;
}

// transpose in place
void Matrix::T_()
{
    float *new_mat;
    cudaAssert(cudaMalloc(&new_mat, bytes()));
    cudaAssert(cudaMemset(new_mat, 0.0, bytes()));

    cublas_transpose(dev_mat, new_mat, rows, cols, handle);
    cudaDeviceSynchronize();

    cudaAssert(cudaFree(dev_mat));
    dev_mat = new_mat;

    std::swap(rows, cols);
}

Matrix Matrix::T() const
{
    Matrix item(cols, rows);

    cublas_transpose(dev_mat, item.dev_mat, rows, cols, item.handle);
    cudaDeviceSynchronize();

    return item;
}

/* Power Function */
void Matrix::pow_(float val)
{
    pow_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val, this->size());
    cudaAssert(cudaDeviceSynchronize());
}

Matrix Matrix::pow(float val) const
{
    Matrix item(*this);

    pow_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, val, this->size());
    cudaAssert(cudaDeviceSynchronize());

    return item;
}

/* Matrix Multiplication */
void Matrix::dot_(const Matrix &val)
{
    assert(cols == val.rows);

    cols = val.cols;

    float *new_mat;
    cudaAssert(cudaMalloc(&new_mat, bytes()));
    cudaAssert(cudaMemset(new_mat, 0.0, bytes()));

    cublas_mat_mult(dev_mat, val.dev_mat, new_mat,
                    rows, val.rows, val.cols, handle);

    cudaDeviceSynchronize();

    cudaAssert(cudaFree(dev_mat));
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

/*
 *
 *
 * -------------- In-place Matrix Operators --------------
 *
 *
 *  */

void Matrix::operator=(const Matrix &val)
{
    rows = val.rows, cols = val.cols;
    init_mat();
    ModifyDevMat(val.dev_mat, 1);
}

void Matrix::operator+=(const Matrix &val)
{
    assert(val.rows == rows && val.cols == cols);

    add_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val.dev_mat, this->size());
    cudaAssert(cudaDeviceSynchronize());
}

void Matrix::operator-=(const Matrix &val)
{
    assert(val.rows == rows && val.cols == cols);

    minus_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val.dev_mat, this->size());
    cudaAssert(cudaDeviceSynchronize());
}

void Matrix::operator*=(const Matrix &val)
{
    assert(val.rows == rows && val.cols == cols);

    mult_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val.dev_mat, this->size());
    cudaAssert(cudaDeviceSynchronize());
}

void Matrix::operator/=(const Matrix &val)
{
    div_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val.dev_mat, this->size());
    cudaAssert(cudaDeviceSynchronize());
}

/*
 *
 *
 * -------------- Non- In-place Matrix Operators --------------
 *
 *
 *  */

Matrix Matrix::operator+(const Matrix &val) const
{
    assert(val.rows == rows && val.cols == cols);
    Matrix item(*this);

    add_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, val.dev_mat, this->size());
    cudaAssert(cudaDeviceSynchronize());

    return item;
}

Matrix Matrix::operator-(const Matrix &val) const
{
    assert(val.rows == rows && val.cols == cols);
    Matrix item(*this);

    minus_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, val.dev_mat, this->size());
    cudaAssert(cudaDeviceSynchronize());

    return item;
}

Matrix Matrix::operator*(const Matrix &val) const
{
    assert(val.rows == rows && val.cols == cols);
    Matrix item(*this);

    mult_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, val.dev_mat, this->size());
    cudaAssert(cudaDeviceSynchronize());

    return item;
}

Matrix Matrix::operator/(const Matrix &val) const
{
    assert(val.rows == rows && val.cols == cols);

    Matrix item(*this);

    div_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, val.dev_mat, this->size());
    cudaAssert(cudaDeviceSynchronize());

    return item;
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

    add_arr_val<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val, this->size());
    cudaAssert(cudaDeviceSynchronize());
}

void Matrix::operator-=(float val)
{

    minus_arr_val<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val, this->size());
    cudaAssert(cudaDeviceSynchronize());
}

void Matrix::operator*=(float val)
{

    mult_arr_val<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val, this->size());
    cudaAssert(cudaDeviceSynchronize());
}

void Matrix::operator/=(float val)
{

    div_arr_val<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, val, this->size());
    cudaAssert(cudaDeviceSynchronize());
}

Matrix Matrix::operator+(float val) const
{
    Matrix item(*this);

    add_arr_val<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, val, this->size());
    cudaAssert(cudaDeviceSynchronize());

    return item;
}

Matrix Matrix::operator-(float val) const
{
    Matrix item(*this);

    minus_arr_val<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, val, this->size());
    cudaAssert(cudaDeviceSynchronize());

    return item;
}

Matrix Matrix::operator*(float val) const
{
    Matrix item(*this);

    mult_arr_val<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, val, this->size());
    cudaAssert(cudaDeviceSynchronize());

    return item;
}

Matrix Matrix::operator/(float val) const
{
    Matrix item(*this);

    div_arr_val<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, val, this->size());
    cudaAssert(cudaDeviceSynchronize());

    return item;
}

/*
 *
 * Special Math Functions
 *
 *
 */

void Matrix::log_()
{
    log_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, this->size());
    cudaAssert(cudaDeviceSynchronize());
}

void Matrix::exp_()
{
    exp_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, this->size());
    cudaAssert(cudaDeviceSynchronize());
}

void Matrix::tanh_()
{
    tanh_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, this->size());
    cudaAssert(cudaDeviceSynchronize());
}

void Matrix::sigmoid_()
{
    sigmoid_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, this->size());
    cudaAssert(cudaDeviceSynchronize());
}

void Matrix::elu_(float alph)
{
    elu_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, alph, this->size());
    cudaAssert(cudaDeviceSynchronize());
}

void Matrix::sign_(float alph)
{
    sign_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, alph, this->size());
    cudaAssert(cudaDeviceSynchronize());
}

void Matrix::relu_(float alph)
{
    relu_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_mat, alph, this->size());
    cudaAssert(cudaDeviceSynchronize());
}

Matrix Matrix::log() const
{
    Matrix item(*this);

    log_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, this->size());
    cudaAssert(cudaDeviceSynchronize());

    return item;
}

Matrix Matrix::exp() const
{
    Matrix item(*this);

    exp_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, this->size());
    cudaAssert(cudaDeviceSynchronize());

    return item;
}
Matrix Matrix::tanh() const
{
    Matrix item(*this);

    tanh_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, this->size());
    cudaAssert(cudaDeviceSynchronize());

    return item;
}
Matrix Matrix::sigmoid() const
{
    Matrix item(*this);

    sigmoid_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, this->size());
    cudaAssert(cudaDeviceSynchronize());

    return item;
}
Matrix Matrix::elu(float alph) const
{
    Matrix item(*this);

    elu_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, alph, this->size());
    cudaAssert(cudaDeviceSynchronize());

    return item;
}
Matrix Matrix::sign(float alph) const
{
    Matrix item(*this);
    sign_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, alph, this->size());
    cudaAssert(cudaDeviceSynchronize());

    return item;
}
Matrix Matrix::relu(float alph) const
{
    Matrix item(*this);
    relu_arr<float><<<(size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(item.dev_mat, alph, this->size());
    cudaAssert(cudaDeviceSynchronize());

    return item;
}

#endif