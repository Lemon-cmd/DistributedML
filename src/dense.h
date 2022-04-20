#ifndef __DENSE__
#define __DENSE__

#include "layer.h"

using std::string;
typedef Eigen::MatrixXf Tensor2d;
typedef std::pair<size_t, size_t> Shape;

class Dense : public Layer
{
public:
	Dense() : vw_(0), vb_(0) {}

	Dense(size_t neurons,
		  const string &afunc = "sigmoid",
		  float lr = 1e-3, float er = 1e-8);

	void init(size_t in_dim);
	void init(size_t batch_dim, size_t in_dim);

	void ToHost();

	const Matrix &Get_bparam() const { return B_; }
	const Matrix &Get_wparam() const { return W_; }

	const Matrix &Get_H() const { return H_; }
	const Matrix &Get_dJ() const { return dH_; }
	const Matrix &Get_delta() const { return lgrad_; }

	void update();
	void forward(const Matrix &X);

	void set_dJ(const Matrix &dJ) { dH_ = dJ; }

	void set_delta(const Matrix &delta) { dH_ *= delta; }

	float MSELoss(const Matrix &Y, float &accuracy) override
	{
		assert(init_);
		dH_ *= (H_ - Y);

		return sqrtf((H_ - Y).pow(2.0).sum()) / H_.shape().first;
	}

	float CrossEntropyLoss(const Matrix &Y, float &accuracy) override
	{
		assert(init_);

		dH_ = H_ - Y;

		return ((-1.0 * Y) * H_.log()).sum() / H_.shape().first;
	}

private:
	float vw_, vb_;
	Matrix W_, B_, H_, dH_, lgrad_, I_, ones_;
	std::function<void(Matrix &, Matrix &)> func_;

	void init_weight()
	{
		B_.Constant(1.0);
		W_.Uniform(-0.2, 0.2);
	}
};

Dense::Dense(size_t neurons,
			 const string &afunc,
			 float lr, float er)
{
	lr_ = lr;
	er_ = er;
	afunc_ = afunc;
	out_dim_ = neurons;
	SetActivation(func_);
}

void Dense::init(size_t in_dim)
{
	assert(!init_);

	init_ = true;

	W_ = Matrix(in_dim,
				out_dim_);

	B_ = Matrix(1, out_dim_);

	init_weight();
}

void Dense::init(size_t batch_dim, size_t in_dim)
{
	init(in_dim);
}

void Dense::ToHost()
{
	assert(init_);

	W_.ToHost();
	B_.ToHost();
	H_.ToHost();
	dH_.ToHost();
}

void Dense::forward(const Matrix &X)
{
	assert(init_);
	ones_ = Matrix(X.shape().first, 1);
	ones_.Constant(1.0);

	// m x d * d x dk + m x 1 * 1 x dk
	H_ = X.dot(W_) + ones_.dot(B_);

	dH_ = H_;

	func_(H_, dH_);

	I_ = X;
}

void Dense::update()
{
	assert(init_);

	static Matrix dW;
	// m x d -> d x m
	I_.T_();

	// d x m * m x dk -> d x dk
	dW = I_.dot(dH_);

	// M x 1 -> 1 x M
	ones_.T_();

	// let ones now be the gradient w.r.t Bias
	// (1 x M) * (M x dk) -> (1 x dK)
	ones_.dot_(dH_); // sum

	// average
	ones_ /= dH_.shape().first;

	// adam parameters
	vw_ = 0.1 * vw_ + 0.9 * (dW.pow(2)).sum();
	vb_ = 0.1 * vb_ + 0.9 * (ones_.pow(2)).sum();

	// W : dk x d
	// dH : m x dk
	// lgrad : m x d
	lgrad_ = dH_.dot(W_.T());

	dW *= (lr_ / sqrtf(vw_ + er_));
	ones_ *= (lr_ / sqrtf(vb_ + er_));

	// update parameters
	W_ -= dW;
	B_ -= ones_;
}

#endif
