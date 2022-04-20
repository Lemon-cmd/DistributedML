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

	float BCELoss(const Matrix &Y, float &accuracy) override
	{
		assert(init_);

		dH_ *= ((-H_ / Y + (1.0 - Y) / (1.0 - H)));

		return -(Y * H_.log() + (1.0 - Y) * (1.0 - H_).log()).sum() / Y.shape().first;
	}

	float MSELoss(const Matrix &Y, float &accuracy) override
	{
		assert(init_);
		dH_ *= (H_ - Y);

		return 0.5 * sqrtf((H_ - Y).pow(2.0).sum()) / Y.shape().first;
	}

	float CrossEntropyLoss(const Matrix &Y, float &accuracy) override
	{
		assert(init_);

		dH_ = H_ - Y;

		return ((-1.0 * Y) * H_.log()).sum() / Y.shape().first;
	}

private:
	float vw_, vb_;
	Matrix W_, B_, H_, dH_, lgrad_, I_, ones_;
	std::function<void(Matrix &, Matrix &)> func_;

	void init_weight()
	{
		B_.Constant(1.0);
		W_.Uniform(-0.5, 0.5);
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

	// m x d * d x dk + m x 1 * 1 x dk
	H_ = X.dot(W_);

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

	// adam parameters
	vw_ = 0.1 * vw_ + 0.9 * (dW.pow(2)).sum();

	// W : dk x d
	// dH : m x dk
	// lgrad : m x d
	lgrad_ = dH_.dot(W_.T());

	dW *= (lr_ / sqrtf(vw_ + er_));

	// update parameters
	W_ -= dW;
}

#endif
