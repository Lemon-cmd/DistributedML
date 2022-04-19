#ifndef __DENSE__
#define __DENSE__

#include "layer.h"

using std::string;
typedef Eigen::MatrixXf Tensor2d;
typedef std::pair<size_t, size_t> Shape;

class Dense : public Layer
{
public:
	Dense();

	Dense(size_t neurons,
		  const string &afunc = "sigmoid",
		  float lr = 1e-3, float er = 1e-8);

	void init(size_t in_dim);

	void ToHost();
	void ToDevice();

	const Matrix &Get_bparam() const { return B_; }
	const Matrix &Get_wparam() const { return W_; }

	Matrix Get_H() const { return H_; }
	Matrix Get_dJ() const { return dH_; }
	const Matrix &Get_delta() const { return lgrad_; }

	void update();
	void forward(const Matrix &X);

	void set_dJ(const Matrix &dJ) { dH_ = dJ; }

	void set_delta(const Matrix &delta) { dH_ *= delta; }

	float MSELoss(const Matrix &Y, float &accuracy) override
	{
		dH_ = dH_ * (H_ - Y);
		return sqrtf((H_ - Y).power(2.0).sum());
	}

	float CrossEntropyLoss(const Matrix &Y, float &accuracy) override
	{
		std::cout << "Fail\n";
		dH_ = H_ - Y;
		std::cout << "Fail?\n";

		J_ = H_;
		J_.Log();
		J_.ToHost();

		std::cout << "Fail??\n";
		J_ *= (-1.0);
		J_.ToHost();

		std::cout << "Fail???\n";
		J_ *= Y;

		std::cout << "Fail????\n";

		return J_.sum();
	}

private:
	float vw_, vb_;
	Matrix W_, B_, H_, dH_, lgrad_, I_, ones_, J_;
	std::function<void(Matrix &, Matrix &)> func_;

	void init_weight()
	{
		B_.Constant(1.0);
		W_.Uniform(-0.2, 0.2);
	}
};

Dense::Dense()
{
	vw_ = 0.0f;
	vb_ = 0.0f;
	out_dim_ = 0.0;
	afunc_ = "identity";
	SetActivation(func_);
}

Dense::Dense(size_t neurons,
			 const string &afunc,
			 float lr, float er)
{
	Dense();
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

void Dense::ToHost()
{
	if (cuda_)
	{
		W_.ToHost();
		B_.ToHost();
		H_.ToHost();
		dH_.ToHost();
		lgrad_.ToHost();
	}
}

void Dense::ToDevice()
{
	cuda_ = true;
	W_.ToDevice();
	B_.ToDevice();
}

void Dense::forward(const Matrix &X)
{
	ones_ = Matrix(X.shape().first, 1);
	ones_.ToDevice();

	std::cout << X.shape() << '\n'
			  << ones_.shape() << '\n'
			  << W_.shape() << '\n'
			  << B_.shape() << '\n';

	// m x d * d x dk + m x 1 * 1 x dk
	H_ = X % W_ + ones_ % B_;

	dH_ = H_;

	func_(H_, dH_);
	I_ = X;
}

void Dense::update()
{
	static Matrix dW;
	// d x m
	I_.T();

	// d x m * m x dk -> d x dk
	dW = I_ % dH_;
	dW.pow(2.0);

	// M x 1 -> 1 x M
	ones_.T();

	// let ones now be the gradient w.r.t Bias
	// (1 x M) * (M x dk) -> (1 x dK)
	ones_.dot(dH_); // sum

	// average
	ones_ /= (float)dH_.shape().first;

	// adam parameters
	vw_ = 0.1 * vw_ + 0.9 * (dW * dW).sum();
	vb_ = 0.1 * vb_ + 0.9 * (ones_ * ones_).sum();

	// W : dk x d
	// dH : m x dk
	// lgrad : m x d
	lgrad_ = dH_ % W_.transpose();

	dW *= (lr_ / sqrtf(vw_ + er_));
	ones_ *= (lr_ / sqrtf(vb_ + er_));

	// update parameters
	W_ -= dW;
	B_ -= ones_;
}

#endif
