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

	Dense(uint neurons,
		  const string &afunc = "sigmoid",
		  float lr = 1e-3, float er = 1e-8);

	void init(int in_dim)
	{
		W_ = Matrix(in_dim);
		B_ = Matrix(1, );
	}

	void ToHost()
	{
		W_.ToHost();
		B_.ToHost();
	}

	void ToDevice()
	{
		W_.ToDevice();
		B_.ToDevice();
	}

	void update();
	void backward();
	void forward(const Tensor2d &X);

	float MSELoss(Matrix &Y, float &accuracy);
	float CrossEntropyLoss(Matrix &Y, float &accuracy);

private:
	Shape out_dim;
	float vu_, vb_;
	Matrix W_, B_, H_, dH_, lgrad_;
	std::function<Matrix &, Matrix &, float> func;

	void init_weight()
	{
		B_.Constant(1.0);
		W_.Uniform(-0.2, 0.2);
	}
};

Dense() : W_(NULL), B_(NULL), H_(NULL), dH_(NULL), lgrad_(NULL)
{
	vu_ = 0.0f;
	vb_ = 0.0f;
}

Dense(uint neurons, const string &afunc = "sigmoid",
	  float lr = 1e-3, float er = 1e-8)
{
	Dense();
}

void Dense::forward(const Tensor2d &X)
{
	W_.T();
	H_ = X % W_ + B_;
	func(H_, dH_);
}

void Dense::backward()
{
}

void Dense::update()
{
}

float Dense::MSELoss(const Matrix &Y, float &accuracy)
{
	dH_ = dH_ * (X - Y);
	return sqrtf((X - Y).power(2.0).sum());
}

float Dense::CrossEntropyLoss(const Matrix &Y, float &accuracy)
{
	dH_ = H_ - Y;

	Matrix J = H_;
	J.Log();
	J *= -1.0 * Y;

	return J.sum();
}

#endif
