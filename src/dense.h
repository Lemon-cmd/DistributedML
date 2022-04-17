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
		W_ = Matrix(in_dim,
					out_dim.second);

		B_ = Matrix(out_dim.second, 1);
	}

	void ToHost()
	{
		if (!cuda_)
		{
			MatToHost(W_);
			MatToHost(B_);
			MatToHost(H_);
			MatToHost(dH_);
			MatToHost(lgrad_);
		}
	}

	void ToDevice()
	{
		cuda_ = true;
		MatToDev(W_);
		MatToDev(B_);
		MatToDev(H_);
		MatToDev(dH_);
		MatToDev(lgrad_);
	}

	const Shape &OutShape() const
	{
		return out_dim;
	}

	const Matrix &get_dJ() const
	{
		return dH_;
	}

	void update();
	void forward(const Matrix &X);
	void set_delta(const Matrix &delta);

	float MSELoss(Matrix &Y, float &accuracy) override;
	float CrossEntropyLoss(Matrix &Y, float &accuracy) override;

private:
	Shape out_dim;
	float vw_, vb_;
	Matrix W_, B_, H_, dH_, lgrad_, I_;
	std::function<void(Matrix &, Matrix &)> func_;

	void init_weight()
	{
		B_.Constant(1.0);
		W_.Uniform(-0.2, 0.2);
	}

	void MatToDev(Matrix &mat)
	{
		if (mat != NULL)
			mat.ToDevice();
	}

	void MatToHost(Matrix &mat)
	{
		if (mat != NULL)
			mat.ToHost();
	}
};

Dense::Dense() : W_(NULL), B_(NULL), H_(NULL), dH_(NULL), lgrad_(NULL)
{
	vw_ = 0.0f;
	vb_ = 0.0f;
	out_dim = std::make_pair(0, 0);
}

Dense::Dense(uint neurons, const string &afunc = "sigmoid",
			 float lr = 1e-3, float er = 1e-8)
{
	Dense();
	lr_ = lr;
	er_ = er;
	afunc_ = afunc;
}

void Dense::forward(const Matrix &X)
{
	H_ = W_.transpose() % X + B_;
	func_(H_, dH_);
	I_ = X;
}

void Dense::update()
{
	static Matrix dW;
	I_.T();
	dW = dH_;
	dW.dot(I_);
	dW.pow(2.0);

	// adam parameters
	vw_ = 0.1 * vw_ + 0.9 * dW.sum();
	vb_ = 0.1 * vb_ + 0.9 * (dH_ * dH_).sum();
	lgrad_ = W_.transpose() % dH_;
}

void Dense::set_delta(const Matrix &delta)
{
	dH_ *= delta;
}

float Dense::MSELoss(const Matrix &Y, float &accuracy) override
{
	dH_ = dH_ * (H_ - Y);
	return sqrtf((H_ - Y).power(2.0).sum());
}

float Dense::CrossEntropyLoss(const Matrix &Y, float &accuracy) override
{
	dH_ = H_ - Y;

	Matrix J = H_;
	J.Log();
	J *= -1.0 * Y;

	return J.sum();
}

#endif
