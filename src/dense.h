#ifndef __DENSE__
#define __DENSE__

#include "layer.h"

using std::string;
typedef Eigen::MatrixXf Tensor2d;
typedef std::pair<size_t, size_t> Shape;
typedef std::tuple<size_t, size_t, size_t, string, float, float> Metadata;

class Dense : public Layer
{
public:
	Dense(uint neurons, const string &afunc = "sigmoid",
		  float lr = 1e-3, float er = 1e-8)
	{
	}

	void init(const Shape &dim)
	{
	}

	void ToHost()
	{
	}

	void ToDevice()
	{
	}

	void update();
	void backward();
	void forward(const Tensor2d &X);

	float MSELoss(Matrix &Y, float &accuracy);
	float CrossEntropyLoss(Matrix &Y, float &accuracy);

private:
	float vu_, vb_;
	Matrix W_, B_, H_, dH_, lgrad_;
	std::function<Matrix &, Matrix &, float> func;

	void init_weight()
	{
		B_.Constant(1.0);
		W_.Uniform(-0.2, 0.2);
	}
};

void Dense::forward(const Tensor2d &X)
{
}

void Dense::backward()
{
}

void Dense::update()
{
}

float Dense::MSELoss(Matrix &Y, float &accuracy)
{
}

float Dense::CrossEntropyLoss(Matrix &Y, float &accuracy)
{
}

#endif
