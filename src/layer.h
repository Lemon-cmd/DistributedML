#ifndef __LAYER__
#define __LAYER__

#include "modules.h"
#include "cu_mat.h"

#define empty std::placeholders

using std::string;
typedef Eigen::MatrixXf Tensor2d;
typedef std::pair<size_t, size_t> Shape;
typedef std::tuple<size_t, size_t, size_t, string, float, float> Metadata;

class Layer
{
public:
	virtual void ToHost() = 0;
	virtual void ToDevice() = 0;
	virtual void init(size_t in_dim) = 0;

	virtual void update() = 0;
	virtual void forward(const Matrix &X) = 0;

	virtual void set_dJ(const Matrix &dJ) = 0;
	virtual void set_delta(const Matrix &delta) = 0;

	virtual const Matrix &Get_wparam() const = 0;
	virtual const Matrix &Get_bparam() const = 0;

	virtual Matrix Get_H() const = 0;
	virtual Matrix Get_dJ() const = 0;

	virtual const Matrix &Get_delta() const = 0;

	virtual size_t OutShape() { return out_dim_; }

	virtual float MSELoss(const Matrix &Y, float &accuracy) { return 0; }
	virtual float CrossEntropyLoss(const Matrix &Y, float &accuracy) { return 0; }

protected:
	string afunc_;
	float lr_, er_;
	size_t out_dim_;
	bool init_ = false, cuda_ = false;

	virtual void init_weight() = 0;

	/*
	 *
	 * Activation Functions
	 *
	 */

	void Identity2d(Matrix &Z, Matrix &dZ)
	{
		dZ.Constant(1.0);
	}

	void Tanh2d(Matrix &Z, Matrix &dZ)
	{
		Z.tanh_();
		dZ = 1.0 - Z.pow(2.0);
	}

	void Sigmoid2d(Matrix &Z, Matrix &dZ)
	{
		Z.sigmoid_();
		dZ = Z - Z.pow(2.0);
	}

	void eLU2d(Matrix &Z, Matrix &dZ)
	{
		dZ = Z;
		Matrix tmp = Z;
		tmp *= -1.0;

		dZ.sign_();
		tmp.sign_();
		Z.elu_(1.0);

		tmp *= Z;
		dZ += tmp;
	}

	void ReLU2d(Matrix &Z, Matrix &dZ)
	{
		Z.relu_(0.0);
		dZ = Z;
		dZ.sign_(0.0);
	}

	void Softmax2d(Matrix &Z, Matrix &dZ)
	{
		Matrix ones(Z.shape().second, Z.shape().second);
		ones.Constant(1.0);

		Z.Exp();

		Matrix denom = Z.dot(ones_cols);
		Z /= denom;
	}

	void SetActivation(std::function<void(Matrix &, Matrix &)> &func)
	{
		func = std::bind(&Layer::Identity2d, this, empty::_1, empty::_2);

		if (afunc_ == "sigmoid")
		{
			func = std::bind(&Layer::Sigmoid2d, this, empty::_1, empty::_2);
		}
		else if (afunc_ == "tanh")
		{
			func = std::bind(&Layer::Tanh2d, this, empty::_1, empty::_2);
		}
		else if (afunc_ == "relu")
		{
			func = std::bind(&Layer::ReLU2d, this, empty::_1, empty::_2);
		}
		else if (afunc_ == "elu")
		{
			func = std::bind(&Layer::eLU2d, this, empty::_1, empty::_2);
		}
		else if (afunc_ == "softmax")
		{
			func = std::bind(&Layer::Softmax2d, this, empty::_1, empty::_2);
		}
	}
};

#endif
