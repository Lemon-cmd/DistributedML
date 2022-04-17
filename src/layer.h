#ifndef __LAYER__
#define __LAYER__

#include "modules.h"
#include "cu_mat.h"

using std::string;
typedef Eigen::MatrixXf Tensor2d;
typedef std::pair<size_t, size_t> Shape;
typedef std::tuple<size_t, size_t, size_t, string, float, float> Metadata;

class Layer
{
public:
	virtual void ToHost() = 0;
	virtual void ToDevice() = 0;
	virtual void init(int in_dim) = 0;

	virtual void update() = 0;
	virtual void backward() = 0;
	virtual void forward(const Tensor2d &X) {}

	virtual const Matrix &get_dJ() const {}
	virtual const Shape &Input2dShape() const {}
	virtual const Shape &Output2dShape() const {}

	virtual float MSELoss(const Matrix &Y, float &accuracy) {}
	virtual float CrossEntropyLoss(const Matrix &Y, float &accuracy) {}

private:
	float lr_, er_;
	string afunc_;
	bool init_ = false, cuda_ = false;

	virtual void init_weight() = 0;

	/*
	 *
	 * Activation Functions
	 *
	 */

	void Identity2d(Matrix &Z, Matrix &dZ)
	{
		dZ.setConstant(1.0);

		if (cuda_)
			dZ.ToDevice();
	}

	void Tanh2d(Matrix &Z, Matrix &dZ)
	{
		Z.Tanh();
		dZ = 1.0 - Z.power(2.0);
	}

	void Sigmoid2d(Matrix &Z, Matrix &dZ)
	{
		Z.Sigmoid();
		dZ = Z - Z.power(2.0);
	}

	void eLU2d(Matrix &Z, Matrix &dZ, float alph = 1.0)
	{
		dZ = Z;
		Matrix tmp1 = -1.0f * Z;

		dZ.Sign();
		tmp.Sign();
		Z.Elu(alph);

		tmp *= Z;
		dZ += tmp;
	}

	void ReLU2d(Matrix &Z, Matrix &dZ, float alph = 0.0)
	{
		Z.Relu(alph);
		dZ = Z;
		dZ.Sign(alph);
	}

	void Softmax2d(Matrix &Z, Matrix &dZ)
	{
		Z.exp();
		Z = Z / Z.sum();
	}

	void SetActivation(std::function<void(Matrix &, Matrix &)> &func)
	{
	}

	void SetActivation(std::function<void(Matrix &, Matrix &, float)> &func)
	{
	}
};

#endif
