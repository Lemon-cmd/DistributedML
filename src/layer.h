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
	virtual void init(const Shape &dim) = 0;

	virtual void update() = 0;
	virtual void backward() = 0;
	virtual void forward(const Tensor2d &X) {}

private:
};

#endif
