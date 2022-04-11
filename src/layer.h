#ifndef __LAYER__
#define __LAYER__

#include "modules.h"

using std::string;

typedef Eigen::MatrixXf Tensor2d;
typedef std::pair <uint, uint> Shape;
typedef std::tuple <uint, uint, uint, string, float, float> Metadata; 


class Layer
{
	public:

		virtual void init(const Shape &dim) = 0;

		virtual void forward(const Tensor2d &X) {}
		
		virtual void update() = 0;

	private:

       	float randint (int min, int max) const {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution <> dist(min, max - 1);
            return dist(gen);
        }

		Tensor2d RandMatrix (const Shape& dim, float min, float max) const {
        	return Tensor2d::NullaryExpr(dim.first, dim.second, [&, this]() { return randint(min, max); } );
		}
};

#endif
