#ifndef __DENSE__
#define __DENSE__

#include "layer.h"

using std::string;

typedef Eigen::MatrixXf Tensor2d;
typedef std::pair <size_t, size_t> Shape;
typedef std::tuple <size_t, size_t, size_t, string, float, float> Metadata; 

class Dense 
{
	public:
		Dense(uint neurons, const string& afunc = "sigmoid", float lr = 1e-3, float er = 1e-8)  
		{
			meta = std::make_tuple(neurons, 0, 0, afunc, lr, er);
		}

		void init(const Shape &dim)
		{
			
		}

		void update()
		{

		}

	private:
		float vu_, vb_;
		Tensor2d w_, b_, 
				 h_, dh_, delta_;
		
		Metadata meta;

};

#endif
