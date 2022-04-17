#include "modules.h"
#include "dense.h"
#include "cu_mat.h"
#include "clockcycle.h"

#define L std::unique_ptr<Layer>

int main()
{
	/*
	Matrix X(2, 1), Y(2, 1);

	X.Uniform(-1, 1);
	Y.Uniform(-1, 1);

	// it is better to fill matrix on the cpu first
	X.ToDevice();
	Y.ToDevice();

	// fill matrix with a val
	// X.Constant(2.0);
	// Y.Constant(4.0);

	X.ToHost();
	Y.ToHost();
	std::cout << X << '\n'
			  << '\n'
			  << Y << std::endl;

	X = 1.0 - X;
	X.ToHost();
	std::cout << X << std::endl;

	Y = 2.0 / Y;
	Y.ToHost();
	std::cout << Y << std::endl;

	*/

	Matrix X(2, 1);
	X.Uniform(-1, 1);
	X.ToDevice();

	L l0{new Dense(100, "identity")};
	l0->init(1);

	l0->ToDevice();

	l0->forward(X);
	l0->ToHost();

	std::cout << l0->Get_H() << std::endl;
}
