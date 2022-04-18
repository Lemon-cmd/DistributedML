#include "modules.h"
#include "dense.h"
#include "cu_mat.h"
#include "clockcycle.h"

#define L std::unique_ptr<Layer>

int main()
{
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

	X.dot(Y.transpose());

	X.ToHost();
	std::cout << X << '\n';

	X.Exp();
	Matrix ones(2, 2);
	ones.Constant(1.0);
	ones.ToDevice();

	ones = X % ones;
	X /= ones;
	X.ToHost();

	std::cout << X << '\n';
}
