#include "modules.h"
#include "dense.h"
#include "cu_mat.h"
#include "clockcycle.h"

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

	/* ToHost() simply copy from device to host */
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
}
