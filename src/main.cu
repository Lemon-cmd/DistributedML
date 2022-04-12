#include "modules.h"
#include "dense.h"
#include "cu_mat.h"

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main()
{
	Matrix X (2, 1), Y(2, 1);
	
	X.Uniform(-0.2, 0.2);
	Y.Uniform(-0.2, 0.2);

	// it is better to fill matrix on the cpu first
	X.ToDevice();
	Y.ToDevice();

	// fill matrix with a val
	//X.Constant(2.0);
	//Y.Constant(4.0);

	/* ToHost() simply copy from device to host */
	X.ToHost();
	Y.ToHost();
	std::cout << "X:\n" <<  X << "\nY:\n" << Y << std::endl;

	Matrix Z = X * X - Y;

	Z.ToHost();

	std::cout << "X * X - Y:\n" << Z << '\n';

	Matrix dX =  X * -1.0;
	dX.ToHost();

	std::cout << "dX: \n" << dX << '\n';
	
}
