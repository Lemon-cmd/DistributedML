#include "modules.h"
#include "dense.h"
#include "cu_mat.h"

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main()
{
	Matrix X (2, 2), Y(2, 2);
	
	// it is better to fill matrix on the cpu first
	X.ToDevice();
	Y.ToDevice();

	// fill matrix with a val
	//X.Constant(2.0);
	//Y.Constant(4.0);

	// fill matrix with vals of a given range
	//X.Uniform(-0.2, 0.2);
	//Y.Uniform(-0.2, 0.2);
	
	// fill matrice with rand. vals 0 -> 1
	X.Random();
	Y.Random();

	/* ToHost() simply copy from device to host */
	X.ToHost();
	Y.ToHost();
	std::cout << X << '\n' << Y << std::endl;

	// transpose and return a matrix
	Matrix XT = X.transpose();
	XT.ToHost();
	std::cout << XT << '\n';
	
	X.T(); // transpose in place
	X.ToHost();
	std::cout << "Transpose In Place:\n\n" << X << '\n';

	X.T(); 
	X.dot(Y); // matrix mult in place
	X.ToHost();
	std::cout << X << '\n';

	/* Non in-place math operators */
	std::cout << "Non in-place math operators\n\n";

	X = X + 1.0;
	X.ToHost();
	std::cout << X << '\n'; 

	X = X + Y;
	X.ToHost();

	std::cout << X << '\n'; 

	std::cout << "Division\n";
	X = X / Y;
	X.ToHost();
	std::cout << X << '\n'; 
	std::cout << Y << '\n'; 


	std::cout << "Multiply\n";
	X = X * Y;
	X.ToHost();
	std::cout << X << '\n'; 

	// To complete remove matrix from GPU
	X.cpu();
	Y.cpu();

	// Call ToHost() only move from device to host

	/* In-place math operators */
	std::cout << "In-place math operators\n\n";

	X = Matrix(5, 5), Y = Matrix(5, 5);
	X.Uniform(-0.2, 0.2);
	Y.Uniform(-5, 5);

	X.ToDevice();
	Y.ToDevice();

	X += Y;
	X.ToHost();
	std::cout << X << '\n'; 

	X -= Y;
	X.ToHost();
	std::cout << X << '\n'; 

	X *= Y;
	X.ToHost();
	std::cout << X << '\n'; 

	X /= Y;
	X.ToHost();
	std::cout << X << '\n'; 
	
	X.pow(3.0);
	X.ToHost();
	std::cout << X << '\n'; 
}
