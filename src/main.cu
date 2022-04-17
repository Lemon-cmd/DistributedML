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

	L h1{new Dense(10, "identity")};
	L h2{new Dense(20)};

	h1->init(5);
	h2->init(h1->OutShape());

	std::vector<L> network;
	network.resize(10);

	network[0] = L{new Dense(10)};

	for (unsigned int i = 1; i < 10; i++)
	{
		if (i - 1 == 0)
			network[i - 1]->init(input_size);
		else
			network[i - 1]->init(network[i - 2]->OutShape());

		network[i] = L{new Dense(20)};
	}

	network.back()->init(network[network.size() - 2]->OutShape());
}
