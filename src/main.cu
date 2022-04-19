#include "modules.h"
#include "dense.h"
#include "cu_mat.h"
#include "clockcycle.h"

#define L std::unique_ptr<Layer>

int main()
{
	/*
	Matrix X(20, 1), Y(5, 1);

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

	// dk x dk
	Matrix ones(5, 5);
	ones.Constant(1.0);
	ones.ToDevice();

	ones = X % ones;
	X /= ones;
	X.ToHost();

	std::cout << X << '\n';

	X = X.bin();
	X.ToHost();
	std::cout << X << '\n';
	*/

	float accuracy = 0.0;

	Matrix X(100, 5), Y(100, 10);
	X.Random();
	Y.Constant(1);
	X.ToDevice();
	Y.ToDevice();

	std::vector<L>
		network(3);

	network[0] = L{new Dense(100)};
	network[0]->init(5);
	network[0]->ToDevice();

	for (uint j = 1; j < network.size(); j++)
	{
		network[j] = L{new Dense(100 - j)};
		std::cout << network[j - 1]->OutShape() << '\n';
		network[j]->init(network[j - 1]->OutShape());
		network[j]->ToDevice();
	}

	network[0]->forward(X);
	for (uint j = 1; j < network.size(); j++)
	{
		network[j]->forward(network[j - 1]->Get_H());
	}

	float loss = network.back()->MSELoss(Y, accuracy);
	std::cout << "L: " << loss << std::endl;
}