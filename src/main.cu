#include "modules.h"
#include "cu_mat.h"
#include "dense.h"
#include "clockcycle.h"

int main()
{
	/*
	Matrix X(3, 1), Y(3, 1);

	X.Random();
	Y.Constant(1);

	X.ToHost();
	Y.ToHost();

	std::cout << X << '\n';
	std::cout << Y << '\n';

	X = X.log();
	Y = Y.log();

	X.ToHost();
	Y.ToHost();

	std::cout << X << '\n';
	std::cout << Y << '\n';

	Matrix A = X.T().dot(Y);
	A.ToHost();
	std::cout << A << '\n';
	*/

	float accuracy = 0.0;

	Matrix X(100, 5), Y(100, 10);
	X.Random();
	Y.Constant(1);
	X.ToDevice();
	Y.ToDevice();

	std::vector<Dense> network(3);

	network[0] = Dense(100, "identity");
	network[0].init(5);
	network[0].ToDevice();

	for (uint j = 1; j < network.size(); j++)
	{
		network[j] = Dense(100 - j, "identity");
		network[j].init(network[j - 1].OutShape());
		network[j].ToDevice();
	}

	network[0].forward(X);
	for (uint j = 1; j < network.size(); j++)
	{
		network[j].forward(network[j - 1].Get_H());
	}

	float loss = network.back().MSELoss(Y, accuracy);
	std::cout << "L: " << loss << std::endl;

	return 0;
}
