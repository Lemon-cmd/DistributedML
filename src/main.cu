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

	Matrix X(3, 5), Y(3, 10);
	X.Random();
	Y.Constant(1);

	std::vector<Dense> network(3);

	std::cout << "Init:\n";

	network[0] = Dense(3, "relu");
	network[0].init(5);

	for (uint j = 1; j < network.size() - 1; j++)
	{
		network[j] = Dense(3, "sigmoid");
		network[j].init(network[j - 1].OutShape());
	}

	network.back() = Dense(10, "relu");
	network.back().init(network[network.size() - 2].OutShape());

	std::cout << "Training:\n";
	uint epochs = 10;
	for (uint e = 0; e < epochs; e++)
	{

		// Forward pass
		network[0].forward(X);
		for (uint j = 1; j < network.size(); j++)
		{
			network[j].forward(network[j - 1].Get_H());
		}

		float loss = network.back().MSELoss(Y, accuracy);
		std::cout << "L: " << loss << std::endl;

		network.back().ToHost();
		std::cout << "H: " << network.back().Get_H() << std::endl;

		// Update
		network.back().update();

		for (int j = network.size() - 2; j >= 0; j--)
		{
			network[j].set_delta(network[j + 1].Get_delta());
			network[j].update();
		}
	}

	Matrix A(3, 1), B(3, 1);
	A.Constant(1.0);
	B.Constant(1.0);

	std::cout << A.compare(B) << std::endl;

	return 0;
}
