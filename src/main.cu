#include "modules.h"
#include "cu_mat.h"
#include "dense.h"
#include "clockcycle.h"
#include "parse_mnist.h"

int main()
{
	/*
	Matrix X(3, 1), Y(3, 1);

	X.Constant(1);
	Y.Constant(1);

	X.ToHost();
	Y.ToHost();

	std::cout << X << "\n\n"
			  << Y << std::endl;

	std::cout << X.compare(Y) << std::endl;
	*/

	std::vector<Matrix> train_images, train_labels;
	load_mnist("../data/train-images-idx3-ubyte",
			   "../data/train-labels-idx1-ubyte",
			   100, train_images, train_labels);

	std::vector<Matrix> test_images, test_labels;
	load_mnist("../data/t10k-images-idx3-ubyte",
			   "../data/t10k-labels-idx1-ubyte",
			   100, test_images, test_labels);

	std::cout << train_images.size() << '\n'
			  << train_labels.size() << '\n';

	std::vector<Dense> network(3);

	std::cout << "Init:\n";

	network[0] = Dense(30, "relu", 0.01);
	network[0].init(28 * 28);

	for (uint j = 1; j < network.size() - 1; j++)
	{
		network[j] = Dense(50, "tanh", 0.1);
		network[j].init(network[j - 1].OutShape());
	}

	network.back() = Dense(1, "sigmoid", 0.001);
	network.back().init(network[network.size() - 2].OutShape());

	std::cout << "Training:\n";
	uint epochs = 10;

	for (uint e = 0; e < epochs; e++)
	{
		float loss = 0.0, acc = 0.0;

		for (uint k = 0; k < train_images.size(); k++)
		{
			float acc_batch = 0.0;
			// Forward pass
			network[0].forward(train_images[k]);
			for (uint j = 1; j < network.size(); j++)
			{
				network[j].forward(network[j - 1].Get_H());
			}

			loss += network.back().BCELoss(train_labels[k], acc_batch);
			acc += acc_batch;

			// network.back().ToHost();
			// std::cout << "H: " << network.back().Get_H() << std::endl;

			// Update
			network.back().update();

			for (int j = network.size() - 2; j >= 0; j--)
			{
				network[j].set_delta(network[j + 1].Get_delta());
				network[j].update();
			}
		}

		loss /= train_images.size();
		acc /= train_images.size();

		std::cout << "L: " << loss << " A: " << acc << std::endl;
	}
}
