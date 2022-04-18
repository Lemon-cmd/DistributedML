#include "../src/modules.h"
#include "../src/dense.h"
#include "../src/cu_mat.h"
#include "../src/clockcycle.h"
#include "../src/utils.h"
#include <vector>
// Neural Network is defined globally for the .cu code, and nowhere in the C++ code
#define L std::unique_ptr<Layer>
std::vector<L> network;
int layers, out_dim;
Matrix X, Y, dJ;

extern void create_network(int input_size, int num_hidden, int* hidden_sizes, int output_size, std::string activation){
	std::cout << "creating network" << std::endl;
    layers = 1 + num_hidden;
    out_dim = output_size;
    dJ = Matrix(output_size, 1);
    network = std::vector<L>(layers);

    network[0] = L{new Dense(hidden_sizes[0], activation)};
    network[0]->init(input_size);

    for(int i = 1; i < num_hidden; i++){
        network[i] = L{new Dense(hidden_sizes[i], activation)};
        network[i] -> init(hidden_sizes[i-1]);
    }
    network[layers-1] = L{new Dense(output_size, activation)};
    network[layers-1]->init(hidden_sizes[num_hidden-1]);

    for(int i = 0; i < layers; i++)
        network[i]->ToDevice();

    // Create Toy Dataset
    X = Matrix(100, input_size);
    X.ToDevice();
    X.Uniform(-1, 1);
    
    Y = Matrix(100, output_size);
    Y.ToDevice();
    Y.Uniform(0, 1);
}

extern void run_forward_prop(){
    std::cout << "running forward prop" << std::endl;
    network[0]->forward(X);
    for(int i = 1; i < layers; i++){
        network[i]->forward(network[i-1]->get_H());    
    }
}

extern void run_back_prop_(){
    std::cout << "running backwrad prop" << std::endl;
    float loss += network.back()->CrossEntropyLoss(Y, 0);
    dJ = network.back()->get_dJ();
}

extern void update_network(){
    std::cout << "updating network" << std::endl;
    
    network.back()->set_dJ(dJ);
    network.back()->update();
    for(int i = layers-2; i >= 0; i--){
        network[i]->set_delta(network[i+1]->get_delta());
        network[i]->update()
    }
}


extern void test_network(){
    std::cout << "testing network" << std::endl;
}

extern void free_network(){
    std::cout << "freeing network" << std::endl;
}

extern float* get_gradient(){
    dj.ToDevice();
    dj = network.back()->get_dJ();
    dj.ToHost();
    return dj.HostData();
}

extern set_gradient(float* grad){
    dJ = Matrix(out_dim, 1, grad);
}
