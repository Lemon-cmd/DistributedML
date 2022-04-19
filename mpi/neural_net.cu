#include "../src/modules.h"
#include "../src/dense.h"
#include "../src/cu_mat.h"
#include "../src/clockcycle.h"
#include "../src/utils.h"
#include <vector>
#define L std::unique_ptr<Layer>

// Neural Network is defined globally for the .cu code, and nowhere in the C++ code
std::vector<L> network;
int layers, out_dim;
Matrix X, Y, dJ;
float loss, acc;

extern int init_cuda(int my_rank){
    int cudaDeviceCount = 0;
    cudaError_t cE;
    if( (cE = cudaGetDeviceCount(&cudaDeviceCount)) != cudaSuccess){
        printf(" Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount);
        return 0;
    }

    if( (cE = cudaSetDevice(my_rank % cudaDeviceCount)) != cudaSuccess){
        printf(" Unable to have rank %d set to cuda device %d, error is %d \n", my_rank, (my_rank % cudaDeviceCount), cE);
        return 0;
    }
    return 1;
}


extern void load_train_test(int input_size, int output_size){
    X = Matrix(100, input_size);
    X.ToDevice();
    X.Uniform(-1, 1);
    
    Y = Matrix(100, output_size);
    Y.ToDevice();
    Y.Uniform(0, 1);
}

extern void create_network(int input_size, int num_hidden, int* hidden_sizes, int output_size, std::string activation){
    std::cout << "creating network" << std::endl;
    layers = 1 + num_hidden;
    out_dim = output_size;
    dJ = Matrix(output_size, 1);
    network = std::vector<L>(layers);
    loss = 0.0;
    acc = 0.0;
    
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
}

extern void run_forward_prop(){
    std::cout << "running forward prop" << std::endl;
    network[0]->forward(X);
    for(int i = 1; i < layers; i++){
        network[i]->forward(network[i-1]->get_H());    
    }
}

extern float run_back_prop(){
    std::cout << "running backward prop" << std::endl;
    std::cout << network.back()->OutShape() << std::endl;
    loss = network.back()->CrossEntropyLoss(Y, acc);
    std::cout << "AHHHHHHHHH" << std::endl;
    network.back()->get_dJ();
    std::cout << "done back prop" << std::endl;
    return loss;
}

extern void update_network(){
    std::cout << "updating network" << std::endl;
    
    network.back()->set_dJ(dJ);
    std::cout << "1\n";
    network.back()->update();
    std::cout << "2\n";
    for(int i = layers-2; i >= 0; i--){
        network[i]->set_delta(network[i+1]->get_delta());
        std::cout << "3\n";
        network[i]->update();
        std::cout << "4\n";
    }
}


extern void test_network(){
    std::cout << "testing network" << std::endl;
}

extern void free_network(){
    std::cout << "freeing network" << std::endl;
}

extern float* get_gradient(){
    dJ.ToDevice();
    dJ = network.back()->get_dJ();
    dJ.ToHost();
    return (float*)dJ.HostData().data();
}

extern void set_gradient(float* grad){
    dJ.ToHost();
    dJ = Matrix(out_dim, 1, grad);
    dJ.ToDevice();
}

extern float* get_yh(){
    float* q = (float*)malloc(100 * sizeof(float));
    return q;
}

extern float get_loss(float* yh){
    return 0.0;
}
