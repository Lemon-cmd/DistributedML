#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <string.h>

extern void create_network(int input_size, int num_layers, int* layer_sizes, int output_size, int activation);
extern void run_forward_prop();
extern void run_back_prop();
extern void update_network();
extern void test_network();
extern void free_network();

bool print_usage(){
    std::cerr << "USAGE:\n mpi_network <Input Dim> <# hidden layers> <layer sizes...> <Output Dim> <Epochs> [--optional] \n"
          << "\t--mode:\n"
          << "\t\t\"ensemble\" default\n"
          << "\t\t\"averaged\"\n"
          << "\t--activation:\n"
          << "\t\t\"ReLU\" default\n "
          << "\t\t\"sigmoid\"\n"
          << "\t\t\"ELU\"\n"
          << "\t\t\"sign\"\n"
          << "\t\t\"tanh\"\n"
          << "\t\t\"identity\""
          << std::endl;
    return false;
}

void read_options(int argc, char** argv, int offset, bool& mode, int& activation){
    if(strcmp(argv[5+offset], "--mode") == 0){
    if(strcmp(argv[6+offset], "averaged") == 0){
         mode = 1;
    }
    }
    else if(strcmp(argv[5+offset], "--activation") == 0){
    if(strcmp(argv[6+offset], "tanh")==0) activation = 1;
    else if(strcmp(argv[6+offset], "sigmoid") == 0) activation = 2;
    else if(strcmp(argv[6+offset], "ELU") == 0) activation = 3;
    else if(strcmp(argv[6+offset], "sign") == 0) activation = 4;
    else if(strcmp(argv[6+offset],"identity") == 0) activation = 5;
    else activation = 0;
    }
}

bool read_parameters(int argc, char** argv, int& input_size, int& num_layers, int*& layer_sizes, int& output_size, int& epochs, bool& mode, int& activation){
    if(argc < 5)
    return print_usage();
    
    input_size = atoi(argv[1]);
    if(input_size == 0)
    return print_usage();

    num_layers = atoi(argv[2]);
    if(num_layers == 0)
    return print_usage();
    
    layer_sizes = (int*)malloc(num_layers * sizeof(int));
    for(int i = 0; i < num_layers; i++){
        layer_sizes[i] = atoi(argv[3+i]);
        if(layer_sizes[i] == 0)
        return print_usage();
    }
    
    output_size = atoi(argv[3+num_layers]);
    if(output_size == 0)
    return print_usage();
    
    epochs = atoi(argv[4+num_layers]);
    if(epochs == 0) 
    return print_usage();

    if(argc > 5 + num_layers)
    read_options(argc, argv, num_layers, mode, activation);
    if(argc > 7 + num_layers)
    read_options(argc, argv, num_layers+2, mode, activation);

    return true;
}

int main(int argc, char** argv){

    int input_size;
    int num_layers;
    int* layer_sizes;
    int output_size;
    int epochs;
    bool mode = 0;
    int activation = 0;
    int valid = read_parameters(argc, argv, input_size, num_layers, layer_sizes, output_size, epochs, mode, activation);
    if(!valid){ return 1;}

    int my_rank;
    int world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Load data onto each rank
    // std::vector <Layer*> network;
    // Initlize NN for each rank
    // Run traning loop for N epoch
    // Each loop does forward on CUDA
    
    create_network(input_size, num_layers, layer_sizes, output_size, activation);

    // Averaged Distributed Network
    if(mode){
    // For each epoch
        for(int epoch = 0; epoch < epochs; epoch++){
            // Run forward propogation on each rank
            run_forward_prop();
            // Calculate gradients with backprop
            run_back_prop();
            // Aggregate gradients in Rank 0
            // Pass gradients back to NN and update
            update_network();
        }
    }
    // Ensemble learning 
    else{
    // For each epoch
        for(int epoch = 0; epoch < epochs; epoch++){
            // Run forward propogation on each rank
            run_forward_prop();
            // Run Backward propogation on each rank
            run_back_prop();
            // Update each network
            update_network();
        }
    }

    test_network();
    // After Epoch
    // Calculate gradient for last layer
    // Two experiments:
    //  Send gradient to rank 0, average, and send back
    //  Just have independent model for each layer.

    // Eigen::MatrixXf X = Eigen::MatrixXf::Random(1024, 1024);

    // 1024 * 1024 array
    // MPI_Isend(X.data(), 1024 * 1024, MPI_FLOAT, );
    free_network();

    MPI_Finalize();
}

