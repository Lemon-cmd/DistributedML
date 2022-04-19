#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <string.h>

extern int init_cuda(int my_rank);
extern void create_network(int input_size, int num_hidden, int* hidden_sizes, int output_size, std::string activation);
extern void load_train_test(int input_size, int output_size);
extern void run_forward_prop();
extern float run_back_prop();
extern void update_network();
extern float test_network();
extern void free_network();
extern float* get_gradient();
extern void set_gradient(float* grad);
extern float* get_yh();
extern float get_loss(float* yh);

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
          << "\t\t\"softmax\""
          << std::endl;
    return false;
}

void read_options(int argc, char** argv, int offset, bool& mode, std::string& activation){
    if(strcmp(argv[5+offset], "--mode") == 0){
    if(strcmp(argv[6+offset], "averaged") == 0){
         mode = 1;
    }
    }
    else if(strcmp(argv[5+offset], "--activation") == 0){
    if(strcmp(argv[6+offset], "tanh")==0) activation = "tanh";
    else if(strcmp(argv[6+offset], "sigmoid") == 0) activation = "sigmoid";
    else if(strcmp(argv[6+offset], "ELU") == 0) activation = "ELU";
    else if(strcmp(argv[6+offset], "sign") == 0) activation = "sign";
    else if(strcmp(argv[6+offset],"identity") == 0) activation = "identity";
    else if(strcmp(argv[6+offset], "softmax") == 0) activation = "softmax";
    else activation = "ReLU";
    }
}

bool read_parameters(int argc, char** argv, int& input_size, int& num_layers, int*& layer_sizes, int& output_size, int& epochs, bool& mode, std::string& activation){
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
    std::string activation = "ReLU";
    int valid = read_parameters(argc, argv, input_size, num_layers, layer_sizes, output_size, epochs, mode, activation);
    if(!valid){ return 1;}

    int test_size = 100;
    float loss = 0.0;
    
    int my_rank;
    int world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
 
    int found_device = init_cuda(my_rank);
    int ok;
    MPI_Allreduce(&found_device, &ok, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(ok != world_size){ return 1;}   

    // Instatiate the networks
    create_network(input_size, num_layers, layer_sizes, output_size, activation);
    load_train_test(input_size, output_size);

    // Averaged Distributed Network
    if(mode){
    // For each epoch
        for(int epoch = 0; epoch < epochs; epoch++){
            // Run forward propogation on each rank
            run_forward_prop();
            // Calculate gradients with backprop
            loss = run_back_prop();
            if(my_rank == 0)
                std::cout << "Loss: " << loss << std::endl;
            // Retrive gradients;
            float* grad = get_gradient();
            float* res_grad;
            // Average gradients accross all processes
            MPI_Allreduce(&grad, &res_grad, output_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            for(int i = 0; i < world_size; i++)
                res_grad[i] /= world_size;
            // Pass gradients back to NN
            set_gradient(res_grad);
            // Update networks with the set gradient
            update_network();
        }
        if(my_rank == 0){
            loss = test_network();
            std::cout << "Averaged Network Loss: " << loss << std::endl;
        }
    }
    // Ensemble learning 
    else{
        // For each epoch
        for(int epoch = 0; epoch < epochs; epoch++){
            // Run forward propogation on each rank
            run_forward_prop();
            // Run Backward propogation on each rank
            loss = run_back_prop();
            std::cout << "Rank: " << my_rank << " loss: " << loss << std::endl;
            // Update each network
            update_network();
        }
        // Print out the average loss of all networks
        float loss = test_network();
        float avg_loss = 0.0;
        MPI_Reduce(&loss, &avg_loss, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        if(my_rank == 0){
            std::cout << "Average Loss of all networks: " << avg_loss/world_size << std::endl;
        }
        
        float* yh = get_yh();
        float* avg_yh = (float*)malloc(test_size * sizeof(float));
        MPI_Reduce(&yh, &avg_yh, test_size, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        if(my_rank == 0){
            for(int i = 0; i < test_size; i++)
                avg_yh[i] /= world_size;
            loss = get_loss(avg_yh);
            std::cout << "Loss of averaged output of networks: " << loss << std::endl;
        }
    }
    
    free_network();

    MPI_Finalize();
}

