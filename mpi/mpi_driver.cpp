#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <string.h>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include "../src/clockcycle.h"

extern int init_cuda(int my_rank);
extern void create_network(int input_size, int num_hidden, int* hidden_sizes, int output_size, std::string activation);
extern int load_train_test(int input_size, int output_size, int batch_size);
extern void run_forward_prop(int batch_id);
extern void update_network(float* grad);
extern float test_network();
extern void free_network();
extern float* get_gradient();
extern std::vector<float*> get_yh();
extern float get_loss();
extern void train_network(int batches);
extern void train_network(int batches, int* batch_idx);
extern float get_acc();
extern std::pair<float, float> run_loss(std::vector<float*> yh);

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

void read_options(int argc, char** argv, int offset, bool& mode, std::string& activation, int& batch_size){
    if(strcmp(argv[5+offset], "--mode") == 0){
        if(strcmp(argv[6+offset], "averaged") == 0){
            mode = 1;
        }
    }
    else if(strcmp(argv[5+offset], "--activation") == 0){
        if(strcmp(argv[6+offset], "tanh")==0)
           activation = "tanh";
        else if(strcmp(argv[6+offset], "sigmoid") == 0)
            activation = "sigmoid";
        else if(strcmp(argv[6+offset], "ELU") == 0)
            activation = "ELU";
        else if(strcmp(argv[6+offset], "sign") == 0)
            activation = "sign";
        else if(strcmp(argv[6+offset],"identity") == 0)
            activation = "identity";
        else if(strcmp(argv[6+offset], "softmax") == 0)
            activation = "softmax";
        else
            activation = "ReLU";
    }
    else if(strcmp(argv[5+offset], "--batch_size") == 0){
        batch_size = atoi(argv[6+offset]);
    }
}

bool read_parameters(int argc, char** argv, int& input_size, int& num_layers, int*& layer_sizes, int& output_size, int& epochs, bool& mode, std::string& activation, int& batch_size){
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
        read_options(argc, argv, num_layers, mode, activation, batch_size);
    if(argc > 7 + num_layers)
        read_options(argc, argv, num_layers+2, mode, activation, batch_size);
    if(argc > 9 + num_layers)
        read_options(argc, argv, num_layers+4, mode, activation, batch_size);
 
    return true;
}

int main(int argc, char** argv){

    int input_size;
    int num_layers;
    int* layer_sizes;
    int output_size;
    int epochs;
    bool mode = 0;
    int batch_size = 512;
    int batches_per_update = 1;
    std::string activation = "ReLU";
    int valid = read_parameters(argc, argv, input_size, num_layers, layer_sizes, output_size, epochs, mode, activation, batch_size);
    if(!valid){ return 1;}

    float loss = 0.0;
    float acc = 0.0;

    unsigned long long start_time;
    unsigned long long end_time;
    int clock_frequency = 512000000;
    
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
    int num_batches = load_train_test(input_size, output_size, batch_size);

    // Averaged Distributed Network
    if(mode){
    // For each epoc
        start_time = clock_now();
        for(int epoch = 0; epoch < epochs; epoch++){
            std::vector<int> indices(num_batches);
            int* indices_raw;
            
            indices_raw = (int*)malloc(sizeof(int)*num_batches); 
            if(my_rank == 0){
                std::iota (indices.begin(), indices.end(), 0);
                auto rng = std::default_random_engine {};
                std::shuffle(indices.begin(), indices.end(), rng);
                for(int i = 0; i< num_batches; i++)
                    indices_raw[i] = indices[i];
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Bcast(indices_raw, num_batches, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);

            for(int batches = 0; batches < 4 * num_batches/world_size; batches++){
                //for(int q = 0; q < 1; q++){
                //    if(batches + 1 < num_batches){
                //        run_forward_prop(indices_raw[batches]);
                //        update_network(get_gradient());
                //        batches++;
                //    }
                //    else{break;}
                //}    
                run_forward_prop(indices_raw[(my_rank + (batches*world_size)) % num_batches]);
                loss = get_loss();
                acc = get_acc();
                // Retrive gradients;
                float* grad = get_gradient();
                float* res_grad = (float*)malloc(sizeof(float)* batch_size * output_size);
                // Average gradients accross all processes
                MPI_Allreduce(grad, res_grad, batch_size * output_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                for(int i = 0; i < batch_size * output_size; i++)
                    res_grad[i] /= world_size;
                // Pass gradients back to NN
                update_network(res_grad);
                // Update networks with the set gradient
            }
            loss = get_loss();
            acc = get_acc();
            float avg_loss = 0.0;
            MPI_Reduce(&loss, &avg_loss, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            float avg_acc = 0.0;
            MPI_Reduce(&acc, &avg_acc, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if(my_rank == 0){
                end_time = clock_now();
                //std::cout << tworks: " << avg_loss/world_size << std::endl;
                std::cout << avg_acc/world_size << std::endl;
                //double time_in_secs = ((double)(end_time - start_time)) / clock_frequency;
                //std::cout << "Time: " << time_in_secs << std::endl;
            }
            free(indices_raw);
        }
        if(my_rank == 0){
            end_time = clock_now();
            loss = test_network();
            acc = get_acc();
            std::cout << "Averaged Network Loss: " << loss << std::endl;
            std::cout << "Averaged Network Acc: " << acc << std::endl;
            double time_in_secs = ((double)(end_time - start_time)) / clock_frequency;
            std::cout << "Time: " << time_in_secs << std::endl;

        }
    }
    // Ensemble learning 
    else{
        // For each epoch
        start_time = clock_now();
        for(int epoch = 0; epoch < epochs; epoch++){
            // Run all batches
            train_network(num_batches);
            // Get the loss 
            loss = get_loss();
            acc = get_acc();
            std::cout << "Rank: " << my_rank << " epoch: " << epoch << " loss: " << loss << " acc: " << acc << std::endl;
            //update_network();
            
        }
        MPI_Barrier(MPI_COMM_WORLD);
        // Print out the average loss of all networks
        float loss = test_network();
        float avg_loss = 0.0;
        MPI_Reduce(&loss, &avg_loss, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        float acc = get_acc();
        float avg_acc = 0.0;
        MPI_Reduce(&acc, &avg_acc, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        end_time = clock_now();
        std::cout << "Rank: " << my_rank << " Final Loss: " << loss << " Final Acc: " << acc << std::endl;
        if(my_rank == 0){
            std::cout << "Average Loss of all networks: " << avg_loss/world_size << std::endl;
            std::cout << "Average Accuracy of all networks: "  << avg_acc/world_size << std::endl;
            double time_in_secs = ((double)(end_time - start_time)) / clock_frequency;
            std::cout << "Time: " << time_in_secs << std::endl;
        }

        std::vector<float*> yh = get_yh();
        std::vector<float*> avg_yh;
        MPI_Barrier(MPI_COMM_WORLD);
        float* res_yh = (float*)malloc(batch_size * sizeof(float));

        for(int i = 0; i < yh.size(); i++){
            MPI_Reduce(yh[i], res_yh, batch_size, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            avg_yh.push_back(res_yh);
        }
        free(res_yh);

        if(my_rank == 0){
            for(int i = 0; i < avg_yh.size(); i++){
                for(int j = 0; j < batch_size; j++)
                    avg_yh[i][j] /= (float)world_size;
            }
            std::pair<float, float> res = run_loss(avg_yh);
            std::cout << "Loss of averaged output of networks: " << res.first << std::endl;
            std::cout << "Acc of averaged output of networks: " << res.second << std::endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); 

    MPI_Finalize();
}

