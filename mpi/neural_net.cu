#include "../src/modules.h"
#include "../src/dense.h"
#include "../src/cu_mat.h"
#include "../src/utils.h"
#include "../src/parse_mnist.h"
#include <vector>
#define L std::unique_ptr<Layer>

// Neural Network is defined globally for the .cu code, and nowhere in the C++ code
std::vector<L> network;
int layers, out_dim;
Matrix X, Y, dJ;
float loss, acc, lr;
std::vector<Matrix> train_images, train_labels;
std::vector<Matrix> test_images, test_labels;
std::vector<float*> yh;

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


extern int load_train_test(int input_size, int output_size, int batch_size){
   
    load_mnist("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte", batch_size, train_images, train_labels);
    load_mnist("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte", batch_size, test_images, test_labels);
    
    //std::cout << "Loaded " << train_images.size() << " train images" << std::endl;
    //std::cout << "Loaded " << test_images.size() << " test images" << std::endl;    
    //std::cout << train_images[0].shape().first << ", " << train_images[0].shape().second << std::endl;
    
    return train_images.size();
}

extern void create_network(int input_size, int num_hidden, int* hidden_sizes, int output_size, std::string activation){
    //std::cout << "creating network" << std::endl;
    layers = 1 + num_hidden;
    out_dim = output_size;
    dJ = Matrix(output_size, 1);
    network = std::vector<L>(layers);
    loss = 0.0;
    acc = 0.0;
    lr = 0.001;
    
    network[0] = L{new Dense(hidden_sizes[0], activation, lr)};
    network[0]->init(input_size);
    
    for(int i = 1; i < num_hidden; i++){
        network[i] = L{new Dense(hidden_sizes[i], activation, lr)};
        network[i] -> init(hidden_sizes[i-1]);
    }
    
    network[layers-1] = L{new Dense(output_size, activation, lr)};
    network[layers-1]->init(hidden_sizes[num_hidden-1]);
    
}

extern void train_network(int num_batches){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, train_images.size()-1);
    loss = 0.0;
    acc = 0.0;
    float ret_acc;
    for(int batch = 0; batch < num_batches; batch++){
        uint batchid = distrib(gen); 
        network[0]->forward(train_images[batchid]);
        for(int i = 1; i < layers; i++){
            network[i]->forward(network[i-1]->Get_H());
        }
        loss += network.back()->CrossEntropyLoss(train_labels[batchid], ret_acc);
        // loss += network.back()->MSELoss(train_labels[batchid], ret_acc);

        acc += ret_acc;
        network.back()->update();
        for(int i = layers-2; i >= 0; i--){
            network[i]->set_delta(network[i+1]->Get_delta());
            network[i]->update();
        }
    }
    for(int i = 0; i < layers; i++)
        network[i]->set_lr(network[i]->get_lr() * 0.9);
    loss /= num_batches;
    acc /= num_batches;
}

extern void train_network(int num_batches, int* batches){
    loss = 0.0;
    acc = 0.0;
    float ret_acc;
    for(int batch = 0; batch < num_batches; batch++){
        network[0]->forward(train_images[batches[batch]]);
        for(int i = 1; i < layers; i++)
            network[i]->forward(network[i-1]->Get_H());
        loss += network.back()->CrossEntropyLoss(train_labels[batches[batch]], ret_acc);
        //loss += network.back()->CrossEntropyLoss(train_labels[batches[batch]], ret_acc);

        acc += ret_acc;
        network.back()->update();
        for(int i = layers-2; i >= 0; i--){
            network[i]->set_delta(network[i+1]->Get_delta());
            network[i]->update();
        }
    }
    acc /= num_batches;
    loss /= num_batches;
}

extern void run_forward_prop(int batch_id){
    //std::cout << "running forward prop" << std::endl;
    //std::cout << batch_id << std::endl;
    loss = 0.0;
    network[0]->forward(train_images[batch_id]);
    for(int i = 1; i < layers; i++){
        network[i]->forward(network[i-1]->Get_H());    
    }
    loss += network.back()->CrossEntropyLoss(train_labels[batch_id], acc);
    //loss += network.back()->MSELoss(train_labels[batch_id], acc);

    //loss /= 2;
 
}

extern void update_network(float* grad){
    //std::cout << "updating network" << std::endl;
    //std::cout << dJ.shape() << std::endl;
    dJ = Matrix(train_images[0].shape().first, out_dim, grad);
    //std::cout << dJ.shape() << std::endl;
    network.back()->set_dJ(dJ);
    network.back()->update();
    for(int i = layers-2; i >= 0; i--){
        network[i]->set_delta(network[i+1]->Get_delta());
        network[i]->update();
    }
    for(int i = 0; i < layers; i++)
        network[i]->set_lr(network[i]->get_lr() * 0.9999);
    // std::cout << "done\n";
}


extern float test_network(){
    std::cout << "testing network" << std::endl;
    float test_loss = 0.0;
    yh.clear();
    acc = 0.0;
    float ret_acc;
    for(int b = 0; b < test_images.size(); b++){
        network[0]->forward(test_images[b]);
        for(int i = 1; i < layers; i++)
            network[i]->forward(network[i-1]->Get_H());
        //test_loss += network.back()->MSELossNoGrad(test_labels[b], ret_acc);
        Matrix h = network.back()->Get_H().bin();
        test_loss += network.back()->CrossEntropyLossNoGrad(test_labels[b], ret_acc);
        //test_loss += network.back()->BCELossNoGrad(test_labels[b], ret_acc);

        acc += ret_acc;
        // Matrix h = network.back()->Get_H().bin();
        h.ToHost();
        //h = h.bin();
        yh.push_back((float*)(h.HostData().data()));
    }
    acc /= test_images.size();
    return test_loss / test_images.size();
}

extern void free_network(){
    std::cout << "freeing network" << std::endl;
}

extern float* get_gradient(){
    dJ = network.back()->Get_dJ();
    dJ.ToHost();
    return (float*)dJ.HostData().data();
}

extern std::vector<float*> get_yh(){
    return yh;
}

extern float get_loss(){
    return loss;
}

extern float get_acc(){
    return acc;
}

extern std::pair<float, float> run_loss(std::vector<float*> avg_yh){
    Matrix my_h;
    //test_labels[0].ToHost();
    //std::cout << std::endl << test_labels[0] << std::endl;
    float avg_loss = 0.0;
    float avg_acc = 0.0;
    for(int i = 0; i < test_labels.size(); i++){
        test_labels[i].ToHost();
        my_h.ToHost();
        my_h = Matrix(test_labels[i].size(), 1, avg_yh[i]);
        my_h.ToHost(); 
        avg_acc += my_h.bin().compare(test_labels[i]) / test_labels[i].shape().first;
        avg_loss += ((-1.0 * test_labels[i]) * my_h.log()).sum() / test_labels[i].shape().first;
        //avg_loss += sqrtf((my_h - test_labels[i]).pow(2.0).sum()) / test_labels[i].shape().first; 
    }
    return std::make_pair(avg_loss/avg_yh.size(), avg_acc/avg_yh.size());
} 
