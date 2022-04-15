#include "../src/modules.h"
#include "../src/dense.h"
#include "../src/cu_mat.h"
#include "../src/clockcycle.h"
#include "../src/utils.h"

extern void create_network(int input_size, int num_layers, int* layer_sizes, int output_size, int activation){
	std::cout << "creating network" << std::endl;
}

extern void run_forward_prop(){
    std::cout << "running forward prop" << std::endl;
}

extern void run_back_prop(){
    std::cout << "running backwrad prop" << std::endl;
}

extern void update_network(){
    std::cout << "updating network" << std::endl;
}

extern void test_network(){
    std::cout << "testing network" << std::endl;
}

extern void free_network(){
    std::cout << "freeing network" << std::endl;
}
