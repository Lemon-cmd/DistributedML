# Distributed Machine Learning Project

## Setting up packages
On AiMOS: 
```
$ export http_proxy=http://proxy:8888
$ export https_proxy=$http_proxy
$ module load xl_r spectrum_mpi cuda 
```
Install Conda
```
$ cd ~/scratch
$ mkdir AI-CONDA
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh
$ bash ./Miniconda3-latest-Linux-ppc64le.sh -p AI-CONDA
```
After reloading
```
conda install eigen -y
```

Edit line 7 in /src/modules.h to your path to the Eigen Install

## Running Neural Netork
Compile
```
cd /mpi/
make
```
Usage:
```
mpi_network <Input Dim> <# hidden layers> <layer sizes...> <Output Dim> <Epochs> [--optional] 
  --mode: "ensemble" default
          "averaged"
  --activation: "ReLU" default 
                "sigmoid"
                "ELU"
                "sign"
                "tanh"
                "identity"
                "softmax"
  --batch_size: [1-1024] 512 default
```

Run a tanh network for 10 epochs with:  
16 sub networks  
5 input channels  
3 hidden layers  
10, 20, 15 layer sizes  
2 output channels  
```
$ mpirun -np 16 mpi_network 5 3 10 20 15 2 10 --activation tanh
```

Modes:  
"ensemble" - create np separate NNs, with aggregation  
"averaged" - after each epoch, each NN recieves an averaged gradient update from across all NNs

