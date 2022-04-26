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

However this only works with the MNIST dataset:
```
$ mpirun -np 16 mpi_network 784 3 256 256 256 1 10 --activation ELU
```
Modes:  
"ensemble" - create np separate NNs, with aggregation  
"averaged" - after each epoch, each NN recieves an averaged gradient update from across all NNs

## cuMatrix
cuMatrix is a light-weight Matrix library that utilizes cuBlas and CUDA. The primary goal of this librabry is to provide a simple interface to perform simple linear algebra operations, such as matrix multiplication, transpose, hadamard product, etc. It is best to stress that there are no linear algebra solvers implemented in cuMatrix. 

There are two types of operation in cuMatrix: in-place and non-in-place 

For example, matrix multiplication
```
A.dot_(B); // in-place
A.dot(B);  // non in-place

```

The same applies to element-wise operations (which are common operators)
```
A += B;   // in-place
A + B     // non in-place
```

There are other important functions as well (e.g., pow, sum, exp, log, etc...).


