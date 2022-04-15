all: mpi_driver.cpp neural_net.cu ../src/modules.h ../src/utils.h
	mpicc -g mpi_driver.cpp -std=c++11 -c -o driver.o
	nvcc -g -O3 -lcublas -lcurand neural_net.cu -c -o network.o
	mpicc -g network.o driver.o -std=c++11 -o mpi_network -L/usr/local/cuda-11.2/lib64/ -lcublasLt -lcublas -lcudadevrt -lcurand -lcudart -lstdc++
	rm network.o
	rm driver.o
