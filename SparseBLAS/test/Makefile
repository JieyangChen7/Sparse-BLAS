#compilers

#GLOBAL_PARAMETERS
VALUE_TYPE = double
#NUM_RUN = 1000

#ENVIRONMENT_PARAMETERS
CUDA_INSTALL_PATH = /home/chen838/cuda-8.0
CUDA_SAMPLES_PATH = /home/chen838/cuda-8.0/samples


#CUDA_PARAMETERS
NVCC_FLAGS = -O3  -w -m64 -gencode=arch=compute_60,code=compute_60 --default-stream per-thread
CUDA_INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SAMPLES_PATH)/common/inc
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcusparse -Xcompiler="-lpthread -fopenmp" 
INC = -I ../include

.DEFAULT_GOAL := all

all: cuda

cuda: csr5_kernel.o dspmv_test.o dspmv_mgpu_baseline.o dspmv_mgpu_v1.o dspmv_mgpu_v2.o spmv_helper.o
	nvcc $(NVCC_FLAGS) ../src/csr5_kernel.o ../src/spmv_helper.o dspmv_test.o ../src/dspmv_mgpu_baseline.o ../src/dspmv_mgpu_v1.o ../src/dspmv_mgpu_v2.o -o spmv $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)

dspmv_test.o: dspmv_test.cu 
	nvcc -c $(NVCC_FLAGS) ./test/dspmv_test.cu $(INC) $(CUDA_INCLUDES)
