#compilers

#GLOBAL_PARAMETERS
VALUE_TYPE = double
#NUM_RUN = 1000

#ENVIRONMENT_PARAMETERS
CUDA_INSTALL_PATH = /opt/linux/centos/7.x/x86_64/pkgs/cuda/9.1
CUDA_SAMPLES_PATH = /opt/linux/centos/7.x/x86_64/pkgs/cuda/9.1/samples


#CUDA_PARAMETERS
NVCC_FLAGS = -O3  -w -m64 -gencode=arch=compute_60,code=compute_60 --default-stream per-thread
CUDA_INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SAMPLES_PATH)/common/inc
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcusparse -Xcompiler="-lpthread -fopenmp" 
INC = -I ./include

cuda: dspmv_test.o anonymouslib_cuda.o dspmv_mgpu_baseline.o dspmv_mgpu_v1 dspmv_mgpu_v2.o dspmv_helper.o
	nvcc $(NVCC_FLAGS) dspmv_helper.o dspmv_test.o dspmv_mgpu_baseline.o dspmv_mgpu_v1 dspmv_mgpu_v2.o -o spmv $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)

dspmv_mgpu_v2.o: dspmv_mgpu_v2.cu 
	nvcc -c dspmv_mgpu_v2.cu $(INC) $(CUDA_INCLUDES)

dspmv_mgpu_v1.o: dspmv_mgpu_v1.cu 
	nvcc -c dspmv_mgpu_v1.cu $(INC) $(CUDA_INCLUDES)

dspmv_mgpu_baseline.o: dspmv_mgpu_baseline.cu 
	nvcc -c dspmv_mgpu_baseline.cu $(INC) $(CUDA_INCLUDES)

anonymouslib_cuda.o: anonymouslib_cuda.h 
	nvcc -c anonymouslib_cuda.h $(INC) $(CUDA_INCLUDES)

dspmv_test.o: dspmv_test.cpp 
	gcc -c dspmv_test.cpp $(INC) $(CUDA_INCLUDES)

dspmv_helper.o: dspmv_helper.cpp 
	gcc -c dspmv_helper.cpp $(INC) $(CUDA_INCLUDES)