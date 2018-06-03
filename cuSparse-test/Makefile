#compilers

#GLOBAL_PARAMETERS
VALUE_TYPE = double
#NUM_RUN = 1000

#ENVIRONMENT_PARAMETERS
CUDA_INSTALL_PATH = /home/chen838/cuda-8.0
CUDA_SAMPLES_PATH = /home/chen838/NVIDIA_CUDA-8.0_Samples


#CUDA_PARAMETERS
NVCC_FLAGS = -O3  -w -m64 -gencode=arch=compute_60,code=compute_60 --default-stream per-thread
CUDA_INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SAMPLES_PATH)/common/inc
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcusparse -Xcompiler="-lpthread"
INC = -I ./include

cuda:
	nvcc $(NVCC_FLAGS) dspmv_mgpu_v2.cu test2.cu -o spmv $(INC) $(CUDA_INCLUDES) $(CUDA_LIBS) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
