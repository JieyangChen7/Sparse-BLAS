#include <cuda_runtime.h>
#include "cusparse.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <pthread.h>
#include "spmv_task.h"
#include "spmv_kernel.h"
#include <omp.h>

using namespace std;

void * spmv_worker(void * arg);

void generate_tasks(int m, int n, int nnz, double * alpha,
				    double * csrVal, int * csrRowPtr, int * csrColIndex, 
				  	double * x, double * beta,
				  	double * y,
				  	int nb,
				  	vector<spmv_task *> * spmv_task_pool_ptr);

void assign_task(spmv_task * t, int dev_id, cudaStream_t stream);

void run_task(spmv_task * t, int dev_id, cusparseHandle_t handle, int kernel);

void finalize_task(spmv_task * t, int dev_id, cudaStream_t stream);

void print_task_info(spmv_task * t);

struct pthread_arg_struct
{
	vector<spmv_task *> * arg_spmv_task_pool;
	vector<spmv_task *> * arg_spmv_task_completed;
	int arg_dev_id;
};


int spMV_mgpu_v2(int m, int n, int nnz, double * alpha,
				  double * csrVal, int * csrRowPtr, int * csrColIndex, 
				  double * x, double * beta,
				  double * y,
				  int ngpu, 
				  int kernel,
				  int nb)
{
	vector<spmv_task *> spmv_task_pool;
	vector<spmv_task *> spmv_task_completed;

	generate_tasks(m, n, nnz, alpha,
				  csrVal, csrRowPtr, csrColIndex, 
				  x, beta, y, nb,
				  &spmv_task_pool);

	pthread_arg_struct * arg1 = new pthread_arg_struct();
	arg1->arg_spmv_task_pool = &spmv_task_pool;
	arg1->arg_spmv_task_completed = &spmv_task_completed;
	arg1->arg_dev_id = 0;


//	pthread_t thread_id;
//	pthread_create(&thread_id, NULL, spmv_worker, (void *)arg1);


	//thread gpu01 (spmv_worker, &spmv_task_pool, &spmv_task_completed);
	// thread gpu02 (spmv_worker, spmv_task_pool, spmv_task_completed);
	// thread gpu03 (spmv_worker, spmv_task_pool, spmv_task_completed);
	// thread gpu04 (spmv_worker, spmv_task_pool, spmv_task_completed);
	// thread gpu05 (spmv_worker, spmv_task_pool, spmv_task_completed);
	// thread gpu06 (spmv_worker, spmv_task_pool, spmv_task_completed);
	// thread gpu07 (spmv_worker, spmv_task_pool, spmv_task_completed);
	// thread gpu08 (spmv_worker, spmv_task_pool, spmv_task_completed);
	//gpu01.join();
	omp_set_num_threads(ngpu);
	#pragma omp parallel
	{
		unsigned int cpu_thread_id = omp_get_thread_num();
		cudaSetDevice(cpu_thread_id);
		cout << "set gpu " << cpu_thread_id << endl;

	}



}


void * spmv_worker(void * arg) {

	//int b = *((int *)arg);
	//cout << "b = " << b << endl;

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int device;
	for (device = 0; device < deviceCount; ++device) 
	{
	    cudaDeviceProp deviceProp;
	    cudaGetDeviceProperties(&deviceProp, device);
	    printf("Device %d has compute capability %d.%d.\n",
	           device, deviceProp.major, deviceProp.minor);
	}




	pthread_arg_struct arg_ptr = *((pthread_arg_struct*)arg);

	vector<spmv_task *> * spmv_task_pool = arg_ptr.arg_spmv_task_pool;
	vector<spmv_task *> * spmv_task_completed = arg_ptr.arg_spmv_task_completed;
	int dev_id = arg_ptr.arg_dev_id;
	cout << "dev_id = " << dev_id << endl;
	cusparseStatus_t status;
	cudaStream_t stream;
	cusparseHandle_t handle;
	//cudaSetDevice(0);
//	cudaStreamCreate(&stream);
	cout << "dev_id = " << dev_id << endl;

	// status = cusparseCreate(&handle); 
	// if (status != CUSPARSE_STATUS_SUCCESS) 
	// { 
	// 	printf("CUSPARSE Library initialization failed");
	// 	//return 1; 
	// } 
	// status = cusparseSetStream(handle, stream);
	// if (status != CUSPARSE_STATUS_SUCCESS) 
	// { 
	// 	printf("Stream bindind failed");
	// 	//return 1;
	// } 
	// while (spmv_task_pool->size() > 0)
	// {
	// 	// Take one task from pool
	// 	spmv_task * curr_spmv_task = (*spmv_task_pool)[spmv_task_pool->size() - 1];
	// 	spmv_task_pool->pop_back();
	// 	//assign_task(curr_spmv_task, dev_id, stream);
	// 	//run_task(curr_spmv_task, dev_id, handle, 1);
	// 	//finalize_task(curr_spmv_task, dev_id, stream);
	// 	print_task_info(curr_spmv_task);
	// 	spmv_task_completed->push_back(curr_spmv_task);
	// }

	//pthread_exit(NULL);
	//return 0;
}




void generate_tasks(int m, int n, int nnz, double * alpha,
				    double * csrVal, int * csrRowPtr, int * csrColIndex, 
				  	double * x, double * beta,
				  	double * y,
				  	int nb,
				  	vector<spmv_task *> * spmv_task_pool_ptr) {

	int num_of_tasks = (nnz + nb - 1) / nb;

	int curr_row;
	int t;
	int d;

	spmv_task * spmv_task_pool = new spmv_task[num_of_tasks];

	// Calculate the start and end index
	for (t = 0; t < num_of_tasks; t++) {
		spmv_task_pool[t].start_idx = floor((t) * nnz / num_of_tasks);
		spmv_task_pool[t].end_idx   = floor((t + 1) * nnz / num_of_tasks) - 1;
		spmv_task_pool[t].dev_nnz = spmv_task_pool[t].end_idx - spmv_task_pool[t].start_idx + 1;
	}

	// Calculate the start and end row
	curr_row = 0;
	for (t = 0; t < num_of_tasks; t++) {
		// while (csrRowPtr[curr_row] <= start_idx[i]) {
		// 	curr_row++;
		// }

		//  start_row[i] = curr_row - 1; 
		spmv_task_pool[t].start_row = get_row_from_index(m, csrRowPtr, spmv_task_pool[t].start_idx);

		// Mark imcomplete rows
		// True: imcomplete
		if (spmv_task_pool[t].start_idx > csrRowPtr[spmv_task_pool[t].start_row]) {
			spmv_task_pool[t].start_flag = true;
			spmv_task_pool[t].y2 = y[spmv_task_pool[t].start_idx];
		} else {
			spmv_task_pool[t].start_flag = false;
		}
	}

	curr_row = 0;
	for (t = 0; t < num_of_tasks; t++) {
		// while (csrRowPtr[curr_row] <= end_idx[i]) {
		// 	curr_row++;
		// 	//cout << "." << csrRowPtr[curr_row] << endl;
		// }

		// end_row[i] = curr_row - 1;
		spmv_task_pool[t].end_row = get_row_from_index(m, csrRowPtr, spmv_task_pool[t].end_idx);

		// Mark imcomplete rows
		// True: imcomplete
		if (spmv_task_pool[t].end_idx < csrRowPtr[spmv_task_pool[t].end_row + 1] - 1)  {
			spmv_task_pool[t].end_flag = true;
		} else {
			spmv_task_pool[t].end_flag = false;
		}
	}

	// Cacluclate dimensions
	for (t = 0; t < num_of_tasks; t++) {
		spmv_task_pool[t].dev_m = spmv_task_pool[t].end_row - spmv_task_pool[t].start_row + 1;
		spmv_task_pool[t].dev_n = n;
	}

	for (t = 0; t < num_of_tasks; t++) {
		spmv_task_pool[t].host_csrRowPtr = new int [spmv_task_pool[t].dev_m + 1];
		spmv_task_pool[t].host_csrRowPtr[0] = 0;
		spmv_task_pool[t].host_csrRowPtr[spmv_task_pool[t].dev_m] = spmv_task_pool[t].dev_nnz;

		// for (int j = 1; j < dev_m[i]; j++) {
		// 	host_csrRowPtr[i][j] = csrRowPtr[start_row[i] + j];
		// }

		memcpy(&(spmv_task_pool[t].host_csrRowPtr[1]), 
			   &csrRowPtr[spmv_task_pool[t].start_row + 1], 
			   (spmv_task_pool[t].dev_m - 1) * sizeof(int) );

		// cout << "host_csrRowPtr: ";
		// for (int j = 0; j <= dev_m[i]; j++) {
		// 	cout << host_csrRowPtr[i][j] << ", ";
		// }
		// cout << endl;

		for (int j = 1; j < spmv_task_pool[t].dev_m; j++) {
			spmv_task_pool[t].host_csrRowPtr[j] -= spmv_task_pool[t].start_idx;
		}

		// cout << "host_csrRowPtr: ";
		// for (int j = 0; j <= dev_m[i]; j++) {
		// 	cout << host_csrRowPtr[i][j] << ", ";
		// }
		// cout << endl;
		spmv_task_pool[t].host_csrColIndex = csrColIndex;
		spmv_task_pool[t].host_csrVal = csrVal;
		spmv_task_pool[t].host_y = y;
		spmv_task_pool[t].host_x = x;
		spmv_task_pool[t].local_result_y = new double[spmv_task_pool[t].dev_m];
	}

	for (t = 0; t < num_of_tasks; t++) {
		cusparseStatus_t status = cusparseCreateMatDescr(&(spmv_task_pool[t].descr));
		if (status != CUSPARSE_STATUS_SUCCESS) 
		{ 
			printf("Matrix descriptor initialization failed");
			//return 1;
		} 	
		cusparseSetMatType(spmv_task_pool[t].descr,CUSPARSE_MATRIX_TYPE_GENERAL); 
		cusparseSetMatIndexBase(spmv_task_pool[t].descr,CUSPARSE_INDEX_BASE_ZERO);
	}

	for (t = 0; t < num_of_tasks; t++) {
		(*spmv_task_pool_ptr).push_back(&spmv_task_pool[t]);
	}

}

void assign_task(spmv_task * t, int dev_id, cudaStream_t stream){
	t->dev_id = dev_id;
	cudaSetDevice(dev_id);
	cudaMalloc((void**)&(t->dev_csrVal),      (t->dev_nnz)   * sizeof(double));
	cudaMalloc((void**)&(t->dev_csrRowPtr),   (t->dev_m + 1) * sizeof(int)   );
	cudaMalloc((void**)&(t->dev_csrColIndex), (t->dev_nnz)   * sizeof(int)   );
	cudaMalloc((void**)&(t->dev_x),           (t->dev_n)     * sizeof(double));
    cudaMalloc((void**)&(t->dev_y),        (t->dev_m)        * sizeof(double));

    cudaMemcpyAsync(t->dev_csrRowPtr,   t->host_csrRowPtr,          
    			   (size_t)((t->dev_m + 1) * sizeof(int)), cudaMemcpyHostToDevice, stream);

	cudaMemcpyAsync(t->dev_csrColIndex, &(t->host_csrColIndex[t->start_idx]), 
		           (size_t)(t->dev_nnz * sizeof(int)), cudaMemcpyHostToDevice, stream); 

	cudaMemcpyAsync(t->dev_csrVal,      &(t->host_csrVal[t->start_idx]),
		           (size_t)(t->dev_nnz * sizeof(double)), cudaMemcpyHostToDevice, stream); 

	cudaMemcpyAsync(t->dev_y, &(t->host_y[t->start_row]), 
		           (size_t)(t->dev_m * sizeof(double)), cudaMemcpyHostToDevice, stream); 
	
	cudaMemcpyAsync(t->dev_x, t->host_x,
				   (size_t)(t->dev_n * sizeof(double)),  cudaMemcpyHostToDevice, stream); 
}

void run_task(spmv_task * t, int dev_id, cusparseHandle_t handle, int kernel){
	cudaSetDevice(dev_id);
	cusparseStatus_t status;
	if(kernel == 1) {
		status = cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
								t->dev_m, t->dev_n, t->dev_nnz, 
								t->alpha, t->descr, t->dev_csrVal, 
								t->dev_csrRowPtr, t->dev_csrColIndex, 
								t->dev_x,  t->beta, t->dev_y); 
	} else if (kernel == 2) {
		status = cusparseDcsrmv_mp(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
									t->dev_m, t->dev_n, t->dev_nnz, 
									t->alpha, t->descr, t->dev_csrVal, 
									t->dev_csrRowPtr, t->dev_csrColIndex, 
									t->dev_x,  t->beta, t->dev_y); 
	} else if (kernel == 3) {
		// int err = 0;
		// anonymouslibHandle<int, unsigned int, double> A(t->dev_m, t->dev_n);
		// err = A.inputCSR(
		// 	            t->dev_nnz, 
		// 				t->dev_csrRowPtr, 
		// 				t->dev_csrColIndex, 
		// 				t->dev_csrVal);
		// //cout << "inputCSR err = " << err << endl;
		// err = A.setX(t->dev_x);
		// //cout << "setX err = " << err << endl;
		// A.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA);
		// A.warmup();
		// err = A.asCSR5();
		// //cout << "asCSR5 err = " << err << endl;
		// err = A.spmv(*(t->alpha), t->dev_y);
	}

}

void finalize_task(spmv_task * t, int dev_id, cudaStream_t stream) {
	cudaSetDevice(dev_id);

	cudaMemcpyAsync(t->local_result_y,   t->dev_y,          
    			   (size_t)((t->dev_m) * sizeof(double)), 
    			   cudaMemcpyDeviceToHost, stream);
	cudaFree(t->dev_csrVal);
	cudaFree(t->dev_csrRowPtr);
	cudaFree(t->dev_csrColIndex);
	cudaFree(t->dev_x);
}

void print_task_info(spmv_task * t) {
	cout << "start_idx = " << t->start_idx << endl;
	cout << "end_idx = " << t->end_idx << endl;
}
