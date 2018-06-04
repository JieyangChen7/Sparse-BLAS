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

	// pthread_arg_struct * arg1 = new pthread_arg_struct();
	// arg1->arg_spmv_task_pool = &spmv_task_pool;
	// arg1->arg_spmv_task_completed = &spmv_task_completed;
	// arg1->arg_dev_id = 0;

	// omp_set_num_threads(ngpu);
	// #pragma omp parallel default (shared)
	// {
	// 	int c;
	// 	unsigned int dev_id = omp_get_thread_num();
	// 	cudaSetDevice(dev_id);
		

	// 	int copy_of_workspace = 2;

	// 	cusparseStatus_t status[copy_of_workspace];
	// 	cudaStream_t stream[copy_of_workspace];
	// 	cusparseHandle_t handle[copy_of_workspace];



	// 	double ** dev_csrVal = new double * [copy_of_workspace];
	// 	int ** dev_csrRowPtr = new int    * [copy_of_workspace];
	// 	int ** dev_csrColIndex = new int  * [copy_of_workspace];
	// 	double ** dev_x = new double      * [copy_of_workspace];
	// 	double ** dev_y = new double      * [copy_of_workspace];

	// 	for (c = 0; c < copy_of_workspace; c++) {
	// 		cudaStreamCreate(&(stream[c]));
	// 		status[c] = cusparseCreate(&(handle[c])); 
	// 		if (status[c] != CUSPARSE_STATUS_SUCCESS) 
	// 		{ 
	// 			printf("CUSPARSE Library initialization failed");
	// 			//return 1; 
	// 		} 
	// 		status[c] = cusparseSetStream(handle[c], stream[c]);
	// 		if (status[c] != CUSPARSE_STATUS_SUCCESS) 
	// 		{ 
	// 			printf("Stream bindind failed");
	// 			//return 1;
	// 		} 

	// 		cudaMalloc((void**)&(dev_csrVal[c]),      nnz      * sizeof(double));
	// 		cudaMalloc((void**)&(dev_csrRowPtr[c]),   (m + 1) * sizeof(int)   );
	// 		cudaMalloc((void**)&(dev_csrColIndex[c]), nnz      * sizeof(int)   );
	// 		cudaMalloc((void**)&(dev_x[c]),           n       * sizeof(double));
	//     	cudaMalloc((void**)&(dev_y[c]),           m       * sizeof(double));

 //    	}

 //    	c = 0; 
    
	// 	while (true) {

	// 		spmv_task * curr_spmv_task;

	// 		for (c = 0; c < copy_of_workspace; c++) {


	// 			#pragma omp critical
	// 			{
	// 				if(spmv_task_pool.size() > 0) {
	// 					curr_spmv_task = spmv_task_pool[spmv_task_pool.size() - 1];
	// 					spmv_task_pool.pop_back();
	// 				} else {
	// 					curr_spmv_task = NULL;
	// 				}
	// 			}

	// 			if (curr_spmv_task) {

	// 				curr_spmv_task->dev_csrVal = dev_csrVal[c];
	// 				curr_spmv_task->dev_csrRowPtr = dev_csrRowPtr[c];
	// 				curr_spmv_task->dev_csrColIndex = dev_csrColIndex[c];
	// 				curr_spmv_task->dev_x = dev_x[c];
	// 				curr_spmv_task->dev_y = dev_y[c];
	// 				assign_task(curr_spmv_task, dev_id, stream[c]);
	// 				run_task(curr_spmv_task, dev_id, handle[c], kernel);
	// 				finalize_task(curr_spmv_task, dev_id, stream[c]);
	// 			}
	// 		}
	// 		if (!curr_spmv_task) {
	// 			break;
	// 		}
	// 	}

	// 	cudaDeviceSynchronize();

	// 	for (c = 0; c < copy_of_workspace; c++) {

	// 		cudaFree(dev_csrVal[c]);
	// 		cudaFree(dev_csrRowPtr[c]);
	// 		cudaFree(dev_csrColIndex[c]);
	// 		cudaFree(dev_x[c]);
	// 		cudaFree(dev_y[c]);
	// 	}
	// }
}



void generate_tasks(int m, int n, int nnz, double * alpha,
				    double * csrVal, int * csrRowPtr, int * csrColIndex, 
				  	double * x, double * beta,
				  	double * y,
				  	int nb,
				  	vector<spmv_task *> * spmv_task_pool_ptr) {

	int num_of_tasks = (nnz + nb - 1) / nb;
	cout << "num_of_tasks = " << num_of_tasks << endl;

	int curr_row;
	int t;
	int d;

	spmv_task * spmv_task_pool = new spmv_task[num_of_tasks];

	// Calculate the start and end index
	for (t = 0; t < num_of_tasks; t++) {
		long long tmp1 = t * (long long)nnz;
		long long tmp2 = (t + 1) * (long long)nnz;

		double tmp3 = (double)(tmp1 / num_of_tasks);
		double tmp4 = (double)(tmp2 / num_of_tasks);

		// cout << "tmp1 = " << tmp1 << endl;
		// cout << "tmp2 = " << tmp2 << endl;

		// cout << "tmp3 = " << tmp3 << endl;
		// cout << "tmp4 = " << tmp4 << endl;

		spmv_task_pool[t].start_idx = floor((double)(tmp1 / num_of_tasks));
		spmv_task_pool[t].end_idx   = floor((double)(tmp2 / num_of_tasks)) - 1;
		spmv_task_pool[t].dev_nnz = spmv_task_pool[t].end_idx - spmv_task_pool[t].start_idx + 1;

		// cout << "spmv_task_pool[t].start_idx = " << spmv_task_pool[t].start_idx << endl;
		// cout << "spmv_task_pool[t].end_idx = " << spmv_task_pool[t].end_idx << endl; 
	}

	// Calculate the start and end row
	curr_row = 0;
	for (t = 0; t < num_of_tasks; t++) {

		spmv_task_pool[t].start_row = get_row_from_index(m, csrRowPtr, spmv_task_pool[t].start_idx);
		//cout << "spmv_task_pool[t].start_row = " << spmv_task_pool[t].start_row << endl;
		// Mark imcomplete rows
		// True: imcomplete
		if (spmv_task_pool[t].start_idx > csrRowPtr[spmv_task_pool[t].start_row]) {
			spmv_task_pool[t].start_flag = true;
			spmv_task_pool[t].y2 = y[spmv_task_pool[t].start_row];
		} else {
			spmv_task_pool[t].start_flag = false;
		}
	}

	curr_row = 0;
	for (t = 0; t < num_of_tasks; t++) {
		spmv_task_pool[t].end_row = get_row_from_index(m, csrRowPtr, spmv_task_pool[t].end_idx);
		//cout << "spmv_task_pool[t].end_row = " << spmv_task_pool[t].end_row << endl;

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
		// cout << "spmv_task_pool[t].start_idx = " << spmv_task_pool[t].start_idx << endl;
		// cout << "spmv_task_pool[t].end_idx = " << spmv_task_pool[t].end_idx << endl; 
		// cout << "spmv_task_pool[t].start_row = " << spmv_task_pool[t].start_row << endl;
		// cout << "spmv_task_pool[t].end_row = " << spmv_task_pool[t].end_row << endl;
		// cout << "spmv_task_pool[t].dev_m = " << spmv_task_pool[t].dev_m << endl;
	}

	for (t = 0; t < num_of_tasks; t++) {


		//cout << "spmv_task_pool[t].dev_m + 1 = " << spmv_task_pool[t].dev_m + 1 << endl;
		spmv_task_pool[t].host_csrRowPtr = new int [spmv_task_pool[t].dev_m + 1];
		spmv_task_pool[t].host_csrRowPtr[0] = 0;
		spmv_task_pool[t].host_csrRowPtr[spmv_task_pool[t].dev_m] = spmv_task_pool[t].dev_nnz;

		memcpy(&(spmv_task_pool[t].host_csrRowPtr[1]), 
			   &csrRowPtr[spmv_task_pool[t].start_row + 1], 
			   (spmv_task_pool[t].dev_m - 1) * sizeof(int) );

		for (int j = 1; j < spmv_task_pool[t].dev_m; j++) {
			spmv_task_pool[t].host_csrRowPtr[j] -= spmv_task_pool[t].start_idx;
		}

		spmv_task_pool[t].host_csrColIndex = csrColIndex;
		spmv_task_pool[t].host_csrVal = csrVal;
		spmv_task_pool[t].host_y = y;
		spmv_task_pool[t].host_x = x;
		spmv_task_pool[t].local_result_y = new double[spmv_task_pool[t].dev_m];
		spmv_task_pool[t].alpha = new double[1];
		spmv_task_pool[t].beta = new double[1]; 
		spmv_task_pool[t].alpha[0] = *alpha;
		spmv_task_pool[t].beta[0] = *beta;

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
	// cudaSetDevice(dev_id);
	// cudaMalloc((void**)&(t->dev_csrVal),      (t->dev_nnz)   * sizeof(double));
	// cudaMalloc((void**)&(t->dev_csrRowPtr),   (t->dev_m + 1) * sizeof(int)   );
	// cudaMalloc((void**)&(t->dev_csrColIndex), (t->dev_nnz)   * sizeof(int)   );
	// cudaMalloc((void**)&(t->dev_x),           (t->dev_n)     * sizeof(double));
 //    cudaMalloc((void**)&(t->dev_y),           (t->dev_m)     * sizeof(double));

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
	//cudaSetDevice(dev_id);

	cudaStream_t stream;

	cusparseGetStream(handle, &stream);

	cout << "dev_m[d] = " << t->dev_m << endl;
	cout << "dev_n[d] = " << t->dev_n << endl;
	cudaMemcpyAsync( t->host_csrRowPtr,  t->dev_csrRowPtr, (size_t)(( t->dev_m + 1) * sizeof(int)), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(&(t->host_csrColIndex[t->start_idx]),  t->dev_csrColIndex,  (size_t)( t->dev_nnz * sizeof(int)),     cudaMemcpyDeviceToHost, stream); 
	cudaMemcpyAsync(&(t->host_csrVal[t->start_idx]),  t->dev_csrVal,            (size_t)( t->dev_nnz * sizeof(double)),  cudaMemcpyDeviceToHost, stream); 

	cudaMemcpyAsync(&(t->host_y[t->start_row]),  t->dev_y,  (size_t)( t->dev_m*sizeof(double)),  cudaMemcpyDeviceToHost, stream); 
	cudaMemcpyAsync(t->host_x, t->dev_x,                (size_t)(t->dev_n*sizeof(double)),  cudaMemcpyDeviceToHost, stream); 

	cudaDeviceSynchronize();

	cout << "dev_csrRowPtr = [";
	for (int i = 0; i < t->dev_m + 1; i++) {
		cout << t->host_csrRowPtr[i] << ", ";
	}
	cout << "]" << endl;
	cout << "csrColIndex = [";
	for (int i = 0; i < t->dev_nnz; i++) {
		cout << t->host_csrColIndex[t->start_idx+i] << ", ";
	}
	cout << "]" << endl;
	cout << "csrVal[start_idx[d]] = [";
	for (int i = 0; i < t->dev_nnz; i++) {
		cout << t->host_csrVal[t->start_idx+i] << ", ";
	}
	cout << "]" << endl;
	cout << "y[start_row[d]] = [";
	for (int i = 0; i < t->dev_m; i++) {
		cout << t->host_y[t->start_row+i] << ", ";
	}
	cout << "]" << endl;
	cout << "dev_x[d] = [";
	for (int i = 0; i < t->dev_n; i++) {
		cout << t->host_x[i] << ", ";
	}
	cout << "]" << endl;

	cout << "t->alpha = " << *(t->alpha) << endl;
	cout << "t->beta = " << *(t->beta) << endl;

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
	//cudaSetDevice(dev_id);

	cudaMemcpyAsync(t->local_result_y,   t->dev_y,          
    			   (size_t)((t->dev_m) * sizeof(double)), 
    			   cudaMemcpyDeviceToHost, stream);
	// cudaFree(t->dev_csrVal);
	// cudaFree(t->dev_csrRowPtr);
	// cudaFree(t->dev_csrColIndex);
	// cudaFree(t->dev_x);
}

void print_task_info(spmv_task * t) {
	cout << "start_idx = " << t->start_idx << endl;
	cout << "end_idx = " << t->end_idx << endl;
}
