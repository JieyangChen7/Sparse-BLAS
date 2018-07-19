#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include <iostream>
#include <cmath>
#include <float.h>
//#include "anonymouslib_cuda.h"
#include "spmv_kernel.h"
using namespace std;

int spMV_mgpu_baseline(int m, int n, int nnz, double * alpha,
				 double * csrVal, int * csrRowPtr, int * csrColIndex, 
				 double * x, double * beta,
				 double * y,
				 int ngpu,
				 double * time_parse,
				 double * time_comm,
				 double * time_comp,
				 double * time_post){

	double curr_time = 0.0;
	double tmp = 0.0;

	curr_time = get_time();


	cudaStream_t * stream = new cudaStream_t [ngpu];

	cudaError_t * cudaStat1 = new cudaError_t[ngpu];
	cudaError_t * cudaStat2 = new cudaError_t[ngpu];
	cudaError_t * cudaStat3 = new cudaError_t[ngpu];
	cudaError_t * cudaStat4 = new cudaError_t[ngpu];
	cudaError_t * cudaStat5 = new cudaError_t[ngpu];
	cudaError_t * cudaStat6 = new cudaError_t[ngpu];

	cusparseStatus_t * status = new cusparseStatus_t[ngpu];
	cusparseHandle_t * handle = new cusparseHandle_t[ngpu];
	cusparseMatDescr_t * descr = new cusparseMatDescr_t[ngpu];

	int  * start_row  = new int[ngpu];
	int  * end_row    = new int[ngpu];
		
	int * dev_m            = new int      [ngpu];
	int * dev_n            = new int      [ngpu];
	int * dev_nnz          = new int      [ngpu];
	int ** host_csrRowPtr  = new int    * [ngpu];
	int ** dev_csrRowPtr   = new int    * [ngpu];
	int ** dev_csrColIndex = new int    * [ngpu];
	double ** dev_csrVal   = new double * [ngpu];


	double ** dev_x = new double * [ngpu];
	double ** dev_y = new double * [ngpu];

	for (int d = 0; d < ngpu; d++){

		cudaSetDevice(d);

		cout << "GPU " << d << ":" << endl;
		start_row[d] = floor((d)     * m / ngpu);
		end_row[d]   = floor((d + 1) * m / ngpu) - 1;

		cout << "start_row: " << start_row[d] << ", " << "end_row: "<< end_row[d] << endl;

		dev_m[d]   = end_row[d] - start_row[d] + 1;
		dev_n[d]   = n;
		dev_nnz[d] = csrRowPtr[end_row[d] + 1] - csrRowPtr[start_row[d]];

		cout << "dev_nnz[d] = " << dev_nnz[d] << " = " << csrRowPtr[end_row[d] + 1] << " - " << csrRowPtr[start_row[d]] << endl;

		cout << "dev_m[d]: " << dev_m[d] << ", dev_n[d]: " << dev_n[d] << ", dev_nnz[d]: " << dev_nnz[d] << endl;

		host_csrRowPtr[d] = new int[dev_m[d] + 1];

		memcpy((void *)host_csrRowPtr[d], 
			   (void *)&csrRowPtr[start_row[d]], 
			   (dev_m[d] + 1) * sizeof(int));

		cout << "csrRowPtr (before): ";
		for (int i = 0; i <= dev_m[d]; i++) {
			cout << host_csrRowPtr[d][i] << ", ";
		}
		cout << endl;

		for (int i = 0; i < dev_m[d] + 1; i++) {
			host_csrRowPtr[d][i] -= csrRowPtr[start_row[d]];
		}

		cout << "csrRowPtr (after): ";
		for (int i = 0; i <= dev_m[d]; i++) {
			cout << host_csrRowPtr[d][i] << ", ";
		}
		cout << endl;

	}


	*time_parse = get_time() - curr_time;
	curr_time = get_time();

	for (int d = 0; d < ngpu; d++){
		cudaSetDevice(d);

		cudaStreamCreate(&(stream[d]));
		
		status[d] = cusparseCreate(&(handle[d])); 
		if (status[d] != CUSPARSE_STATUS_SUCCESS) 
		{ 
			printf("CUSPARSE Library initialization failed");
			return 1; 
		} 
		status[d] = cusparseSetStream(handle[d], stream[d]);
		if (status[d] != CUSPARSE_STATUS_SUCCESS) 
		{ 
			printf("Stream bindind failed");
			return 1;
		} 
		status[d] = cusparseCreateMatDescr(&descr[d]);
		if (status[d] != CUSPARSE_STATUS_SUCCESS) 
		{ 
			printf("Matrix descriptor initialization failed");
			return 1;
		} 	
		cusparseSetMatType(descr[d],CUSPARSE_MATRIX_TYPE_GENERAL); 
		cusparseSetMatIndexBase(descr[d],CUSPARSE_INDEX_BASE_ZERO); 

		cudaStat1[d] = cudaMalloc((void**)&dev_csrRowPtr[d],   (dev_m[d] + 1) * sizeof(int));
		cudaStat2[d] = cudaMalloc((void**)&dev_csrColIndex[d],100 * sizeof(int)); 
		cudaStat3[d] = cudaMalloc((void**)&dev_csrVal[d],      dev_nnz[d] * sizeof(double)); 

		cudaStat4[d] = cudaMalloc((void**)&dev_x[d],           dev_n[d] * sizeof(double)); 
		cudaStat5[d] = cudaMalloc((void**)&dev_y[d],           dev_m[d] * sizeof(double)); 
		

		if ((cudaStat1[d] != cudaSuccess) || 
			(cudaStat2[d] != cudaSuccess) || 
			(cudaStat3[d] != cudaSuccess) || 
			(cudaStat4[d] != cudaSuccess) || 
			(cudaStat5[d] != cudaSuccess)) 
		{ 
			printf("Device malloc failed");
			return 1; 
		} 

		cout << "Start copy to GPUs...";
		cudaStat1[d] = cudaMemcpy(dev_csrRowPtr[d],   host_csrRowPtr[d],                  (size_t)((dev_m[d] + 1) * sizeof(int)), cudaMemcpyHostToDevice);
		if (cudaStat1[d] != cudaSuccess) cout << "error 1" << endl;
		cout << "host_csrRowPtr[d] = ";
		for (int i = 0; i < dev_m[d] + 1; ++i)
		{
			cout << host_csrRowPtr[d][i] << ", ";
		}
		cout << endl;
		//cudaStat2[d] = cudaMemcpy(dev_csrColIndex[d], &csrColIndex[csrRowPtr[start_row[d]]], (size_t)(dev_nnz[d] * sizeof(int)),   cudaMemcpyHostToDevice); 
		for (int i = 0 ; i<dev_nnz[d]; i+=1) {
			cudaStat2[d] = cudaMemcpy(dev_csrColIndex[d], &csrColIndex[csrRowPtr[start_row[d]]], i*sizeof(int),   cudaMemcpyHostToDevice); 
			
			if (cudaStat2[d] != cudaSuccess) cout << "i=" << i <<" error 2 " << cudaStat2[d] <<  endl;
		}
		cout << "csrColIndex[d] = ";
		for (int i = 0; i < dev_nnz[d]; ++i)
		{
			cout << csrColIndex[csrRowPtr[start_row[d]]+i] << ", ";
		}
		cout << endl;
		cudaStat3[d] = cudaMemcpy(dev_csrVal[d],      &csrVal[csrRowPtr[start_row[d]]],      (size_t)(dev_nnz[d] * sizeof(double)), cudaMemcpyHostToDevice);
		if (cudaStat3[d] != cudaSuccess) cout << "error 3 " << cudaStat3[d] <<  endl; 

		cout << "csrVal[d] = ";
		for (int i = 0; i < dev_nnz[d]; ++i)
		{
			cout << csrVal[csrRowPtr[start_row[d]]+i] << ", ";
		}
		cout << endl;


		cudaStat4[d] = cudaMemcpy(dev_y[d], &y[start_row[d]], (size_t)(dev_m[d]*sizeof(double)), cudaMemcpyHostToDevice); 
		if (cudaStat4[d] != cudaSuccess) cout << "error 4" << endl;

		cudaStat5[d] = cudaMemcpy(dev_x[d], x,                (size_t)(dev_n[d]*sizeof(double)), cudaMemcpyHostToDevice); 
		if (cudaStat5[d] != cudaSuccess) cout << "error 5" << endl;

		// cout << "x = ";
		// for (int i = 0; i < dev_n[d]; ++i)
		// {
		// 	cout << x[i] << ", ";
		// }
		// cout << endl;

		// cout << "y = ";
		// for (int i = 0; i < dev_m[d]; ++i)
		// {
		// 	cout << y[i] << ", ";
		// }
		// cout << endl;

		if ((cudaStat1[d] != cudaSuccess) ||
		 	(cudaStat2[d] != cudaSuccess) ||
		  	(cudaStat3[d] != cudaSuccess) ||
		   	(cudaStat4[d] != cudaSuccess) ||
		    (cudaStat5[d] != cudaSuccess)) 
		{ 
			printf("Memcpy from Host to Device failed"); 
			//return 1; 
		} 

	}

	*time_comm = get_time() - curr_time;
	curr_time = get_time();


	cout << "Start computation ... " << endl;
	 int repeat_test = 1;
	 double start = get_time();
	 for (int i = 0; i < repeat_test; i++) 
	 {
		for (int d = 0; d < ngpu; ++d) 
		{
			//tmp = get_time();
			cudaSetDevice(d);
			//cout << "dev_m[d]: " << dev_m[d] << ", dev_n[d]: " << dev_n[d] << ", dev_nnz[d]: " << dev_nnz[d] << endl;
			status[d] = cusparseDcsrmv(handle[d],CUSPARSE_OPERATION_NON_TRANSPOSE, 
									   dev_m[d], dev_n[d], dev_nnz[d], 
									   alpha, descr[d], dev_csrVal[d], 
									   dev_csrRowPtr[d], dev_csrColIndex[d], 
									   dev_x[d], beta, dev_y[d]);		 	
			// cudaDeviceSynchronize();
			// cout << "computation " << d << " : " << get_time()-tmp << endl;
			
		 	
		}
		for (int d = 0; d < ngpu; ++d) 
		{
			cudaSetDevice(d);
			cudaDeviceSynchronize();
		}


	}

	*time_comp = get_time() - curr_time;
	curr_time = get_time();

	for (int d = 0; d < ngpu; d++)
	{
		cudaMemcpy( &y[start_row[d]], dev_y[d], (size_t)(dev_m[d]*sizeof(double)),  cudaMemcpyDeviceToHost);
	}

	


	// double end = get_time();
	// double time = end - start;
	// printf("spMV_mgpu_v1 time = %f s\n", time);	
	// long long flop = nnz * 2;
	// flop *= repeat_test;
	// double gflop = (double)flop/1e9;
	// printf("gflop = %f\n", gflop);
	// double gflops = gflop / time;
	// printf("GFLOPS = %f\n", gflops);
	// return gflops;

	for (int d = 0; d < ngpu; d++) {
		cudaSetDevice(d);
		cudaFree(dev_csrVal[d]);
		cudaFree(dev_csrRowPtr[d]);
		cudaFree(dev_csrColIndex[d]);
		cudaFree(dev_x[d]);
		cudaFree(dev_y[d]);
	}

	*time_post = get_time() - curr_time;
	// 	delete[] dev_csrVal;
	// 	delete[] dev_csrRowPtr;
	// 	delete[] dev_csrColIndex;
	// 	delete[] dev_x;
	// 	delete[] dev_y;
	// 	delete[] host_csrRowPtr;
	// 	delete[] start_row;
	// 	delete[] end_row;
		
	

}