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


int spMV_mgpu_v1(int m, int n, long long nnz, double * alpha,
				  double * csrVal, long long * csrRowPtr, int * csrColIndex, 
				  double * x, double * beta,
				  double * y,
				  int ngpu, 
				  int kernel){

		double curr_time = 0.0;
		double time_parse = 0.0;
		double time_comm = 0.0;
		double time_comp = 0.0;
		double time_post = 0.0;


		curr_time = get_time();


		long long  * start_idx  = new long long[ngpu];
		long long  * end_idx    = new long long[ngpu];
		int        * start_row  = new int[ngpu];
		int        * end_row    = new int[ngpu];
		bool       * start_flag = new bool[ngpu];
		bool       * end_flag   = new bool[ngpu];

		//int curr_row;

		double ** dev_csrVal      = new double * [ngpu];
		int    ** host_csrRowPtr  = new int    * [ngpu];
		int    ** dev_csrRowPtr   = new int    * [ngpu];
		int    ** dev_csrColIndex = new int    * [ngpu];
		int    *         dev_nnz  = new int      [ngpu];
		int    *           dev_m  = new int      [ngpu];
		int    *           dev_n  = new int      [ngpu];

		double ** dev_x  = new double * [ngpu];
		double ** dev_y  = new double * [ngpu];
		double ** host_y = new double * [ngpu];
		double *  y2     = new double   [ngpu];

		cudaStream_t       * stream = new cudaStream_t [ngpu];
		cusparseStatus_t   * status = new cusparseStatus_t[ngpu];
		cusparseHandle_t   * handle = new cusparseHandle_t[ngpu];
		cusparseMatDescr_t * descr  = new cusparseMatDescr_t[ngpu];

		

		// tmp =  get_time() - tmp;
		// cout << "t1 = " << tmp << endl;

		//tmp = get_time();

		// Calculate the start and end index
		for (int i = 0; i < ngpu; i++) {

			long long tmp1 = i * nnz;
			long long tmp2 = (i + 1) * nnz;

			double tmp3 = (double)(tmp1 / ngpu);
			double tmp4 = (double)(tmp2 / ngpu);

			// cout << "tmp1 = " << tmp1 << endl;
			// cout << "tmp2 = " << tmp2 << endl;

			// cout << "tmp3 = " << tmp3 << endl;
			// cout << "tmp4 = " << tmp4 << endl;

			start_idx[i] = floor((double)tmp1 / ngpu);
			end_idx[i]   = floor((double)tmp2 / ngpu) - 1;
		}

		// tmp = get_time() - tmp;
		// cout << "t2 = " << tmp << endl;

		// tmp = get_time();

		// Calculate the start and end row

		//cout << "test1" << endl;

		//curr_row = 0;
		for (int i = 0; i < ngpu; i++) {
			// while (csrRowPtr[curr_row] <= start_idx[i]) {
			// 	curr_row++;
			// }

			//  start_row[i] = curr_row - 1; 
			start_row[i] = get_row_from_index(m, csrRowPtr, start_idx[i]);

			//cout << "test1-1" << endl;

			// Mark imcomplete rows
			// True: imcomplete
			//cout << "start_idx[i] = " << start_idx[i] << endl;
			//cout << "csrRowPtr[start_row[i]] = " << csrRowPtr[start_row[i]] << endl;
			//cout << "y[start_row[i]] = " << y[start_row[i]] << endl;
			//cout << "y2[i] = " << y2[i] << endl;
			if (start_idx[i] > csrRowPtr[start_row[i]]) {
				//cout << "test1-2" << endl;
				start_flag[i] = true;
				//cout << "test1-3" << endl;
				y2[i] = y[start_row[i]];
				//cout << "test1-4" << endl;
			} else {
				//cout << "test1-5" << endl;
				start_flag[i] = false;
				//cout << "test1-6" << endl;
			}
		}

		// tmp = get_time() - tmp;
		// cout << "t3 = " << tmp << endl;

		// tmp = get_time();

		//cout << "test2" << endl;

		//curr_row = 0;
		for (int i = 0; i < ngpu; i++) {
			// while (csrRowPtr[curr_row] <= end_idx[i]) {
			// 	curr_row++;
			// 	//cout << "->" << csrRowPtr[curr_row] << endl;
			// }

			// end_row[i] = curr_row - 1;
			end_row[i] = get_row_from_index(m, csrRowPtr, end_idx[i]);

			// Mark imcomplete rows
			// True: imcomplete
			if (end_idx[i] < csrRowPtr[end_row[i] + 1] - 1)  {
				end_flag[i] = true;
			} else {
				end_flag[i] = false;
			}
		}

		//cout << "test3" << endl;

		// tmp = get_time() - tmp;
		// cout << "t4 = " << tmp << endl;

		// tmp = get_time();

		// Cacluclate dimensions
		for (int i = 0; i < ngpu; i++) {
			dev_m[i] = end_row[i] - start_row[i] + 1;
			dev_n[i] = n;
		}

		//cout << "test4" << endl;

		for (int i = 0; i < ngpu; i++) {
			host_y[i] = new double[dev_m[i]];
		}

		for (int d = 0; d < ngpu; d++) {
			long long nnz_ll = end_idx[d] - start_idx[d] + 1;
			long long matrix_data_space = nnz_ll * sizeof(double) + 
										nnz_ll * sizeof(int) + 
										(long long)(dev_m[d]+1) * sizeof(int) + 
										(long long)dev_n[d] * sizeof(double) +
										(long long)dev_m[d] * sizeof(double);
			double matrix_size_in_gb = (double)matrix_data_space / 1e9;
			//cout << matrix_size_in_gb << " - " << get_gpu_availble_mem(ngpu) << endl;
			if ( matrix_size_in_gb > 0.8 * get_gpu_availble_mem(ngpu)) {
				return -1;
			}


			dev_nnz[d]   = (int)(end_idx[d] - start_idx[d] + 1);
		}


		//cout << "test5" << endl;
		// tmp = get_time() - tmp;
		// cout << "t5 = " << tmp << endl;

		// tmp = get_time();

		// for (int d = 0; d < ngpu; d++) {
		//  	cout << "GPU " << d << ":" << endl;
		// // 	cout << " start_idx: " << start_idx[d] << ", ";
		// // 	cout << " end_idx: " << end_idx[d] << ", ";
		//  	cout << " start_row: " << start_row[d] << ", ";
		//  	cout << " end_row: " << end_row[d] << ", ";
		// // 	cout << " start_flag: " << start_flag[d] << ", ";
		// // 	cout << " end_flag: " << end_flag[d] << ", ";
		//  	cout << endl;
		//  	cout << " dev_m: " << dev_m[d] << ", ";
		 // 	cout << " dev_n: " << dev_n[d] << ", ";
		////  	cout << " dev_nnz: " << dev_nnz[d] << ", ";
		//  	cout << endl;
//
		// }

		//  tmp = get_time() - tmp;
		// cout << "t6 = " << tmp << endl;

		// tmp = get_time();

		for (int i = 0; i < ngpu; i++) {
			host_csrRowPtr[i] = new int [dev_m[i] + 1];
			host_csrRowPtr[i][0] = 0;
			host_csrRowPtr[i][dev_m[i]] = dev_nnz[i];

			for (int j = 1; j < dev_m[i]; j++) {
				host_csrRowPtr[i][j] = (int)(csrRowPtr[start_row[i] + j] - start_idx[i]);
			}

			//memcpy(&host_csrRowPtr[i][1], &csrRowPtr[start_row[i] + 1], (dev_m[i] - 1) * sizeof(int) );

			// cout << "host_csrRowPtr: ";
			// for (int j = 0; j <= dev_m[i]; j++) {
			// 	cout << host_csrRowPtr[i][j] << ", ";
			// }
			// cout << endl;

			// for (int j = 1; j < dev_m[i]; j++) {
			// 	host_csrRowPtr[i][j] = (int)((long long)host_csrRowPtr[i][j] - start_idx[i]);
			// }

			// cout << "host_csrRowPtr: ";
			// for (int j = 0; j <= dev_m[i]; j++) {
			// 	cout << host_csrRowPtr[i][j] << ", ";
			// }
			// cout << endl;
		}

		//cout << "test6" << endl;

		// tmp = get_time() - tmp;
		// cout << "t7 = " << tmp << endl;

		// tmp = get_time();

		
		//curr_time = get_time();
			
		for (int d = 0; d < ngpu; d++) {

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
		}

		//cout << "aaa: " << get_time() - curr_time << endl;

		//cout << "test7" << endl;

		for (int d = 0; d < ngpu; d++) {
			cudaSetDevice(d);
			cudaMalloc((void**)&dev_csrVal[d],      dev_nnz[d]     * sizeof(double));
			cudaMalloc((void**)&dev_csrRowPtr[d],   (dev_m[d] + 1) * sizeof(int)   );
			cudaMalloc((void**)&dev_csrColIndex[d], dev_nnz[d]     * sizeof(int)   );
			cudaMalloc((void**)&dev_x[d],           dev_n[d]       * sizeof(double)); 
		    cudaMalloc((void**)&dev_y[d],           dev_m[d]       * sizeof(double)); 
		}


		time_parse = get_time() - curr_time;

		curr_time = get_time();

		//cout << "test8" << endl;


		for (int d = 0; d < ngpu; d++) {
			cudaSetDevice(d);
			cudaMemcpyAsync(dev_csrRowPtr[d],   host_csrRowPtr[d],          (size_t)((dev_m[d] + 1) * sizeof(int)), cudaMemcpyHostToDevice, stream[d]);
			cudaMemcpyAsync(dev_csrColIndex[d], &csrColIndex[start_idx[d]], (size_t)(dev_nnz[d] * sizeof(int)),     cudaMemcpyHostToDevice, stream[d]); 
			cudaMemcpyAsync(dev_csrVal[d],      &csrVal[start_idx[d]],      (size_t)(dev_nnz[d] * sizeof(double)),  cudaMemcpyHostToDevice, stream[d]); 

			cudaMemcpyAsync(dev_y[d], &y[start_row[d]], (size_t)(dev_m[d]*sizeof(double)),  cudaMemcpyHostToDevice, stream[d]); 
			cudaMemcpyAsync(dev_x[d], x,                (size_t)(dev_n[d]*sizeof(double)),  cudaMemcpyHostToDevice, stream[d]); 
		}


		// for (int d = 0; d < ngpu; d++) {
		// 	cudaSetDevice(d);
		// 	cudaMemcpy(dev_csrRowPtr[d],   host_csrRowPtr[d],          (size_t)((dev_m[d] + 1) * sizeof(int)), cudaMemcpyHostToDevice);
		// 	cudaMemcpy(dev_csrColIndex[d], &csrColIndex[start_idx[d]], (size_t)(dev_nnz[d] * sizeof(int)),     cudaMemcpyHostToDevice); 
		// 	cudaMemcpy(dev_csrVal[d],      &csrVal[start_idx[d]],      (size_t)(dev_nnz[d] * sizeof(double)),  cudaMemcpyHostToDevice); 

		// 	cudaMemcpy(dev_y[d], &y[start_row[d]], (size_t)(dev_m[d]*sizeof(double)),  cudaMemcpyHostToDevice); 
		// 	cudaMemcpy(dev_x[d], x,                (size_t)(dev_n[d]*sizeof(double)),  cudaMemcpyHostToDevice); 
		// }

		for (int d = 0; d < ngpu; ++d) 
		{
			cudaSetDevice(d);
			cudaDeviceSynchronize();
		}
		time_comm = get_time() - curr_time;

		//cout << "test9" << endl;

		curr_time = get_time();


		for (int d = 0; d < ngpu; ++d) 
		{
			// tmp = get_time();
			cudaSetDevice(d);
			if (kernel == 1) {
				status[d] = cusparseDcsrmv(handle[d],CUSPARSE_OPERATION_NON_TRANSPOSE, 
											dev_m[d], dev_n[d], dev_nnz[d], 
											alpha, descr[d], dev_csrVal[d], 
											dev_csrRowPtr[d], dev_csrColIndex[d], 
											dev_x[d],  beta, dev_y[d]); 
			} else if (kernel == 2) {
				status[d] = cusparseDcsrmv_mp(handle[d],CUSPARSE_OPERATION_NON_TRANSPOSE, 
											dev_m[d], dev_n[d], dev_nnz[d], 
											alpha, descr[d], dev_csrVal[d], 
											dev_csrRowPtr[d], dev_csrColIndex[d], 
											dev_x[d],  beta, dev_y[d]); 
			} else if (kernel == 3) {
				csr5_kernel(dev_m[d], dev_n[d], dev_nnz[d], 
							alpha, dev_csrVal[d], 
							dev_csrRowPtr[d], dev_csrColIndex[d], 
							dev_x[d],  beta, dev_y[d]); 
			}
			// cudaDeviceSynchronize();
			// cout << "computation " << d << " : " << get_time()-tmp << endl;

			// print_error(status[d]);
				
			for (int d = 0; d < ngpu; ++d) 
			{
				cudaSetDevice(d);
				cudaDeviceSynchronize();
			}
		}

		time_comp = get_time() - curr_time;

		curr_time = get_time();


		//cout << "test10" << endl;

		for (int d = 0; d < ngpu; d++) {
			double tmp = 0.0;
			
			if (start_flag[d]) {
				tmp = y[start_row[d]];
			}
	
			cudaMemcpy(&y[start_row[d]], dev_y[d], (size_t)(dev_m[d]*sizeof(double)),  cudaMemcpyDeviceToHost); 

			if (start_flag[d]) {
				y[start_row[d]] += tmp;
				y[start_row[d]] -= y2[d] * (*beta);
			}
		}

		//cout << "test11" << endl;

		// double * partial_result = new double[ngpu];
		// for (int d = 0; d < ngpu; d++) {
		// 	cudaMemcpyAsync(&partial_result[d], &dev_y[d][dev_m[d] - 1], (size_t)(1*sizeof(double)),  cudaMemcpyDeviceToHost, stream[d]); 
		// }

		// for (int d = 0; d < ngpu; d++) {
		// 	cudaMemcpyAsync(&y[start_row[d]], dev_y[d], (size_t)(dev_m[d]*sizeof(double)),  cudaMemcpyDeviceToHost, stream[d]);
		// } 

		// for (int d = 0; d < ngpu; ++d) 
		// {
		// 	cudaSetDevice(d);
		// 	cudaDeviceSynchronize();
		// }

		// for (int d = 0; d < ngpu; d++) {
		// 	if (start_flag[d]) {
		// 		y[start_row[d]] += partial_result[d - 1];
		// 		y[start_row[d]] -= y2[d] * (*beta);
		// 	}
		// }


		

		for (int d = 0; d < ngpu; d++) {
			cudaSetDevice(d);
			cudaFree(dev_csrVal[d]);
			cudaFree(dev_csrRowPtr[d]);
			cudaFree(dev_csrColIndex[d]);
			cudaFree(dev_x[d]);
			cudaFree(dev_y[d]);
			delete [] host_y[d];
			delete [] host_csrRowPtr[d];
			cusparseDestroyMatDescr(descr[d]);
			cusparseDestroy(handle[d]);
			cudaStreamDestroy(stream[d]);

		}

		

		delete[] dev_csrVal;
		delete[] dev_csrRowPtr;
		delete[] dev_csrColIndex;
		delete[] dev_x;
		delete[] dev_y;
		delete[] host_csrRowPtr;
		delete[] start_row;
		delete[] end_row;
		//delete[] host_y;
		//delete[] host_csrRowPtr;

		time_post = get_time() - curr_time;

		//cout << "time_parse = " << time_parse << ", time_comm = " << time_comm << ", time_comp = "<< time_comp <<", time_post = " << time_post << endl;



		// printf("spMV_mgpu_v2 time = %f s\n", time);
		// long long flop = nnz * 2;
		// flop *= repeat_test;
		// double gflop = (double)flop/1e9;
		// printf("gflop = %f\n", gflop);
		// double gflops = gflop / time;
		// printf("GFLOPS = %f\n", gflops);
		//return gflops;
		return 0;
	}

