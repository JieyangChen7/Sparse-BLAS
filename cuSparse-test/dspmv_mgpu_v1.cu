#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include <iostream>
#include <cmath>
#include "mmio.h"
#include <float.h>
#include "anonymouslib_cuda.h"
#include "spmv_kernel.h"


int spMV_mgpu_v1(int m, int n, int nnz, double * alpha,
				  double * csrVal, int * csrRowPtr, int * csrColIndex, 
				  double * x, double * beta,
				  double * y,
				  int ngpu, 
				  double * time_parse,
				  double * time_comm,
				  double * time_comp,
				  double * time_post,
				  int kernel){

		double curr_time = 0.0;

		double tmp = 0.0;
		// tmp = get_time();

		curr_time = get_time();


		//tmp = get_time();


		int  * start_idx  = new int[ngpu];
		int  * end_idx    = new int[ngpu];
		int  * start_row  = new int[ngpu];
		int  * end_row    = new int[ngpu];
		bool * start_flag = new bool[ngpu];
		bool * end_flag   = new bool[ngpu];

		int curr_row;

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
			start_idx[i]   = floor((i)     * nnz / ngpu);
			end_idx[i]     = floor((i + 1) * nnz / ngpu) - 1;
			dev_nnz[i] = end_idx[i] - start_idx[i] + 1;
		}

		// tmp = get_time() - tmp;
		// cout << "t2 = " << tmp << endl;

		// tmp = get_time();

		// Calculate the start and end row
		curr_row = 0;
		for (int i = 0; i < ngpu; i++) {
			// while (csrRowPtr[curr_row] <= start_idx[i]) {
			// 	curr_row++;
			// }

			//  start_row[i] = curr_row - 1; 
			start_row[i] = get_row_from_index(m, csrRowPtr, start_idx[i]);

			// Mark imcomplete rows
			// True: imcomplete
			if (start_idx[i] > csrRowPtr[start_row[i]]) {
				start_flag[i] = true;
				y2[i] = y[start_idx[i]];
			} else {
				start_flag[i] = false;
			}
		}

		// tmp = get_time() - tmp;
		// cout << "t3 = " << tmp << endl;

		// tmp = get_time();

		curr_row = 0;
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

		// tmp = get_time() - tmp;
		// cout << "t4 = " << tmp << endl;

		// tmp = get_time();

		// Cacluclate dimensions
		for (int i = 0; i < ngpu; i++) {
			dev_m[i] = end_row[i] - start_row[i] + 1;
			dev_n[i] = n;
		}

		for (int i = 0; i < ngpu; i++) {
			host_y[i] = new double[dev_m[i]];
		}

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

			// for (int j = 1; j < dev_m[i]; j++) {
			// 	host_csrRowPtr[i][j] = csrRowPtr[start_row[i] + j];
			// }

			memcpy(&host_csrRowPtr[i][1], &csrRowPtr[start_row[i] + 1], (dev_m[i] - 1)* sizeof(int) );

			// cout << "host_csrRowPtr: ";
			// for (int j = 0; j <= dev_m[i]; j++) {
			// 	cout << host_csrRowPtr[i][j] << ", ";
			// }
			// cout << endl;

			for (int j = 1; j < dev_m[i]; j++) {
				host_csrRowPtr[i][j] -= start_idx[i];
			}

			// cout << "host_csrRowPtr: ";
			// for (int j = 0; j <= dev_m[i]; j++) {
			// 	cout << host_csrRowPtr[i][j] << ", ";
			// }
			// cout << endl;
		}

		// tmp = get_time() - tmp;
		// cout << "t7 = " << tmp << endl;

		// tmp = get_time();

		*time_parse = get_time() - curr_time;

		curr_time = get_time();

			
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

		for (int d = 0; d < ngpu; d++) {
			cudaSetDevice(d);
			cudaMalloc((void**)&dev_csrVal[d],      dev_nnz[d]     * sizeof(double));
			cudaMalloc((void**)&dev_csrRowPtr[d],   (dev_m[d] + 1) * sizeof(int)   );
			cudaMalloc((void**)&dev_csrColIndex[d], dev_nnz[d]     * sizeof(int)   );
			cudaMalloc((void**)&dev_x[d],           dev_n[d]       * sizeof(double)); 
		    cudaMalloc((void**)&dev_y[d],           dev_m[d]       * sizeof(double)); 
		}


	


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
		*time_comm = get_time() - curr_time;


		curr_time = get_time();


		int repeat_test = 1;
		for (int i = 0; i < repeat_test; i++) 
		{
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
					int err = 0;
					// cout << "before CSR5" << endl;
					// cout << "dev_m[d] = " << dev_m[d] << endl;
					// cout << "dev_n[d] = " << dev_n[d] << endl;
					// cudaMemcpyAsync(host_csrRowPtr[d], dev_csrRowPtr[d], (size_t)((dev_m[d] + 1) * sizeof(int)), cudaMemcpyDeviceToHost, stream[d]);
					// cudaMemcpyAsync(&csrColIndex[start_idx[d]], dev_csrColIndex[d],  (size_t)(dev_nnz[d] * sizeof(int)),     cudaMemcpyDeviceToHost, stream[d]); 
					// cudaMemcpyAsync(&csrVal[start_idx[d]], dev_csrVal[d],            (size_t)(dev_nnz[d] * sizeof(double)),  cudaMemcpyDeviceToHost, stream[d]); 

					// cudaMemcpyAsync(&y[start_row[d]], dev_y[d],  (size_t)(dev_m[d]*sizeof(double)),  cudaMemcpyDeviceToHost, stream[d]); 
					// cudaMemcpyAsync(x, dev_x[d],                (size_t)(dev_n[d]*sizeof(double)),  cudaMemcpyDeviceToHost, stream[d]); 

					// cudaDeviceSynchronize();

					// cout << "dev_csrRowPtr = [";
					// for (int i = 0; i < dev_m[d] + 1; i++) {
					// 	cout << host_csrRowPtr[d][i] << ", ";
					// }
					// cout << "]" << endl;
					// cout << "csrColIndex = [";
					// for (int i = 0; i < dev_nnz[d]; i++) {
					// 	cout << csrColIndex[start_idx[d]+i] << ", ";
					// }
					// cout << "]" << endl;
					// cout << "csrVal[start_idx[d]] = [";
					// for (int i = 0; i < dev_nnz[d]; i++) {
					// 	cout << csrVal[start_idx[d]+i] << ", ";
					// }
					// cout << "]" << endl;
					// cout << "y[start_row[d]] = [";
					// for (int i = 0; i < dev_m[d]; i++) {
					// 	cout << y[start_row[d]+i] << ", ";
					// }
					// cout << "]" << endl;
					// cout << "dev_x[d] = [";
					// for (int i = 0; i < dev_n[d]; i++) {
					// 	cout << x[i] << ", ";
					// }
					// cout << "]" << endl;

					anonymouslibHandle<int, unsigned int, double> A(dev_m[d], dev_n[d]);
					err = A.inputCSR(
						dev_nnz[d], 
						dev_csrRowPtr[d], 
						dev_csrColIndex[d], 
						dev_csrVal[d]);
					//cout << "inputCSR err = " << err << endl;
					err = A.setX(dev_x[d]);
					//cout << "setX err = " << err << endl;
					A.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA);
					A.warmup();
					err = A.asCSR5();
					//cout << "asCSR5 err = " << err << endl;
					err = A.spmv(*alpha, dev_y[d]);
					//cout << "spmv err = " << err << endl;

					// cudaMemcpyAsync(&y[start_row[d]], dev_y[d],  (size_t)(dev_m[d]*sizeof(double)),  cudaMemcpyDeviceToHost, stream[d]); 
					// cudaDeviceSynchronize();
					// cout << "after:" <<endl;
					// cout << "y[start_row[d]] = [";
					// 			for (int i = 0; i < dev_m[d]; i++) {
					// 				cout << y[start_row[d]+i] << ", ";
					// 			}
					// cout << "]" << endl;		

				}
				// cudaDeviceSynchronize();
				// cout << "computation " << d << " : " << get_time()-tmp << endl;

				// print_error(status[d]);
				
			}
			for (int d = 0; d < ngpu; ++d) 
			{
				cudaSetDevice(d);
				cudaDeviceSynchronize();
			}
		}

		*time_comp = get_time() - curr_time;

		curr_time = get_time();




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


		*time_post = get_time() - curr_time;

		// for (int d = 0; d < ngpu; d++) {
		// 	cudaSetDevice(d);
		// 	cudafree(dev_csrVal[d]);
		// 	cudafree(dev_csrRowPtr[d]);
		// 	cudafree(dev_csrColIndex[d]);
		// 	cudafree(dev_x[d]);
		// 	cudafree(dev_y[d]);
		// 	delete [] host_csrRowPtr[d]
		// }
		// delete[] dev_csrVal;
		// delete[] dev_csrRowPtr;
		// delete[] dev_csrColIndex;
		// delete[] dev_x;
		// delete[] dev_y;
		// delete[] host_csrRowPtr;
		// delete[] start_row;
		// delete[] end_row;



		// printf("spMV_mgpu_v2 time = %f s\n", time);
		// long long flop = nnz * 2;
		// flop *= repeat_test;
		// double gflop = (double)flop/1e9;
		// printf("gflop = %f\n", gflop);
		// double gflops = gflop / time;
		// printf("GFLOPS = %f\n", gflops);
		//return gflops;

	}

