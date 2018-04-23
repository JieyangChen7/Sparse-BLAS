#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include <iostream>
#include <cmath>
using namespace std;
#define CLEANUP(s) \
	do { \
		 printf ("%s\n", s); \
		 cudaDeviceReset(); \
		 fflush (stdout); \
	} while (0)


int test(int iban_gpus, double r1, double r2);
double get_time()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	double ms = (double)tp.tv_sec * 1000 + (double)tp.tv_usec / 1000; //get current timestamp in milliseconds
	return ms / 1000;
}

int main(int argc, char *argv[]){

	//for (int i = 0; i < 8; i++){
	int i = atoi(argv[1]);
	
	double r = atof(argv[2]);
	//for (double r = 0; r <= 1; r += 0.1){	
		printf("i=%d, r=%f\n", i, r);
			test(i, r, r);
	//}
	//}


}

int test(int iban_gpus, double  r11, double r22){

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int device;
	for (device = 0; device < deviceCount; ++device) 
	{
	    cudaDeviceProp deviceProp;
	    cudaGetDeviceProperties(&deviceProp, device);
	    //printf("Device %d has compute capability %d.%d.\n",
	    //       device, deviceProp.major, deviceProp.minor);
	}

	cudaStream_t * stream = new cudaStream_t [deviceCount];

	cudaError_t * cudaStat1 = new cudaError_t[deviceCount];
	cudaError_t * cudaStat2 = new cudaError_t[deviceCount];
	cudaError_t * cudaStat3 = new cudaError_t[deviceCount];
	cudaError_t * cudaStat4 = new cudaError_t[deviceCount];
	cudaError_t * cudaStat5 = new cudaError_t[deviceCount];
	cudaError_t * cudaStat6 = new cudaError_t[deviceCount];

	cusparseStatus_t * status = new cusparseStatus_t[deviceCount];
	cusparseHandle_t * handle = new cusparseHandle_t[deviceCount];
	cusparseMatDescr_t * descr = new cusparseMatDescr_t[deviceCount];

	// CPU A
	int ** cooRowIndexHostPtr = new int * [deviceCount];
	int ** cooColIndexHostPtr = new int * [deviceCount];
	double ** cooValHostPtr = new double * [deviceCount];

	// CPU x
	double * xHostPtr;

	// CPU y
	double * yHostPtr;
	
	// GPU A
	int ** cooRowIndex = new int * [deviceCount];
	int ** cooColIndex = new int * [deviceCount];
	double ** cooVal = new double * [deviceCount];
	int ** csrRowPtr = new int * [deviceCount];

	// CPU x
	double ** x = new double * [deviceCount];

	// GPU y
	double ** y = new double * [deviceCount];
	

	int n = 10000; 
	//int nb = n/deviceCount;
	int nb = 10000;
	int * nnz = new int[deviceCount];
	int nnz_vector;
	double dzero =0.0;
	double dtwo =2.0;
	double dthree=3.0;
	double dfive =5.0;

	// printf("testing example\n");
	// /* create the  sparse test matrix in COO format */

	double * r = new double [deviceCount];  //0.1
	double * r1 = new double [deviceCount]; //1
	double * r2 = new double [deviceCount]; //0.001
 	
 	r[0] = 1;
 	r[1] = 1;
	r[2] = 1;
	r[3] = 1;
	r[4] = 1;
	r[5] = 1;
	r[6] = 1;
	r[7] = 1;

	for (int i = 0; i < deviceCount; i++){
		if (i < iban_gpus){
			r1[i] = r11;
        		r2[i] = r22;
		}else{
			r1[i] = 1;
        		r2[i] = 1;
		}

	}

 	for (int d = 0; d < deviceCount; ++d) 
 	{ 
 		cudaSetDevice(d);
 		cudaStreamCreate(&(stream[d]));

 		nnz[d]=nb*r[d]*n*r1[d] + nb*(1-r[d])*n*r2[d];
	 	cooRowIndexHostPtr[d] = (int *) malloc(nnz[d]*sizeof(int));
	 	cooColIndexHostPtr[d] = (int *) malloc(nnz[d]*sizeof(int));
	 	cooValHostPtr[d] = (double *)malloc(nnz[d]*sizeof(double));
		cout <<"nnz=" << nnz[d] << endl;
//cout << "test2" << endl;


	 	if ((!cooRowIndexHostPtr[d]) || (!cooColIndexHostPtr[d]) || (!cooValHostPtr[d]))
		{
			CLEANUP("Host malloc failed (matrix)");
			return 1;
		}

		int counter = 0;
		for (int i = 0; i < nb; i++) 
		{
			if (counter < nnz[d]){
			if (i < nb * r[d]) {
				for (int j = 0; j < n * r1[d]; j++) 
				{
					
					cooRowIndexHostPtr[d][counter] = i;
					cooColIndexHostPtr[d][counter] = j;
					cooValHostPtr[d][counter] = ((double) rand() / (RAND_MAX));
					counter++;
				}
			} else {
				for (int j = 0; j < n * r2[d]; j++) 
				{
					cooRowIndexHostPtr[d][counter] = i;
					cooColIndexHostPtr[d][counter] = j;
					cooValHostPtr[d][counter] = ((double) rand() / (RAND_MAX));
					counter++;
				}
			}
			}
		}

		if (d == 0)
		{		
			xHostPtr = (double *)malloc(n * sizeof(double)); 
			yHostPtr = (double *)malloc(n * sizeof(double)); 

			if((!yHostPtr) || (!xHostPtr))
			{ 
				CLEANUP("Host malloc failed (vectors)"); 
				return 1; 
			} 

			for (int i = 0; i < n; i++)
			{
				xHostPtr[i] = ((double) rand() / (RAND_MAX)); 
				yHostPtr[i] = 0.0;
			}

		}

		cudaStat1[d] = cudaMalloc((void**)&cooRowIndex[d],nnz[d]*sizeof(int));
		cudaStat2[d] = cudaMalloc((void**)&cooColIndex[d],nnz[d]*sizeof(int)); 
		cudaStat3[d] = cudaMalloc((void**)&cooVal[d], nnz[d]*sizeof(double)); 
		cudaStat4[d] = cudaMalloc((void**)&y[d], nb*sizeof(double)); 
		cudaStat5[d] = cudaMalloc((void**)&x[d], n*sizeof(double)); 
		if ((cudaStat1[d] != cudaSuccess) || 
			(cudaStat2[d] != cudaSuccess) || 
			(cudaStat3[d] != cudaSuccess) || 
			(cudaStat4[d] != cudaSuccess) || 
			(cudaStat5[d] != cudaSuccess)) 
		{ 
			CLEANUP("Device malloc failed");
			return 1; 
		} 


		cudaStat1[d] = cudaMemcpy(cooRowIndex[d], cooRowIndexHostPtr[d], 
							  (size_t)(nnz[d]*sizeof(int)), 
							  cudaMemcpyHostToDevice);
		cudaStat2[d] = cudaMemcpy(cooColIndex[d], cooColIndexHostPtr[d], 
							  (size_t)(nnz[d]*sizeof(int)), 
							  cudaMemcpyHostToDevice); 
		cudaStat3[d] = cudaMemcpy(cooVal[d], cooValHostPtr[d], 
							  (size_t)(nnz[d]*sizeof(double)), 
							  cudaMemcpyHostToDevice); 
		cudaStat4[d] = cudaMemcpy(y[d], yHostPtr + d * nb, 
							  (size_t)(nb*sizeof(double)), 
							  cudaMemcpyHostToDevice); 
		cudaStat5[d] = cudaMemcpy(x[d], xHostPtr, 
							  (size_t)(n*sizeof(double)), 
							  cudaMemcpyHostToDevice); 

		if ((cudaStat1[d] != cudaSuccess) ||
		 	(cudaStat2[d] != cudaSuccess) ||
		  	(cudaStat3[d] != cudaSuccess) ||
		   	(cudaStat4[d] != cudaSuccess) ||
		    (cudaStat5[d] != cudaSuccess)) 
		{ 
			CLEANUP("Memcpy from Host to Device failed"); 
			return 1; 
		} 
		
		status[d] = cusparseCreate(&(handle[d])); 
		if (status[d] != CUSPARSE_STATUS_SUCCESS) 
		{ 
			CLEANUP("CUSPARSE Library initialization failed");
			return 1; 
		} 
		status[d] = cusparseSetStream(handle[d], stream[d]);
		if (status[d] != CUSPARSE_STATUS_SUCCESS) 
		{ 
			CLEANUP("Stream bindind failed");
			return 1;
		} 
		status[d] = cusparseCreateMatDescr(&descr[d]);
		if (status[d] != CUSPARSE_STATUS_SUCCESS) 
		{ 
			CLEANUP("Matrix descriptor initialization failed");
			return 1;
		} 	
		cusparseSetMatType(descr[d],CUSPARSE_MATRIX_TYPE_GENERAL); 
		cusparseSetMatIndexBase(descr[d],CUSPARSE_INDEX_BASE_ZERO); 

 
		cudaStat1[d] = cudaMalloc((void**)&csrRowPtr[d],(nb+1)*sizeof(int)); 
		if (cudaStat1[d] != cudaSuccess) 
		{ 
			CLEANUP("Device malloc failed (csrRowPtr)"); 
			return 1; 
		} 

		status[d] = cusparseXcoo2csr(handle[d],
								     cooRowIndex[d],nnz[d],nb, 
								     csrRowPtr[d],CUSPARSE_INDEX_BASE_ZERO);
		if (status[d] != CUSPARSE_STATUS_SUCCESS) 
		{ 
			CLEANUP("Conversion from COO to CSR format failed"); 
			return 1; 
		} 


	}

	int repeat_test = 10;
	double start = get_time();
	for (int i = 0; i < repeat_test; i++) 
	{
		for (int d = 0; d < deviceCount; ++d) 
		{
			cudaSetDevice(d);
			// cudaEvent_t start, stop;
			// cudaEventCreate(&start);
			// cudaEventCreate(&stop);

			//cudaEventRecord(start);
			
				status[d] = cusparseDcsrmv(handle[d],CUSPARSE_OPERATION_NON_TRANSPOSE, 
											nb, n, nnz[d], 
											&dtwo, descr[d], cooVal[d], 
											csrRowPtr[d], cooColIndex[d], 
											x[d], &dthree, y[d]); 
			// cudaEventRecord(stop);
			// cudaEventSynchronize(stop);
			// float milliseconds = 0;
			// cudaEventElapsedTime(&milliseconds, start, stop);

			
		 
		}
		for (int d = 0; d < deviceCount; ++d) 
		{
			cudaSetDevice(d);
			cudaDeviceSynchronize();
		}
	}
	double end = get_time();

	cout << start << "  " << end << endl;

	double time = end - start;

	printf("cusparseDcsrmv time = %f s\n", time);
	

	long long flop = 0;
	for (int d = 0; d < deviceCount; ++d)
	{
		flop += nnz[d] * 2;
	}
	flop *= repeat_test;
	double gflop = (double)flop/1e9;
	printf("gflop = %f\n", gflop);
	double gflops = gflop / time;
	printf("GFLOPS = %f\n", gflops);

	 }

	spMV_mgpu(char transA, 
              int m, int n, int nnz, double * alpha, 
              double * csrValA, int * csrRowPtrA, int * csrColIndA,
              double * x, double * beta, 
              double * y,
              int ngpu){
		int * start_idx = new int[ngpu];
		int * end_idx   = new int[ngpu];
		int * start_row = new int[ngpu];
		int * end_row   = new int[ngpu];

		int curr_row;

		double **    csrValA_partial = new double * [ngpu];
		int    ** csrRowPtrA_partial = new int    * [ngpu];
		int    ** csrColIndA_partial = new int    * [ngpu];
		int    *         nnz_partial = new int      [ngpu];
		int    *           m_partial = new int    * [ngpu];
		int    *           n_partial = new int    * [ngpu];

		for (int i = 0; i < ngpu; i++) {
			start_idx[i]   = floor((i)     * nnz / ngpu);
			end_idx[i]     = floor((i + 1) * nnz / ngpu) - 1;
			nnz_partial[i] = end_idx[i] - start_idx[i] + 1;
		}

		curr_row = 0;
		for (int i = 0; i < ngpu; i++) {
			while (csrRowPtrA[curr_row] < start_idx[i]) curr_row++;
			start_row[i] = curr_row - 1;
		}

		curr_row = 0;
		for (int i = 0; i < ngpu; i++) {
			while (csrRowPtrA[curr_row] < end_idx[i]) curr_row++;
			end_row[i] = curr_row - 1;
		}

		for (int i = 0; i < ngpu; i++) {
			m_partial[i] = end_row[i] - start_row[i] + 1;
			n_partial[i] = n;
		}

		for (int i = 0; i < ngpu; i++) {
			csrValA_partial[i] = new double[nnz_partial[i]];
			memcpy((void *)csrValA_partial[i], 
				   (void *)&csrValA[start_idx[i]], 
				   sizeof(double) * nnz_partial[i]);
		}

		curr_row = 0;
		for (int i = 0; i < ngpu; i++) {
			csrRowPtrA_partial[i] = new int [m_partial[i] + 1];
			csrRowPtrA_partial[i][0] = 0;
			csrRowPtrA_partial[i][m_partial[i]] = nnz_partial[i];

			for (int j = 1; j < m_partial[i]; i++) {
				csrRowPtrA_partial[i][j] = csrRowPtrA[start_row[i] + j];
			}
			for (int j = 1; j < m_partial[i]; i++) {
				csrRowPtrA_partial[i][j] -= start_idx[i];
			}
		}

		for (int i = 0; i < ngpu; i++) {
			csrColIndA_partial[i] = new double[nnz_partial[i]];
			memcpy((void *)csrColIndA_partial[i], 
				   (void *)&csrColIndA[start_idx[i]], 
				   sizeof(double) * nnz_partial[i]);
		}

		cudaStream_t       * stream = new cudaStream_t [deviceCount];
		cusparseStatus_t   * status = new cusparseStatus_t[ngpu];
		cusparseHandle_t   * handle = new cusparseHandle_t[ngpu];
		cusparseMatDescr_t * descr  = new cusparseMatDescr_t[ngpu];
		
		for (int d = 0; d < ngpu; d++) {

			cudaSetDevice(d);
 			cudaStreamCreate(&(stream[d]));

			status[d] = cusparseCreate(&(handle[d])); 
			if (status[d] != CUSPARSE_STATUS_SUCCESS) 
			{ 
				CLEANUP("CUSPARSE Library initialization failed");
				return 1; 
			} 
			status[d] = cusparseSetStream(handle[d], stream[d]);
			if (status[d] != CUSPARSE_STATUS_SUCCESS) 
			{ 
				CLEANUP("Stream bindind failed");
				return 1;
			} 
			status[d] = cusparseCreateMatDescr(&descr[d]);
			if (status[d] != CUSPARSE_STATUS_SUCCESS) 
			{ 
				CLEANUP("Matrix descriptor initialization failed");
				return 1;
			} 	
			cusparseSetMatType(descr[d],CUSPARSE_MATRIX_TYPE_GENERAL); 
			cusparseSetMatIndexBase(descr[d],CUSPARSE_INDEX_BASE_ZERO);
		}

		double **    d_csrValA_partial = new double * [ngpu];
		int    ** d_csrRowPtrA_partial = new int    * [ngpu];
		int    ** d_csrColIndA_partial = new int    * [ngpu];

		for (int i = 0; i < ngpu; i++) {
			cudaSetDevice(d);
			cudaMalloc((void**)d_csrValA_partial[i],    nnz_partial[i]     * sizeof(double));
			cudaMalloc((void**)d_csrRowPtrA_partial[i], (m_partial[i] + 1) * sizeof(int)   );
			cudaMalloc((void**)d_csrColIndA_partial[i], nnz_partial[i]     * sizeof(int)   );
		}

		int repeat_test = 10;
		double start = get_time();
		for (int i = 0; i < repeat_test; i++) 
		{
			for (int d = 0; d < ngpu; ++d) 
			{
				cudaSetDevice(d);				
				status[d] = cusparseDcsrmv(handle[d],CUSPARSE_OPERATION_NON_TRANSPOSE, 
											m_partial[d], n_partial[d], nnz_partial[d], 
											alpha, descr[d], d_csrValA_partial[d], 
											d_csrRowPtrA_partial[d], d_csrColIndA_partial[d], 
											x[d],  beta, y[d]); 
			}
			for (int d = 0; d < deviceCount; ++d) 
			{
				cudaSetDevice(d);
				cudaDeviceSynchronize();
			}
		}
		double end = get_time();

		cout << start << "  " << end << endl;

		double time = end - start;

		printf("cusparseDcsrmv time = %f s\n", time);


		


	}

