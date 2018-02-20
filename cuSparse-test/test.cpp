#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include <iostream>
using namespace std;
#define CLEANUP(s) \
	do { \
		 printf ("%s\n", s); \
		 cudaDeviceReset(); \
		 fflush (stdout); \
	} while (0)


long int get_time()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000; //get current timestamp in milliseconds
	return ms;
}

int main(){

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
	int nb = n/deviceCount;
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
 	
 	r[0] = 0.1;
 	r[1] = 0.1;
 	r1[0] = 1;
 	r1[1] = 1;
 	r2[0] = 0.001;
 	r2[1] = 1;


 	
 	for (int d = 0; d < deviceCount; ++d) 
 	{ 
 		cudaSetDevice(d);
 		cudaStreamCreate(&(stream[d]));

 		nnz[d]=nb*r[d]*n*r1[d] + nb*(1-r[d])*n*r2[d];
	 	cooRowIndexHostPtr[d] = (int *) malloc(nnz[d]*sizeof(int));
	 	cooColIndexHostPtr[d] = (int *) malloc(nnz[d]*sizeof(int));
	 	cooValHostPtr[d] = (double *)malloc(nnz[d]*sizeof(double));

	 	if ((!cooRowIndexHostPtr[d]) || (!cooColIndexHostPtr[d]) || (!cooValHostPtr[d]))
		{
			CLEANUP("Host malloc failed (matrix)");
			return 1;
		}
		int counter = 0;
		for (int i = 0; i < nb; i++) 
		{
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

		/* create and setup matrix descriptor */ 
		status[d] = cusparseCreateMatDescr(&descr[d]);
		if (status[d] != CUSPARSE_STATUS_SUCCESS) 
		{ 
			CLEANUP("Matrix descriptor initialization failed");
			return 1;
		} 

	
		cusparseSetMatType(descr[d],CUSPARSE_MATRIX_TYPE_GENERAL); 
		cusparseSetMatIndexBase(descr[d],CUSPARSE_INDEX_BASE_ZERO); 

		/* exercise conversion routines (convert matrix from COO 2 CSR format) */ 
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
	long int start = get_time();
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
	long int end = get_time();

	cout << start << "  " << end << endl;

	long int time = end - start;


	printf("cusparseDcsrmv time = %d s\n", time);
	

	long long flop = 0;
	for (int d = 0; d < deviceCount; ++d)
	{
		flop += nnz[d] * 2;
	}
	flop *= repeat_test;
	
	double gflops = ((double)flop/1000000000) / (double)(time/1000);
	printf("GFLOPS = %f\n", gflops);


	
	
	// //y = [10 20 30 40 | 100 200 70 400] 
	// /* exercise Level 2 routines (csrmv) */

	// cudaEvent_t start, stop;
	// cudaEventCreate(&start);
	// cudaEventCreate(&stop);

	// cudaEventRecord(start);
	// for (int i = 0; i < 10; i++) 
	// {
	// 	status = cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, 
	// 							n, n, nnz, 
	// 							&dtwo, descr, cooVal, 
	// 							csrRowPtr, cooColIndex, 
	// 							&y[0], &dthree, &y[n]); 
	
	// }
	// cudaEventRecord(stop);
	// cudaEventSynchronize(stop);
	// float milliseconds = 0;
	// cudaEventElapsedTime(&milliseconds, start, stop);
	// printf("cusparseDcsrmv time = %f\n", milliseconds);
	// long long flop = nnz * 2;
	// double gflops = (flop / (milliseconds/1000))/1000000000;
	// printf("GFLOPS = %f\n", gflops);

	// cudaEventRecord(start);
	// for (int i = 0; i < 10; i++) 
	// {
	// 	status = cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, 
	// 							n, n, nnz, 
	// 							&dtwo, descr, cooVal, 
	// 							csrRowPtr, cooColIndex, 
	// 							&y[0], &dthree, &y[n]); 
	
	// }
	// cudaEventRecord(stop);
	// cudaEventSynchronize(stop);
	// milliseconds = 0;
	// cudaEventElapsedTime(&milliseconds, start, stop);
	// printf("cusparseDcsrmv_mp time = %f\n", milliseconds);
	// flop = nnz * 2;
	// gflops = (flop / (milliseconds/1000))/1000000000;
	// printf("GFLOPS = %f\n", gflops);



	// if (status != CUSPARSE_STATUS_SUCCESS) 
	// { 
	// 	CLEANUP("Matrix-vector multiplication failed");
	//  	return 1;
	// } 

	// //y = [10 20 30 40 | 680 760 1230 2240] 
	// cudaMemcpy(yHostPtr, y, (size_t)(n*sizeof(y[0])), cudaMemcpyDeviceToHost); 

	
	 }

