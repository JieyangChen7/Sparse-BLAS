#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#define CLEANUP(s) \
	do { \
		 printf ("%s\n", s); \
		 if (yHostPtr) free(yHostPtr); \
		 if (zHostPtr) free(zHostPtr); \
		 if (xIndHostPtr) free(xIndHostPtr); \
		 if (xValHostPtr) free(xValHostPtr); \
		 if (cooRowIndexHostPtr) free(cooRowIndexHostPtr);\
		 if (cooColIndexHostPtr) free(cooColIndexHostPtr);\
		 if (cooValHostPtr) free(cooValHostPtr); \
		 if (y) cudaFree(y); \
		 if (z) cudaFree(z); \
		 if (xInd) cudaFree(xInd); \
		 if (xVal) cudaFree(xVal); \
		 if (csrRowPtr) cudaFree(csrRowPtr); \
		 if (cooRowIndex) cudaFree(cooRowIndex); \
		 if (cooColIndex) cudaFree(cooColIndex); \
		 if (cooVal) cudaFree(cooVal); \
		 if (descr) cusparseDestroyMatDescr(descr);\
		 if (handle) cusparseDestroy(handle); \
		 cudaDeviceReset(); \
		 fflush (stdout); \
	} while (0)


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
	int nb = 5000;
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
 	r2[0] = 0.1;
 	r2[1] = 0.1;
 	
 	for (int d = 0; d < deviceCount; ++d) 
 	{ 
 		cudaSetDevice(d);
 		cudaStreamCreate(stream[d]);

 		nnz[d]=nb*r[d]*n*r1[d] + nb*(1-r[d])*n*r2[d];
	 	cooRowIndexHostPtr[d] = (int *) malloc(nnz*sizeof(int));
	 	cooColIndexHostPtr[d] = (int *) malloc(nnz*sizeof(int))
	 	cooValHostPtr[d] = (double *)malloc(nnz*sizeof(double));

	 	if ((!cooRowIndexHostPtr[d]) || (!cooColIndexHostPtr[d]) || (!cooValHostPtr[d]))
		{
			CLEANUP("Host malloc failed (matrix)");
			return 1;
		}
		int counter = 0;
		for (int i = 0; i < nb; i++) 
		{
			if (i < n * r[d]) {
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
		cudaStat2[d] = cudaMalloc((void**)&cooColIndex[dd],nnz[d]*sizeof(int)); 
		cudaStat3[d] = cudaMalloc((void**)&cooVal[d], nnz[d]*sizeof(double)); 
		cudaStat4[d] = cudaMalloc((void**)&y[d], nb*sizeof(double)); 
		cudaStat5[d] = cudaMalloc((void**)&x[d], n*sizeof(int)); 
		if ((cudaStat1 != cudaSuccess) || 
			(cudaStat2 != cudaSuccess) || 
			(cudaStat3 != cudaSuccess) || 
			(cudaStat4 != cudaSuccess) || 
			(cudaStat5 != cudaSuccess)) 
		{ 
			CLEANUP("Device malloc failed");
			return 1; 
		} 

		cudaStat1 = cudaMemcpy(cooRowIndex[d], cooRowIndexHostPtr[d], 
							  (size_t)(nnz[d]*sizeof(int)), 
							  cudaMemcpyHostToDevice);
		cudaStat2 = cudaMemcpy(cooColIndex[d], cooColIndexHostPtr[d], 
							  (size_t)(nnz[d]*sizeof(int)), 
							  cudaMemcpyHostToDevice); 
		cudaStat3 = cudaMemcpy(cooVal[d], cooValHostPtr[d], 
							  (size_t)(nnz[d]*sizeof(double)), 
							  cudaMemcpyHostToDevice); 
		cudaStat4 = cudaMemcpy(y[d], yHostPtr + d * nb, 
							  (size_t)(nb*sizeof(double)), 
							  cudaMemcpyHostToDevice); 
		cudaStat5 = cudaMemcpy(x[d], xHostPtr, 
							  (size_t)(n*sizeof(int])), 
							  cudaMemcpyHostToDevice); 

		if ((cudaStat1 != cudaSuccess) ||
		 	(cudaStat2 != cudaSuccess) ||
		  	(cudaStat3 != cudaSuccess) ||
		   	(cudaStat4 != cudaSuccess) ||
		    (cudaStat5 != cudaSuccess)) 
		{ 
			CLEANUP("Memcpy from Host to Device failed"); 
			return 1; 
		} 

		/* initialize cusparse library */ 
		status = cusparseCreate(&handle[d]); 
		if (status != CUSPARSE_STATUS_SUCCESS) 
		{ 
			CLEANUP("CUSPARSE Library initialization failed");
			return 1; 
		} 

		status = cusparseSetStream(handle[d], stream[d]);
		if (status != CUSPARSE_STATUS_SUCCESS) 
		{ 
			CLEANUP("Stream bindind failed");
			return 1;
		} 

		/* create and setup matrix descriptor */ 
		status = cusparseCreateMatDescr(&descr[d]);
		if (status != CUSPARSE_STATUS_SUCCESS) 
		{ 
			CLEANUP("Matrix descriptor initialization failed");
			return 1;
		} 

	
		cusparseSetMatType(descr[d],CUSPARSE_MATRIX_TYPE_GENERAL); 
		cusparseSetMatIndexBase(descr[d],CUSPARSE_INDEX_BASE_ZERO); 

		/* exercise conversion routines (convert matrix from COO 2 CSR format) */ 
		cudaStat1 = cudaMalloc((void**)&csrRowPtr[d],(nb+1)*sizeof(int)); 
		if (cudaStat1 != cudaSuccess) 
		{ 
			CLEANUP("Device malloc failed (csrRowPtr)"); 
			return 1; 
		} 

		status= cusparseXcoo2csr(handle,
								cooRowIndex[d],nnz[d],nb, 
								csrRowPtr[d],CUSPARSE_INDEX_BASE_ZERO);
		if (status != CUSPARSE_STATUS_SUCCESS) 
		{ 
			CLEANUP("Conversion from COO to CSR format failed"); 
			return 1; 
		} 


		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
		for (int i = 0; i < 10; i++) 
		{
			status = cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, 
									nb, n, nnz[d], 
									&dtwo, descr[d], cooVal[d], 
									csrRowPtr[d], cooColIndex[d], 
									&x[d], &dthree, &y[d]); 
		
		}
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("cusparseDcsrmv time = %f\n", milliseconds);
		long long flop = nnz * 2;
		double gflops = (flop / (milliseconds/1000))/1000000000;
		printf("GFLOPS = %f\n", gflops);
	 
	}
	
	

	
	
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

