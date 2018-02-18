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
	for (device = 0; device < deviceCount; ++device) {
	    cudaDeviceProp deviceProp;
	    cudaGetDeviceProperties(&deviceProp, device);
	    printf("Device %d has compute capability %d.%d.\n",
	           device, deviceProp.major, deviceProp.minor);
	}



	// cudaError_t cudaStat1,cudaStat2,cudaStat3,cudaStat4,cudaStat5,cudaStat6;
	// cusparseStatus_t * status;
	// cusparseHandle_t * handle=0;
	// cusparseMatDescr_t * descr=0;

	// // CPU A
	// int * cooRowIndexHostPtr=0;
	// int * cooColIndexHostPtr=0;
	// double * cooValHostPtr=0;

	// // CPU x
	// int * xIndHostPtr=0;
	// double * xValHostPtr=0;

	// // CPU y
	// double * yHostPtr=0;
	
	// // GPU A
	// int * cooRowIndex=0;
	// int * cooColIndex=0;
	// double * cooVal=0;
	// int * csrRowPtr=0;

	// // GPU x
	// int * xInd=0;
	// double * xVal=0;
	
	// // GPU y
	// double * y=0;
	

	// int n, nnz, nnz_vector;
	// double dzero =0.0;
	// double dtwo =2.0;
	// double dthree=3.0;
	// double dfive =5.0;

	// printf("testing example\n");
	// /* create the  sparse test matrix in COO format */
	// double r = 0.1;
	// double r1 = 1;
	// double r2 = 0.001;
 //  	n=10000; nnz=n*r*n*r1 + n*(1-r)*n*r2;
 // 	cooRowIndexHostPtr = (int *) malloc(nnz*sizeof(cooRowIndexHostPtr[0]));
 // 	cooColIndexHostPtr = (int *) malloc(nnz*sizeof(cooColIndexHostPtr[0]));
 // 	cooValHostPtr = (double *)malloc(nnz*sizeof(cooValHostPtr[0]));
	// if ((!cooRowIndexHostPtr) || (!cooColIndexHostPtr) || (!cooValHostPtr))
	// {
	// 	CLEANUP("Host malloc failed (matrix)");
	// 	return 1;
	// }
	// int counter = 0;
	// for (int i = 0; i < n; i++) 
	// {
	// 	if (i < n * r) {
	// 		for (int j = 0; j < n * r1 ; j++) 
	// 		{
	// 			cooRowIndexHostPtr[counter] = i;
	// 			cooColIndexHostPtr[counter] = j;
	// 			cooValHostPtr[counter] = ((double) rand() / (RAND_MAX));
	// 			counter++;
	// 		}
	// 	} else {
	// 		for (int j = 0; j < n * r2 ; j++) 
	// 		{
	// 			cooRowIndexHostPtr[counter] = i;
	// 			cooColIndexHostPtr[counter] = j;
	// 			cooValHostPtr[counter] = ((double) rand() / (RAND_MAX));
	// 			counter++;
	// 		}
	// 	}
	// }
	 
	// nnz_vector = n; 
	// xIndHostPtr = (int *) malloc(nnz_vector*sizeof(xIndHostPtr[0])); 
	// xValHostPtr = (double *)malloc(nnz_vector*sizeof(xValHostPtr[0])); 
	// yHostPtr = (double *)malloc(n *sizeof(yHostPtr[0])); 
	// zHostPtr = (double *)malloc(2*(n+1) *sizeof(zHostPtr[0])); 
	// if((!xIndHostPtr) || (!xValHostPtr) || (!yHostPtr) || (!zHostPtr))
	// { 
	// 	CLEANUP("Host malloc failed (vectors)"); 
	// 	return 1; 
	// } 
	
	// for (int i = 0; i < n; i++)
	// {
	// 	xIndHostPtr[i] = i; 
	// 	xValHostPtr[i] = ((double) rand() / (RAND_MAX)); 
	// 	yHostPtr[i] = ((double) rand() / (RAND_MAX));
	// }
	
	// /* allocate GPU memory and copy the matrix and vectors into it */ 
	// cudaStat1 = cudaMalloc((void**)&cooRowIndex,nnz*sizeof(cooRowIndex[0]));
	// cudaStat2 = cudaMalloc((void**)&cooColIndex,nnz*sizeof(cooColIndex[0])); 
	// cudaStat3 = cudaMalloc((void**)&cooVal, nnz*sizeof(cooVal[0])); 
	// cudaStat4 = cudaMalloc((void**)&y, n*sizeof(y[0])); 
	// cudaStat5 = cudaMalloc((void**)&xInd,nnz_vector*sizeof(xInd[0])); 
	// cudaStat6 = cudaMalloc((void**)&xVal,nnz_vector*sizeof(xVal[0])); 
	// if ((cudaStat1 != cudaSuccess) || 
	// (cudaStat2 != cudaSuccess) || 
	// (cudaStat3 != cudaSuccess) || 
	// (cudaStat4 != cudaSuccess) || 
	// (cudaStat5 != cudaSuccess) || 
	// (cudaStat6 != cudaSuccess)) 
	// { 
	// 	CLEANUP("Device malloc failed");
	// 	return 1; 
	// } 

	// cudaStat1 = cudaMemcpy(cooRowIndex, cooRowIndexHostPtr, 
	// 					(size_t)(nnz*sizeof(cooRowIndex[0])), 
	// 					cudaMemcpyHostToDevice);
	// cudaStat2 = cudaMemcpy(cooColIndex, cooColIndexHostPtr, 
	// 					(size_t)(nnz*sizeof(cooColIndex[0])), 
	// 					cudaMemcpyHostToDevice); 
	// cudaStat3 = cudaMemcpy(cooVal, cooValHostPtr, 
	// 					(size_t)(nnz*sizeof(cooVal[0])), 
	// 					cudaMemcpyHostToDevice); 
	// cudaStat4 = cudaMemcpy(y, yHostPtr, 
	// 					(size_t)(n*sizeof(y[0])), 
	// 					cudaMemcpyHostToDevice); 
	// cudaStat5 = cudaMemcpy(xInd, xIndHostPtr, 
	// 					(size_t)(nnz_vector*sizeof(xInd[0])), 
	// 					cudaMemcpyHostToDevice); 
	// cudaStat6 = cudaMemcpy(xVal, xValHostPtr, 
	// 					(size_t)(nnz_vector*sizeof(xVal[0])), 
	// 					cudaMemcpyHostToDevice); 

	// if ((cudaStat1 != cudaSuccess) ||
	//  	(cudaStat2 != cudaSuccess) ||
	//   	(cudaStat3 != cudaSuccess) ||
	//    	(cudaStat4 != cudaSuccess) ||
	//     (cudaStat5 != cudaSuccess) ||
	//     (cudaStat6 != cudaSuccess)) 
	// { 
	// 	CLEANUP("Memcpy from Host to Device failed"); 
	// 	return 1; 
	// } 


	// /* initialize cusparse library */ 
	// status= cusparseCreate(&handle); 
	// if (status != CUSPARSE_STATUS_SUCCESS) 
	// { 
	// 	CLEANUP("CUSPARSE Library initialization failed");
	// 	return 1; 
	// } 

	// /* create and setup matrix descriptor */ 
	// status= cusparseCreateMatDescr(&descr);
	// if (status != CUSPARSE_STATUS_SUCCESS) 
	// { 
	// 	CLEANUP("Matrix descriptor initialization failed");
	// 	return 1;
	// } 

	// cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL); 
	// cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO); 

	// /* exercise conversion routines (convert matrix from COO 2 CSR format) */ 
	// cudaStat1 = cudaMalloc((void**)&csrRowPtr,(n+1)*sizeof(csrRowPtr[0])); 
	// if (cudaStat1 != cudaSuccess) 
	// { 
	// 	CLEANUP("Device malloc failed (csrRowPtr)"); 
	// 	return 1; 
	// } 

	// status= cusparseXcoo2csr(handle,
	// 						cooRowIndex,nnz,n, 
	// 						csrRowPtr,CUSPARSE_INDEX_BASE_ZERO);
	// if (status != CUSPARSE_STATUS_SUCCESS) 
	// { 
	// 	CLEANUP("Conversion from COO to CSR format failed"); 
	// 	return 1; 
	// } 

	// //csrRowPtr = [0 3 4 7 9] 
	// // The following test only works for compute capability 1.3 and above 
	// // because it needs double precision. 
	// int devId; 
	// cudaDeviceProp prop; 
	// cudaError_t cudaStat; 
	// cudaStat = cudaGetDevice(&devId); 
	// if (cudaSuccess != cudaStat)
	// { 
	// 	CLEANUP("cudaGetDevice failed"); 
	// 	printf("Error: cudaStat %d, %s\n", cudaStat, cudaGetErrorString(cudaStat));
	// 	return 1;
	// } 
	// cudaStat = cudaGetDeviceProperties( &prop, devId);
	// if (cudaSuccess != cudaStat)
	// { 
	// 	CLEANUP("cudaGetDeviceProperties failed"); 
	// 	printf("Error: cudaStat %d, %s\n", cudaStat, cudaGetErrorString(cudaStat));
	// 	return 1; 
	// } 
	// int cc = 100*prop.major + 10*prop.minor; 
	// if (cc < 130)
	// { 
	// 	CLEANUP("waive the test because only sm13 and above are supported\n"); 
	// 	printf("the device has compute capability %d\n", cc); 
	// 	printf("example test WAIVED"); 
	// 	return 2; 
	// } 
	// /* exercise Level 1 routines (scatter vector elements) */ 
	// status = cusparseDsctr(handle, 
	// 					nnz_vector, xVal, xInd, 
	// 					&y[n], CUSPARSE_INDEX_BASE_ZERO);
	// if (status != CUSPARSE_STATUS_SUCCESS) 
	// { 
	// 	CLEANUP("Scatter from sparse to dense vector failed"); 
	// 	return 1; 
	// } 

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

	
	// }

