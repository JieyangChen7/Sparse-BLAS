#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include <iostream>
#include <cmath>
#include "mmio.h"
using namespace std;

int spMV_mgpu_v1(int m, int n, int nnz, double * alpha,
				 double * csrVal, int * csrRowPtr, int * csrColIndex, 
				 double * x, double * beta,
				 double * y,
				 int ngpu);
int spMV_mgpu_v2(int m, int n, int nnz, double * alpha,
				  double * csrVal, int * csrRowPtr, int * csrColIndex, 
				  double * x, double * beta,
				  double * y,
				  int ngpu);

void print_error(cusparseStatus_t status) {
	// cout << CUSPARSE_STATUS_SUCCESS << endl;
	// cout << CUSPARSE_STATUS_NOT_INITIALIZED << endl;
	// cout << CUSPARSE_STATUS_ALLOC_FAILED << endl;
	// cout << CUSPARSE_STATUS_INVALID_VALUE << endl;
	// cout << CUSPARSE_STATUS_ARCH_MISMATCH << endl;
	// cout << CUSPARSE_STATUS_INTERNAL_ERROR << endl;
	// cout << CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED << endl;

	if (status == CUSPARSE_STATUS_NOT_INITIALIZED)
		cout << "CUSPARSE_STATUS_NOT_INITIALIZED" << endl;
		
	else if (status == CUSPARSE_STATUS_ALLOC_FAILED)
		cout << "CUSPARSE_STATUS_ALLOC_FAILED" << endl;
		
	else if (status == CUSPARSE_STATUS_INVALID_VALUE)
		cout << "CUSPARSE_STATUS_INVALID_VALUE" << endl;
	else if (status == CUSPARSE_STATUS_ARCH_MISMATCH)
		cout << "CUSPARSE_STATUS_ARCH_MISMATCH" << endl;
	else if (status == CUSPARSE_STATUS_INTERNAL_ERROR)
		cout << "CUSPARSE_STATUS_INTERNAL_ERROR" << endl;
	else if (status == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
		cout << "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED" << endl;

}

double get_time()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	double ms = (double)tp.tv_sec * 1000 + (double)tp.tv_usec / 1000; //get current timestamp in milliseconds
	return ms / 1000;
}

int main(int argc, char *argv[]) {
	int ngpu = atoi(argv[2]);
	char * filename = argv[1];
	int ret_code;
    MM_typecode matcode;
    FILE *f;
    int m, n, nnz;   
    int * cooRowIndex;
    int * cooColIndex;
    double * cooVal;
    int * csrRowPtr;

    cout << "loading input matrix from " << filename << endl;

    if ((f = fopen(filename, "r")) == NULL) {
        exit(1);
    }


    if (mm_read_banner(f, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }


    if ((ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnz)) !=0) {
        exit(1);
    }


    cout << "m: " << m << " n: " << n << " nnz: " << nnz << endl;

    cooRowIndex = (int *) malloc(nnz * sizeof(int));
    cooColIndex = (int *) malloc(nnz * sizeof(int));
    cooVal      = (double *) malloc(nnz * sizeof(double));


    for (int i = 0; i < nnz; i++) {
        fscanf(f, "%d %d %lg\n", &cooRowIndex[i], &cooColIndex[i], &cooVal[i]);
        cooRowIndex[i]--;  
        cooColIndex[i]--;
    }


 //    cout << "cooVal: ";
	// for (int i = 0; i < nnz; i++) {
	// 	cout << cooVal[i] << ", ";
	// }
	// cout << endl;

	// cout << "cooRowIndex: ";
	// for (int i = 0; i < nnz; i++) {
	// 	cout << cooRowIndex[i] << ", ";
	// }
	// cout << endl;

	// cout << "cooColIndex: ";
	// for (int i = 0; i < nnz; i++) {
	// 	cout << cooColIndex[i] << ", ";
	// }
	// cout << endl;


    csrRowPtr = (int *) malloc((n+1) * sizeof(int));
    int * counter = new int[m];
	for (int i = 0; i < nnz; i++) {
		counter[cooRowIndex[i]]++;
	}

	// cout << "counter: ";
	// for (int i = 0; i < m; i++) {
	// 	cout << counter[i] << ", ";
	// }
	// cout << endl;


	csrRowPtr[0] = 0;
	for (int i = 1; i <= m; i++) {
		csrRowPtr[i] = csrRowPtr[i - 1] + counter[i - 1];
	}

	// cout << "csrRowPtr: ";
	// for (int i = 0; i <= m; i++) {
	// 	cout << csrRowPtr[i] << ", ";
	// }
	// cout << endl;

	double * x = (double *)malloc(n * sizeof(double)); 
	double * y = (double *)malloc(m * sizeof(double)); 

	for (int i = 0; i < n; i++)
	{
		x[i] = 1.0;//((double) rand() / (RAND_MAX)); 
		y[i] = 0.0;
	}



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

	double ONE = 1.0;
	double ZERO = 0.0;

	spMV_mgpu_v1(m, n, nnz, &ONE,
				 cooVal, csrRowPtr, cooColIndex, 
				 x, &ZERO,
				 y,
				 ngpu);

	spMV_mgpu_v1(m, n, nnz, &ONE,
				 cooVal, csrRowPtr, cooColIndex, 
				 x, &ZERO,
				 y,
				 ngpu);

	spMV_mgpu_v2(m, n, nnz, &ONE,
				 cooVal, csrRowPtr, cooColIndex, 
				 x, &ZERO,
				 y,
				 ngpu);
	spMV_mgpu_v2(m, n, nnz, &ONE,
				 cooVal, csrRowPtr, cooColIndex, 
				 x, &ZERO,
				 y,
				 ngpu);


}




int spMV_mgpu_v1(int m, int n, int nnz, double * alpha,
				 double * csrVal, int * csrRowPtr, int * csrColIndex, 
				 double * x, double * beta,
				 double * y,
				 int ngpu){

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

		cout << "dev_m[d]: " << dev_m[d] << ", dev_n[d]: " << dev_n[d] << ", dev_nnz[d]: " << dev_nnz[d] << endl;

		host_csrRowPtr[d] = new int[dev_m[d] + 1];


		cudaStat1[d] = cudaMalloc((void**)&dev_csrRowPtr[d],   (dev_m[d] + 1) * sizeof(int));
		cudaStat2[d] = cudaMalloc((void**)&dev_csrColIndex[d], dev_nnz[d] * sizeof(int)); 
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

		memcpy((void *)host_csrRowPtr[d], 
			   (void *)&csrRowPtr[start_row[d]], 
			   (dev_m[d] + 1) * sizeof(int));

		// cout << "csrRowPtr (before): ";
		// for (int i = 0; i <= dev_m[d]; i++) {
		// 	cout << host_csrRowPtr[d][i] << ", ";
		// }
		// cout << endl;

		for (int i = 0; i < dev_m[d] + 1; i++) {
			host_csrRowPtr[d][i] -= csrRowPtr[start_row[d]];
		}

		// cout << "csrRowPtr (after): ";
		// for (int i = 0; i <= dev_m[d]; i++) {
		// 	cout << host_csrRowPtr[d][i] << ", ";
		// }
		// cout << endl;


		//cout << "Start copy to GPUs...";
		cudaStat1[d] = cudaMemcpy(dev_csrRowPtr[d],   host_csrRowPtr[d],                  (size_t)((dev_m[d] + 1) * sizeof(int)), cudaMemcpyHostToDevice);
		// cout << "host_csrRowPtr[d] = ";
		// for (int i = 0; i < dev_m[d] + 1; ++i)
		// {
		// 	cout << host_csrRowPtr[d][i] << ", ";
		// }
		// cout << endl;
		cudaStat2[d] = cudaMemcpy(dev_csrColIndex[d], &csrColIndex[csrRowPtr[start_row[d]]], (size_t)(dev_nnz[d] * sizeof(int)),   cudaMemcpyHostToDevice); 
		// cout << "csrColIndex[d] = ";
		// for (int i = 0; i < dev_nnz[d]; ++i)
		// {
		// 	cout << csrColIndex[csrRowPtr[start_row[d]]+i] << ", ";
		// }
		// cout << endl;
		cudaStat3[d] = cudaMemcpy(dev_csrVal[d],      &csrVal[csrRowPtr[start_row[d]]],      (size_t)(dev_nnz[d] * sizeof(double)), cudaMemcpyHostToDevice); 
		// cout << "csrVal[d] = ";
		// for (int i = 0; i < dev_nnz[d]; ++i)
		// {
		// 	cout << csrVal[csrRowPtr[start_row[d]]+i] << ", ";
		// }
		// cout << endl;


		cudaStat4[d] = cudaMemcpy(dev_y[d], &y[start_row[d]], (size_t)(dev_m[d]*sizeof(double)), cudaMemcpyHostToDevice); 

		cudaStat5[d] = cudaMemcpy(dev_x[d], x,                (size_t)(dev_n[d]*sizeof(double)), cudaMemcpyHostToDevice); 

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
			return 1; 
		} 

		//cout << "Done" << endl;

		//cout << "initialize cuSparse ...";

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
		//cout << "Done" << endl;

	}
	for (int d = 0; d < ngpu; ++d) 
	{
		cudaSetDevice(d);
		cudaDeviceSynchronize();
	}





	//cout << "Start computation ... " << endl;
	 int repeat_test = 10;
	 double start = get_time();
	 for (int i = 0; i < repeat_test; i++) 
	 {
		for (int d = 0; d < ngpu; ++d) 
		{
			cudaSetDevice(d);
			//cout << "dev_m[d]: " << dev_m[d] << ", dev_n[d]: " << dev_n[d] << ", dev_nnz[d]: " << dev_nnz[d] << endl;
			status[d] = cusparseDcsrmv(handle[d],CUSPARSE_OPERATION_NON_TRANSPOSE, 
									   dev_m[d], dev_n[d], dev_nnz[d], 
									   alpha, descr[d], dev_csrVal[d], 
									   dev_csrRowPtr[d], dev_csrColIndex[d], 
									   dev_x[d], beta, dev_y[d]); 	 
		 	print_error(status[d]);

			
			cudaDeviceSynchronize();
		 	cudaStat1[d] = cudaMemcpy( &y[start_row[d]], dev_y[d], (size_t)(dev_m[d]*sizeof(double)),  cudaMemcpyDeviceToHost);
		 	//cudaStat1[d] = cudaMemcpy( &y[0], dev_y[0], 1*sizeof(double),  cudaMemcpyDeviceToHost);
			if ((cudaStat1[d] != cudaSuccess)) 
			{ 
				printf("Memcpy from Host to Device failed"); 
				return 1; 
			} 
		}
		for (int d = 0; d < ngpu; ++d) 
		{
			cudaSetDevice(d);
			cudaDeviceSynchronize();
		}


	}



	double end = get_time();
	double time = end - start;

	printf("spMV_mgpu_v1 time = %f s\n", time);
	
	long long flop = nnz * 2;
	flop *= repeat_test;
	double gflop = (double)flop/1e9;
	printf("gflop = %f\n", gflop);
	double gflops = gflop / time;
	printf("GFLOPS = %f\n", gflops);
	// cout << "y = [";
	// for(int i = 0; i < m; i++) {
	// 	cout << y[i] << ", ";
	// }
	// cout << "]" << endl;

}

int spMV_mgpu_v2(int m, int n, int nnz, double * alpha,
				  double * csrVal, int * csrRowPtr, int * csrColIndex, 
				  double * x, double * beta,
				  double * y,
				  int ngpu){

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

		cudaStream_t       * stream = new cudaStream_t [ngpu];
		cusparseStatus_t   * status = new cusparseStatus_t[ngpu];
		cusparseHandle_t   * handle = new cusparseHandle_t[ngpu];
		cusparseMatDescr_t * descr  = new cusparseMatDescr_t[ngpu];

		// Calculate the start and end index
		for (int i = 0; i < ngpu; i++) {
			start_idx[i]   = floor((i)     * nnz / ngpu);
			end_idx[i]     = floor((i + 1) * nnz / ngpu) - 1;
			dev_nnz[i] = end_idx[i] - start_idx[i] + 1;
		}

		// Calculate the start and end row
		curr_row = 0;
		for (int i = 0; i < ngpu; i++) {
			while (csrRowPtr[curr_row] <= start_idx[i]) {
				curr_row++;
			}

			start_row[i] = curr_row - 1; 

			// Mark imcomplete rows
			// True: imcomplete
			if (start_idx[i] > csrRowPtr[start_row[i]]) {
				start_flag[i] = true;
			} else {
				start_flag[i] = false;
			}
		}

		curr_row = 0;
		for (int i = 0; i < ngpu; i++) {
			while (csrRowPtr[curr_row] <= end_idx[i]) {
				curr_row++;
				cout << "->" << csrRowPtr[curr_row] << endl;
			}

			end_row[i] = curr_row - 1;

			// Mark imcomplete rows
			// True: imcomplete
			if (end_idx[i] < csrRowPtr[end_row[i] + 1] - 1)  {
				end_flag[i] = true;
			} else {
				end_flag[i] = false;
			}
		}

		// Cacluclate dimensions
		for (int i = 0; i < ngpu; i++) {
			dev_m[i] = end_row[i] - start_row[i] + 1;
			dev_n[i] = n;
		}

		for (int i = 0; i < ngpu; i++) {
			host_y[i] = new double[dev_m[i]];
		}

		for (int d = 0; d < ngpu; d++) {
			cout << "GPU" << d << ":" << endl;
			cout << " start_idx: " << start_idx[d] << ", ";
			cout << " end_idx: " << end_idx[d] << ", ";
			cout << " start_row: " << start_row[d] << ", ";
			cout << " end_row: " << end_row[d] << ", ";
			cout << " start_flag: " << start_flag[d] << ", ";
			cout << " end_flag: " << end_flag[d] << ", ";
			cout << endl;
			cout << " dev_m: " << dev_m[d] << ", ";
			cout << " dev_n: " << dev_n[d] << ", ";
			cout << " dev_nnz: " << dev_nnz[d] << ", ";
			cout << endl;

		}

		for (int i = 0; i < ngpu; i++) {
			host_csrRowPtr[i] = new int [dev_m[i] + 1];
			host_csrRowPtr[i][0] = 0;
			host_csrRowPtr[i][dev_m[i]] = dev_nnz[i];

			for (int j = 1; j < dev_m[i]; j++) {
				host_csrRowPtr[i][j] = csrRowPtr[start_row[i] + j];
			}

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
			cudaMemcpy(dev_csrRowPtr[d],   host_csrRowPtr[d],          (size_t)((dev_m[d] + 1) * sizeof(int)), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_csrColIndex[d], &csrColIndex[start_idx[d]], (size_t)(dev_nnz[d] * sizeof(int)),     cudaMemcpyHostToDevice); 
			cudaMemcpy(dev_csrVal[d],      &csrVal[start_idx[d]],      (size_t)(dev_nnz[d] * sizeof(double)),  cudaMemcpyHostToDevice); 

			cudaMemcpy(dev_y[d], &y[start_row[d]], (size_t)(dev_m[d]*sizeof(double)),  cudaMemcpyHostToDevice); 
			cudaMemcpy(dev_x[d], x,                (size_t)(dev_n[d]*sizeof(double)),  cudaMemcpyHostToDevice); 
		}

		for (int d = 0; d < ngpu; ++d) 
		{
			cudaSetDevice(d);
			cudaDeviceSynchronize();
		}

		int repeat_test = 10;
		double start = get_time();
		for (int i = 0; i < repeat_test; i++) 
		{
			for (int d = 0; d < ngpu; ++d) 
			{
				cudaSetDevice(d);				
				status[d] = cusparseDcsrmv(handle[d],CUSPARSE_OPERATION_NON_TRANSPOSE, 
											dev_m[d], dev_n[d], dev_nnz[d], 
											alpha, descr[d], dev_csrVal[d], 
											dev_csrRowPtr[d], dev_csrColIndex[d], 
											dev_x[d],  beta, dev_y[d]); 

				//print_error(status[d]);
				cudaMemcpy(host_y[d], dev_y[d], (size_t)(dev_m[d]*sizeof(double)),  cudaMemcpyDeviceToHost); 
			}
			for (int d = 0; d < ngpu; ++d) 
			{
				cudaSetDevice(d);
				cudaDeviceSynchronize();
			}
		}
		double end = get_time();

		cout << start << "  " << end << endl;

		double time = end - start;

		printf("spMV_mgpu_v2 time = %f s\n", time);

		long long flop = nnz * 2;
		flop *= repeat_test;
		double gflop = (double)flop/1e9;
		printf("gflop = %f\n", gflop);
		double gflops = gflop / time;
		printf("GFLOPS = %f\n", gflops);

		// for (int i = 0; i < ngpu; i++) {
		// 	cout << "host_y[i] = [";
		// 	for (int j = 0; j < dev_m[i]; j++) {
		// 		cout << host_y[i][j] << ", ";
		// 	}
		// 	cout << "]" << endl;
		// }
		


	}

