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
using namespace std;

int spMV_mgpu_v1(int m, int n, int nnz, double * alpha,
				 double * csrVal, int * csrRowPtr, int * csrColIndex, 
				 double * x, double * beta,
				 double * y,
				 int ngpu,
				 double * time_parse,
				 double * time_comm,
				 double * time_comp,
				 double * time_post);
int spMV_mgpu_v2(int m, int n, int nnz, double * alpha,
				  double * csrVal, int * csrRowPtr, int * csrColIndex, 
				  double * x, double * beta,
				  double * y,
				  int ngpu,
				  double * time_parse,
				  double * time_comm,
				  double * time_comp,
				  double * time_post);

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

int get_row_from_index(int n, int * a, int idx) {
	int l = 0;
	int r = n - 1;
	while (l < r - 1 ) {
		int m = (l + r) / 2;
		if (idx < a[m]) {
			r = m;
		} else if (idx > a[m]) {
			l = m;
		} else {
			return m;
		}
	}
	if (idx == a[l]) return l;
	if (idx == a[r]) return r;
	return l;

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
	int repeat_test = atoi(argv[3]);
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

    float progress = 0.0;
    int barWidth = 70;
    for (int i = 0; i < nnz; i++) {
    	if (argc == 5) {
    		fscanf(f, "%d %d\n", &cooRowIndex[i], &cooColIndex[i]);
    		cooVal[i] = 1;
    	} else {
        	fscanf(f, "%d %d %lg\n", &cooRowIndex[i], &cooColIndex[i], &cooVal[i]);
        }
        cooRowIndex[i]--;  
        cooColIndex[i]--;
       
    	// std::cout << "[";
    	// int pos = barWidth * progress;
    	// for (int b = 0; b < barWidth; ++b) {
     //    	if (b < pos) std::cout << "=";
     //    	else if (b == pos) std::cout << ">";
     //    	else std::cout << " ";
    	// }
    	// std::cout << "] " << int(progress * 100.0) << " %\r";
    	// std::cout.flush();

    	// progress = (float)i / nnz; 
	}
	std::cout << std::endl;
       // cout << cooRowIndex[i] << "---" << cooColIndex[i] << " : " << cooVal[i] << endl;

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


    csrRowPtr = (int *) malloc((m+1) * sizeof(int));
    int * counter = new int[m];
    for (int i = 0; i < m; i++) {
    	counter[i] = 0;
    }
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
	double * y1 = (double *)malloc(m * sizeof(double)); 
	double * y2 = (double *)malloc(m * sizeof(double)); 

	for (int i = 0; i < n; i++)
	{
		x[i] = 1.0;//((double) rand() / (RAND_MAX)); 
	}


	for (int i = 0; i < m; i++)
	{
		y1[i] = 0.0;
		y2[i] = 0.0;
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

	double time_parse = 0.0;
	double time_comm = 0.0;
	double time_comp = 0.0;
	double time_post = 0.0;

	double min_time_parse1 = DBL_MAX;
	double min_time_comm1 = DBL_MAX;
	double min_time_comp1 = DBL_MAX;
	double min_time_post1 = DBL_MAX;

	double min_time_parse2 = DBL_MAX;
	double min_time_comm2 = DBL_MAX;
	double min_time_comp2 = DBL_MAX;
	double min_time_post2 = DBL_MAX;


	double avg_time_parse1 = 0.0;
	double avg_time_comm1 = 0.0;
	double avg_time_comp1 = 0.0;
	double avg_time_post1 = 0.0;

	double avg_time_parse2 =  0.0;
	double avg_time_comm2 = 0.0;
	double avg_time_comp2 = 0.0;
	double avg_time_post2 = 0.0;

	int warm_up_iter = 10;
	// double start = get_time();
	//int repeat_test = 100;
	for (int i = 0; i < repeat_test + warm_up_iter; i++) {
		
		spMV_mgpu_v1(m, n, nnz, &ONE,
					 cooVal, csrRowPtr, cooColIndex, 
					 x, &ZERO,
					 y1,
					 ngpu,
					 &time_parse,
					 &time_comm,
					 &time_comp,
					 &time_post);
		//cout << "time_comm = " << time_comm << endl;

		if (i >= warm_up_iter) {
 
			if (time_parse < min_time_parse1) min_time_parse1 = time_parse;
			if (time_comm < min_time_comm1) min_time_comm1 = time_comm;
			if (time_comp < min_time_comp1) min_time_comp1 = time_comp;
			if (time_post < min_time_post1) min_time_post1 = time_post;

			avg_time_parse1 += time_parse;
			avg_time_comm1  += time_comm;
			avg_time_comp1  += time_comp;
			avg_time_post1  += time_post;
		}
	
		spMV_mgpu_v2(m, n, nnz, &ONE,
					 cooVal, csrRowPtr, cooColIndex, 
					 x, &ZERO,
					 y2,
					 ngpu,
					 &time_parse,
					 &time_comm,
					 &time_comp,
					 &time_post);
	
		if (i >= warm_up_iter) {
			if (time_parse < min_time_parse2) min_time_parse2 = time_parse;
			if (time_comm < min_time_comm2) min_time_comm2 = time_comm;
			if (time_comp < min_time_comp2) min_time_comp2 = time_comp;
			if (time_post < min_time_post2) min_time_post2 = time_post;

			avg_time_parse2 += time_parse;
			avg_time_comm2  += time_comm;
			avg_time_comp2  += time_comp;
			avg_time_post2  += time_post;
		}


	
	}


	//cout << "y = [";
	bool correct = true;
	for(int i = 0; i < m; i++) {
		//cout << y[i] << ", ";
		//cout << y1[i] << " - " << y2[i] << endl;
		if (abs(y1[i] - y2[i]) > 1e-5 ) {
			correct = false;
		}
	}

	printf("Check result: %s\n", correct ? "True" : "False");
	cout << "min_time_parse1 = " << min_time_parse1 << endl;
	cout << "min_time_comm1 = " << min_time_comm1 << endl;
	cout << "min_time_comp1 = " << min_time_comp1 << endl;
	cout << "min_time_post1 = " << min_time_post1 << endl;
	cout << "total_time = " << min_time_parse1+min_time_comm1+min_time_comp1+min_time_post1 << endl;

	cout << endl;

	cout << "min_time_parse2 = " << min_time_parse2 << endl;
	cout << "min_time_comm2 = " << min_time_comm2 << endl;
	cout << "min_time_comp2 = " << min_time_comp2 << endl;
	cout << "min_time_post2 = " << min_time_post2 << endl;
	cout << "total_time = " << min_time_parse2+min_time_comm2+min_time_comp2+min_time_post2 << endl;

	cout << endl;


	avg_time_parse1/=repeat_test;
	avg_time_comm1/=repeat_test;
	avg_time_comp1/=repeat_test;
	avg_time_post1/=repeat_test;

	avg_time_parse2/=repeat_test;
	avg_time_comm2/=repeat_test;
	avg_time_comp2/=repeat_test;
	avg_time_post2/=repeat_test;

    cout << "avg_time_parse1 = " << avg_time_parse1 << endl;
	cout << "avg_time_comm1 = "  << avg_time_comm1 << endl;
	cout << "avg_time_comp1 = "  << avg_time_comp1 << endl;
	cout << "avg_time_post1 = "  << avg_time_post1 << endl;
	cout << "total_time = " << avg_time_parse1+avg_time_comm1+avg_time_comp1+avg_time_post1 << endl;


	cout << endl;

	cout << "avg_time_parse2 = " << avg_time_parse2 << endl;
	cout << "avg_time_comm2 = "  << avg_time_comm2 << endl;
	cout << "avg_time_comp2 = "  << avg_time_comp2 << endl;
	cout << "avg_time_post2 = "  << avg_time_post2 << endl;
	cout << "total_time = " << avg_time_parse2+avg_time_comm2+avg_time_comp2+avg_time_post2 << endl;

	// double end = get_time();
	// double time = end - start;
	// printf("spMV_mgpu time = %f s\n", time);

}




int spMV_mgpu_v1(int m, int n, int nnz, double * alpha,
				 double * csrVal, int * csrRowPtr, int * csrColIndex, 
				 double * x, double * beta,
				 double * y,
				 int ngpu,
				 double * time_parse,
				 double * time_comm,
				 double * time_comp,
				 double * time_post){

	double curr_time = 0.0;

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

		//cout << "start_row: " << start_row[d] << ", " << "end_row: "<< end_row[d] << endl;

		dev_m[d]   = end_row[d] - start_row[d] + 1;
		dev_n[d]   = n;
		dev_nnz[d] = csrRowPtr[end_row[d] + 1] - csrRowPtr[start_row[d]];

		cout << "dev_m[d]: " << dev_m[d] << ", dev_n[d]: " << dev_n[d] << ", dev_nnz[d]: " << dev_nnz[d] << endl;

		host_csrRowPtr[d] = new int[dev_m[d] + 1];

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

	}

	*time_comm = get_time() - curr_time;
	curr_time = get_time();


	//cout << "Start computation ... " << endl;
	 int repeat_test = 1;
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

	*time_post = get_time() - curr_time;


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

	// for (int d = 0; d < ngpu; d++) {
	// 		cudaSetDevice(d);
	// 		cudafree(dev_csrVal[d]);
	// 		cudafree(dev_csrRowPtr[d]);
	// 		cudafree(dev_csrColIndex[d]);
	// 		cudafree(dev_x[d]);
	// 		cudafree(dev_y[d]);
	// 		delete [] host_csrRowPtr[d]
	// 	}
	// 	delete[] dev_csrVal;
	// 	delete[] dev_csrRowPtr;
	// 	delete[] dev_csrColIndex;
	// 	delete[] dev_x;
	// 	delete[] dev_y;
	// 	delete[] host_csrRowPtr;
	// 	delete[] start_row;
	// 	delete[] end_row;
		
	

}

int spMV_mgpu_v2(int m, int n, int nnz, double * alpha,
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
		tmp = get_time();

		curr_time = get_time();


		tmp = get_time();


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


		tmp =  get_time() - tmp;
		cout << "t1 = " << tmp << endl;

		tmp = get_time();

		// Calculate the start and end index
		for (int i = 0; i < ngpu; i++) {
			start_idx[i]   = floor((i)     * nnz / ngpu);
			end_idx[i]     = floor((i + 1) * nnz / ngpu) - 1;
			dev_nnz[i] = end_idx[i] - start_idx[i] + 1;
		}

		tmp = get_time() - tmp;
		cout << "t2 = " << tmp << endl;

		tmp = get_time();

		// Calculate the start and end row
		curr_row = 0;
		for (int i = 0; i < ngpu; i++) {
			// while (csrRowPtr[curr_row] <= start_idx[i]) {
			// 	curr_row++;
			// }

			// start_row[i] = curr_row - 1; 
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

		tmp = get_time() - tmp;
		cout << "t3 = " << tmp << endl;

		tmp = get_time();

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

		tmp = get_time() - tmp;
		cout << "t4 = " << tmp << endl;

		tmp = get_time();

		// Cacluclate dimensions
		for (int i = 0; i < ngpu; i++) {
			dev_m[i] = end_row[i] - start_row[i] + 1;
			dev_n[i] = n;
		}

		for (int i = 0; i < ngpu; i++) {
			host_y[i] = new double[dev_m[i]];
		}

		tmp = get_time() - tmp;
		cout << "t5 = " << tmp << endl;

		tmp = get_time();

		 for (int d = 0; d < ngpu; d++) {
		 	cout << "GPU " << d << ":" << endl;
		// 	cout << " start_idx: " << start_idx[d] << ", ";
		// 	cout << " end_idx: " << end_idx[d] << ", ";
		// 	cout << " start_row: " << start_row[d] << ", ";
		// 	cout << " end_row: " << end_row[d] << ", ";
		// 	cout << " start_flag: " << start_flag[d] << ", ";
		// 	cout << " end_flag: " << end_flag[d] << ", ";
		// 	cout << endl;
		 	cout << " dev_m: " << dev_m[d] << ", ";
		 	cout << " dev_n: " << dev_n[d] << ", ";
		 	cout << " dev_nnz: " << dev_nnz[d] << ", ";
		 	cout << endl;

		 }

		 tmp = get_time() - tmp;
		cout << "t6 = " << tmp << endl;

		tmp = get_time();

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

		tmp = get_time() - tmp;
		cout << "t7 = " << tmp << endl;

		tmp = get_time();

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
				cudaSetDevice(d);				
				status[d] = cusparseDcsrmv(handle[d],CUSPARSE_OPERATION_NON_TRANSPOSE, 
											dev_m[d], dev_n[d], dev_nnz[d], 
											alpha, descr[d], dev_csrVal[d], 
											dev_csrRowPtr[d], dev_csrColIndex[d], 
											dev_x[d],  beta, dev_y[d]); 

				//print_error(status[d]);
				
			}
			for (int d = 0; d < ngpu; ++d) 
			{
				cudaSetDevice(d);
				cudaDeviceSynchronize();
			}
		}

		*time_comp = get_time() - curr_time;

		curr_time = get_time();

		



		// for (int d = 0; d < ngpu; d++) {
		// 	double tmp = 0.0;
			
		// 	if (start_flag[d]) {
		// 		tmp = y[start_row[d]];
		// 	}
	
		// 	cudaMemcpy(&y[start_row[d]], dev_y[d], (size_t)(dev_m[d]*sizeof(double)),  cudaMemcpyDeviceToHost); 

		// 	if (start_flag[d]) {
		// 		y[start_row[d]] += tmp;
		// 		y[start_row[d]] -= y2[d] * (*beta);
		// 	}
		// }

		double * partial_result = new double[ngpu];
		for (int d = 0; d < ngpu; d++) {
			cudaMemcpyAsync(&partial_result[d], &dev_y[d][dev_m - 1], (size_t)(1*sizeof(double)),  cudaMemcpyDeviceToHost, stream[d]); 
		}

		for (int d = 0; d < ngpu; d++) {
			cudaMemcpyAsync(&y[start_row[d]], dev_y[d], (size_t)(dev_m[d]*sizeof(double)),  cudaMemcpyDeviceToHost, stream[d]);
		} 

		for (int d = 0; d < ngpu; ++d) 
		{
			cudaSetDevice(d);
			cudaDeviceSynchronize();
		}

		for (int d = 0; d < ngpu; d++) {
			if (start_flag[d]) {
				y[start_row[d]] += partial_result[d - 1];
				y[start_row[d]] -= y2[d] * (*beta);
			}
		}


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

