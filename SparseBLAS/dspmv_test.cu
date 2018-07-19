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
#include <omp.h>
//#include "anonymouslib_cuda.h"
#include <cuda_profiler_api.h>
#include "spmv_kernel.h"
using namespace std;



void print_error(cusparseStatus_t status) {
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


int main(int argc, char *argv[]) {


	// omp_set_num_threads(8);
	// cout << "omp_get_max_threads = " << omp_get_max_threads() << endl;
	// cout << "omp_get_thread_limit = " << omp_get_thread_limit() << endl;
	// #pragma omp parallel// default (shared)
	// {
	// 	cout << "omp_get_num_threads = " << omp_get_num_threads() << endl;
	// 	cout << "omp_get_max_threads = " << omp_get_max_threads() << endl;
	// 	cout << "omp_get_thread_limit = " << omp_get_thread_limit() << endl;


	// }

	if (argc < 6) {
		cout << "Incorrect number of arguments!" << endl;
		cout << "Usage ./spmv [input matrix file] [number of GPU(s)] [number of test(s)] [kernel version (1-3)] [data type ('f' or 'b')]"  << endl;
		return -1;
	}

	char input_type = argv[1][0];

	char * filename = argv[2];

	int ngpu = atoi(argv[3]);
	int repeat_test = atoi(argv[4]);
	int kernel_version = atoi(argv[5]);
	char data_type = argv[6][0];
	int divide = atoi(argv[7]);
	int copy_of_workspace = atoi(argv[8]);

	int ret_code;
    MM_typecode matcode;
    FILE *f;
    int m, n, nnz;   
    int * cooRowIndex;
    int * cooColIndex;
    double * cooVal;
    int * csrRowPtr;

    if (input_type == 'f') {

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

	    //cooRowIndex = (int *) malloc(nnz * sizeof(int));
	    //cooColIndex = (int *) malloc(nnz * sizeof(int));
	    //cooVal      = (double *) malloc(nnz * sizeof(double));

	    cudaMallocHost((void **)&cooRowIndex, nnz * sizeof(int));
	    cudaMallocHost((void **)&cooColIndex, nnz * sizeof(int));
	    cudaMallocHost((void **)&cooVal, nnz * sizeof(double));
	   
	    // Read matrix from file into COO format
	    for (int i = 0; i < nnz; i++) {
	    	if (data_type == 'b') { // binary input
	    		fscanf(f, "%d %d\n", &cooRowIndex[i], &cooColIndex[i]);
	    		cooVal[i] = 0.00001;

	    	} else if (data_type == 'f'){ // float input
	        	fscanf(f, "%d %d %lg\n", &cooRowIndex[i], &cooColIndex[i], &cooVal[i]);
	        }
	        cooRowIndex[i]--;  
	        cooColIndex[i]--;

	        if (cooRowIndex[i] < 0 || cooColIndex[i] < 0) { // report error
	       		cout << "i = " << i << " [" <<cooRowIndex[i] << ", " << cooColIndex[i] << "] = " << cooVal[i] << endl;
	       	}
		}
	} else if(input_type == 'g') { // generate data
		//int n = 10000;
		n = atoi(filename);

		m = n;
		int nb = m / 8;
		double r;
		double r1 = 0.9;
		double r2 = 0.1;

		int p = 0;

		for (int i = 0; i < m; i += nb) {
			if (i == 0) {
				r = r1;
			} else {
				r = r2;
			}
			for (int ii = i; ii < i + nb; ii++) {
				for (int j = 0; j < n * r; j++) {
					p++;
				}
			}
		}


		nnz = p;

		cout << "m: " << m << " n: " << n << " nnz: " << nnz << endl;

		cudaMallocHost((void **)&cooRowIndex, nnz * sizeof(int));
	    cudaMallocHost((void **)&cooColIndex, nnz * sizeof(int));
	    cudaMallocHost((void **)&cooVal, nnz * sizeof(double));

	    p = 0;
		

		cout << "Start generating data ..." << endl;
		for (int i = 0; i < m; i += nb) {
			cout << ((double)p / nnz) * 100 << "%" << endl;
			cout << p << endl;
			if (i == 0) {
				r = r1;
			} else {
				r = r2;
			}
			cout << "Matrix:" << endl;
			for (int ii = i; ii < i + nb; ii++) {
				for (int j = 0; j < n * r; j++) {
					//if (p > nnz) { cout << "error" << endl; break;}
					//else {

					cooRowIndex[p] = ii;
					cooColIndex[p] = j;
					cooVal[p] = 1;//((double) rand() / (RAND_MAX));
					p++;
					cout << 1 << " ";
					//}

				}
				cout << endl;
			}


		}

		cout << "m: " << m << " n: " << n << " nnz: " << p << endl;


		cout << "Done generating data." << endl;


	}



    




	// Convert COO to CSR
    //csrRowPtr = (int *) malloc((m+1) * sizeof(int));
    cudaMallocHost((void **)&csrRowPtr, (m+1) * sizeof(int));

    //cout << "m: " << m << " n: " << n << " nnz: " << nnz << endl;
    long long matrix_data_space = nnz * sizeof(double) + nnz * sizeof(int) + (m+1) * sizeof(int);
    //cout << matrix_data_space << endl;


    cout << "Matrix space size: " << (double)matrix_data_space / 1e9 << " GB." << endl;

    int * counter = new int[m];
    for (int i = 0; i < m; i++) {
    	counter[i] = 0;
    }
	for (int i = 0; i < nnz; i++) {
		counter[cooRowIndex[i]]++;
	}
	//cout << "nnz: " << nnz << endl;
	//cout << "counter: ";
	int t = 0;
	for (int i = 0; i < m; i++) {
		//cout << counter[i] << ", ";
		t += counter[i];
	}
	//cout << t << endl;
	//cout << endl;


	csrRowPtr[0] = 0;
	for (int i = 1; i <= m; i++) {
		csrRowPtr[i] = csrRowPtr[i - 1] + counter[i - 1];
	}

	// cout << "csrRowPtr: ";
	// for (int i = 0; i <= m; i++) {
	// 	cout << csrRowPtr[i] << ", ";
	// }
	// cout << endl;

	double * x;
	double * y1;
	double * y2;
	double * y3;

	//x = (double *)malloc(n * sizeof(double)); 
	//y1 = (double *)malloc(m * sizeof(double)); 
	y2 = (double *)malloc(m * sizeof(double)); 
	//y3 = (double *)malloc(m * sizeof(double)); 

// 	cudaMallocHost((void **)&x, n * sizeof(double));
// 	cudaMallocHost((void **)&y1, m * sizeof(double));
// 	//cudaMallocHost((void **)&y2, m * sizeof(double));
// 	cudaMallocHost((void **)&y3, m * sizeof(double));

// 	for (int i = 0; i < n; i++)
// 	{
// 		x[i] = 1.0;//((double) rand() / (RAND_MAX)); 
// 	}


// 	for (int i = 0; i < m; i++)
// 	{
// 		y1[i] = 0.0;
// 		y2[i] = 0.0;
// 		y3[i] = 0.0;
// 	}



// 	int deviceCount;
// 	cudaGetDeviceCount(&deviceCount);
// 	int device;
// 	for (device = 0; device < deviceCount; ++device) 
// 	{
// 	    cudaDeviceProp deviceProp;
// 	    cudaGetDeviceProperties(&deviceProp, device);
// 	    printf("Device %d has compute capability %d.%d.\n",
// 	           device, deviceProp.major, deviceProp.minor);
// 	}

// 	cout << "Using " << ngpu << " GPU(s)." << endl; 

// 	double ALPHA = 1.0;
// 	double BETA = 0.0;

// 	double time_parse = 0.0;
// 	double time_comm = 0.0;
// 	double time_comp = 0.0;
// 	double time_post = 0.0;

// 	double avg_time_parse1 = 0.0;
// 	double avg_time_comm1 = 0.0;
// 	double avg_time_comp1 = 0.0;
// 	double avg_time_post1 = 0.0;

// 	double avg_time_parse2 =  0.0;
// 	double avg_time_comm2 = 0.0;
// 	double avg_time_comp2 = 0.0;
// 	double avg_time_post2 = 0.0;

// 	double avg_time_parse3 =  0.0;
// 	double avg_time_comm3= 0.0;
// 	double avg_time_comp3 = 0.0;
// 	double avg_time_post3 = 0.0;

// 	int warm_up_iter = 1;

// 	//cudaProfilerStart();

// 	for (int i = 0; i < repeat_test + warm_up_iter; i++) {
// 		if (i == 0) {
// 			cout << "Warming up GPU(s)..." << endl;
// 		}
// 		if (i == warm_up_iter) {
// 			cout << "Starting tests..." << endl;
// 		}
// 		for (int i = 0; i < m; i++)
// 		{
// 			y1[i] = 0.0;
// 			y2[i] = 0.0;
// 			y3[i] = 0.0;
// 		}

		
// 		time_parse = 0.0;
// 		time_comm = 0.0;
// 		time_comp = 0.0;
// 		time_post = 0.0;

// 		cout << "=============Baseline[start]============" <<endl;

// 		// spMV_mgpu_baseline(m, n, nnz, &ALPHA,
// 		// 			 cooVal, csrRowPtr, cooColIndex, 
// 		// 			 x, &BETA,
// 		// 			 y1,
// 		// 			 ngpu,
// 		// 			 &time_parse,
// 		// 			 &time_comm,
// 		// 			 &time_comp,
// 		// 			 &time_post);
// 		cout << "=============Baseline[done]============" <<endl;

		
	
		
// 		if (i >= warm_up_iter) {
// 			avg_time_parse1 += time_parse;
// 			avg_time_comm1  += time_comm;
// 			avg_time_comp1  += time_comp;
// 			avg_time_post1  += time_post;
// 		}
		
// 		time_parse = 0.0;
// 		time_comm = 0.0;
// 		time_comp = 0.0;
// 		time_post = 0.0;
		
// 		cout << "=============Version 1[start]============" <<endl;

// 		// spMV_mgpu_v1(m, n, nnz, &ALPHA,
// 		// 			 cooVal, csrRowPtr, cooColIndex, 
// 		// 			 x, &BETA,
// 		// 			 y2,
// 		// 			 ngpu,
// 		// 			 &time_parse,
// 		// 			 &time_comm,
// 		// 			 &time_comp,
// 		// 			 &time_post,
// 		// 			 kernel_version);

// 		cout << "=============Version 1[done]============" <<endl;
		
// 		if (i >= warm_up_iter) {
// 			avg_time_parse2 += time_parse;
// 			avg_time_comm2  += time_comm;
// 			avg_time_comp2  += time_comp;
// 			avg_time_post2  += time_post;
// 		}

// 		time_parse = 0.0;
// 		time_comm = 0.0;
// 		time_comp = 0.0;
// 		time_post = 0.0;

// 		cout << "=============Version 2[start]============" <<endl;


// 		//cudaProfilerStart();


// 		// spMV_mgpu_v2(m, n, nnz, &ALPHA,
// 		// 			 cooVal, csrRowPtr, cooColIndex, 
// 		// 			 x, &BETA,
// 		// 			 y3,
// 		// 			 ngpu,
// 		// 			 kernel_version,
// 		// 			 ceil(nnz/divide),
// 		// 			 copy_of_workspace,
// 		// 			 &time_parse,
// 		// 			 &time_comp,
// 		// 			 &time_post);

// 		//
// 		cout << "=============Version 2[done]============" <<endl;
		
// 		if (i >= warm_up_iter) {
// 			avg_time_parse3 += time_parse;
// 			avg_time_comm3  += time_comm;
// 			avg_time_comp3  += time_comp;
// 			avg_time_post3  += time_post;
// 		}

	
// 	}

// 	//cudaProfilerStop();

	


// 	//cout << "y = [";
// 	bool correct = true;
// 	for(int i = 0; i < m; i++) {
// 		cout << y1[i] << " - "  << y2[i] << " - "<< y3[i] << endl;
// 		if (abs(y1[i] - y3[i]) > 1e-3 ) {
// 			//cout << y1[i] << " - " << y3[i] << endl;
// 			correct = false;
// 		}
// 	}

// 	if (correct) cout << "Pass" << endl;
// 	else cout << "No pass" << endl;
	
	
// 	avg_time_parse1/=repeat_test;
// 	avg_time_comm1/=repeat_test;
// 	avg_time_comp1/=repeat_test;
// 	avg_time_post1/=repeat_test;

// 	avg_time_parse2/=repeat_test;
// 	avg_time_comm2/=repeat_test;
// 	avg_time_comp2/=repeat_test;
// 	avg_time_post2/=repeat_test;

// 	avg_time_parse3/=repeat_test;
// 	avg_time_comm3/=repeat_test;
// 	avg_time_comp3/=repeat_test;
// 	avg_time_post3/=repeat_test;

// 	cout << "[BASELINE]" << endl;
//     cout << "avg_time_parse = " << avg_time_parse1 << endl;
// 	cout << "avg_time_comm = "  << avg_time_comm1 << endl;
// 	cout << "avg_time_comp = "  << avg_time_comp1 << endl;
// 	cout << "avg_time_post = "  << avg_time_post1 << endl;
// 	cout << "total_time = " << avg_time_parse1+avg_time_comm1+avg_time_comp1+avg_time_post1 << endl;

// 	cout << endl;

// 	cout << "[Version 1]" << endl;
// 	cout << "avg_time_parse = " << avg_time_parse2 << endl;
// 	cout << "avg_time_comm = "  << avg_time_comm2 << endl;
// 	cout << "avg_time_comp = "  << avg_time_comp2 << endl;
// 	cout << "avg_time_post = "  << avg_time_post2 << endl;
// 	cout << "total_time = " << avg_time_parse2+avg_time_comm2+avg_time_comp2+avg_time_post2 << endl;

// 	cout << endl;

// 	cout << "[Version2]" << endl;
// 	cout << "avg_time_parse = " << avg_time_parse3 << endl;
// //	cout << "avg_time_comm = "  << avg_time_comm3 << endl;
// 	cout << "avg_time_comp = "  << avg_time_comp3 << endl;
// 	cout << "avg_time_post = "  << avg_time_post3 << endl;
// 	cout << "total_time = " << avg_time_parse3+avg_time_comp3+avg_time_post3 << endl;

}