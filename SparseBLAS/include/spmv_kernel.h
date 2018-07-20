#ifndef SPMV_KERNEL
#define SPMV_KERNEL

int csr5_kernel(int m, int n, int nnz, double * alpha,
				  double * csrVal, int * csrRowPtr, int * csrColIndex, 
				  double * x, double * beta,
				  double * y);

int spMV_mgpu_baseline(int m, int n, long long nnz, double * alpha,
				 double * csrVal, int * csrRowPtr, int * csrColIndex, 
				 double * x, double * beta,
				 double * y,
				 int ngpu);
int spMV_mgpu_v1(int m, int n, long long nnz, double * alpha,
				  double * csrVal, int * csrRowPtr, int * csrColIndex, 
				  double * x, double * beta,
				  double * y,
				  int ngpu,
				  int kernel);

int spMV_mgpu_v2(int m, int n, int nnz, double * alpha,
				  double * csrVal, int * csrRowPtr, int * csrColIndex, 
				  double * x, double * beta,
				  double * y,
				  int ngpu, 
				  int kernel,
				  int nb,
				  int copy_of_workspace);

int get_row_from_index(int n, int * a, int idx);

double get_time();


#endif /* SPMV_KERNEL */