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

int get_row_from_index(int n, int * a, int idx) {
	int l = 0;
	int r = n;
	while (l < r - 1 ) {
		int m = l + (r - l) / 2;
		if (idx < a[m]) {
			r = m;
		} else if (idx > a[m]) {
			l = m;
		} else {
			return m;
		}
	}
	// cout << "a[" << l << "] = " <<  a[l];
	// cout << " a[" << r << "] = " << a[r];
	// cout << " idx = " << idx << endl;
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
