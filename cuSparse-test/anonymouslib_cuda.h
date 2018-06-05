#ifndef ANONYMOUSLIB_CUDA_H
#define ANONYMOUSLIB_CUDA_H

#include "detail/utils.h"
#include "detail/cuda/utils_cuda.h"

#include "detail/cuda/common_cuda.h"
#include "detail/cuda/format_cuda.h"
#include "detail/cuda/csr5_spmv_cuda.h"

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
class anonymouslibHandle
{
public:
    anonymouslibHandle(ANONYMOUSLIB_IT m, ANONYMOUSLIB_IT n) { _m = m; _n = n; }
    int warmup();
    int inputCSR(ANONYMOUSLIB_IT  nnz, ANONYMOUSLIB_IT *csr_row_pointer, ANONYMOUSLIB_IT *csr_column_index, ANONYMOUSLIB_VT *csr_value);
    int asCSR();
    int asCSR5();
    int setX(ANONYMOUSLIB_VT *x);
    int spmv(const ANONYMOUSLIB_VT alpha, ANONYMOUSLIB_VT *y);
    int destroy();
    void setSigma(int sigma);

private:
    int computeSigma();
    int _format;
    ANONYMOUSLIB_IT _m;
    ANONYMOUSLIB_IT _n;
    ANONYMOUSLIB_IT _nnz;

    ANONYMOUSLIB_IT *_csr_row_pointer;
    ANONYMOUSLIB_IT *_csr_column_index;
    ANONYMOUSLIB_VT *_csr_value;

    int         _csr5_sigma;
    int         _bit_y_offset;
    int         _bit_scansum_offset;
    int         _num_packet;
    ANONYMOUSLIB_IT _tail_partition_start;

    ANONYMOUSLIB_IT _p;
    ANONYMOUSLIB_UIT *_csr5_partition_pointer;
    ANONYMOUSLIB_UIT *_csr5_partition_descriptor;

    ANONYMOUSLIB_IT   _num_offsets;
    ANONYMOUSLIB_IT  *_csr5_partition_descriptor_offset_pointer;
    ANONYMOUSLIB_IT  *_csr5_partition_descriptor_offset;
    ANONYMOUSLIB_VT  *_temp_calibrator;

    ANONYMOUSLIB_VT         *_x;
    cudaTextureObject_t  _x_tex;
};

#endif // ANONYMOUSLIB_CUDA_H
