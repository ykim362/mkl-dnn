/*******************************************************************************
* Copyright 2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "c_types_map.hpp"
#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

#ifdef USE_MKL
#include "mkl_cblas.h"
#include "mkl_vml_functions.h"
#include "mkl_trans.h"
typedef MKL_INT cblas_int;
#endif

#ifdef USE_CBLAS
namespace cpu_blas {

template <data_type_t data_type>
using data_t = typename prec_trait<data_type>::type;

template <data_type_t data_type>
inline void cblas_gemm(CBLAS_LAYOUT layout,
        CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
        cblas_int M, cblas_int N, cblas_int K,
        data_t<data_type> alpha, const data_t<data_type> *A, cblas_int lda,
        const data_t<data_type> *B, cblas_int ldb,
        data_t<data_type> beta, data_t<data_type> *C, cblas_int ldc);

template <>
inline void cblas_gemm<data_type::f32>(CBLAS_LAYOUT layout,
        CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
        cblas_int M, cblas_int N, cblas_int K,
        float alpha, const float *A, cblas_int lda,
        const float *B, cblas_int ldb,
        float beta, float *C, cblas_int ldc) {
    cblas_sgemm(layout, transa, transb,
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <data_type_t data_type>
inline void cblas_axpy(cblas_int N,
        data_t<data_type> alpha, const data_t<data_type> *X, cblas_int incx,
        data_t<data_type> *Y, cblas_int incy);

template <>
inline void cblas_axpy<data_type::f32>(cblas_int N,
        float alpha, const float *X, cblas_int incx,
        float *Y, cblas_int incy) {
    cblas_saxpy(N, alpha, X, incx, Y, incy);
}

template <data_type_t data_type>
inline void cblas_scal(cblas_int N,
        data_t<data_type> a, data_t<data_type> *X, cblas_int incx);

template <>
inline void cblas_scal<data_type::f32>(cblas_int N,
        float a, float *X, cblas_int incx) {
    cblas_sscal(N, a, X, incx);
}

}
#endif //USE_CBLAS

#ifdef USE_TRANS
namespace cpu_trans {

template <data_type_t data_type>
using data_t = typename prec_trait<data_type>::type;

template <data_type_t data_type>
inline void omatcopy(char ordering, char trans,
				cblas_int rows, cblas_int cols,
				data_t<data_type> alpha,
				const data_t<data_type> *A, cblas_int lda,
                data_t<data_type> *B, cblas_int ldb);

template <>
inline void omatcopy<data_type::f32>(char ordering, char trans,
				cblas_int rows, cblas_int cols,
				float alpha,
				const float *A, cblas_int lda,
                float *B, cblas_int ldb) {
    mkl_somatcopy(ordering, trans, rows, cols, alpha,
    		A, lda, B, ldb);
}
}
#endif // USE_TRANS

#ifdef USE_VML
namespace cpu_vml {

template <data_type_t data_type>
using data_t = typename prec_trait<data_type>::type;

template <data_type_t data_type>
inline void vAdd(cblas_int N,
        const data_t<data_type> *A, const data_t<data_type> *B,
        data_t<data_type> *Y);

template <>
inline void vAdd<data_type::f32>(cblas_int N,
        const float *A, const float *B,
        float *Y) {
    vsAdd(N, A, B, Y);
}

template <data_type_t data_type>
inline void vSub(cblas_int N,
        const data_t<data_type> *A, const data_t<data_type> *B,
        data_t<data_type> *Y);

template <>
inline void vSub<data_type::f32>(cblas_int N,
        const float *A, const float *B,
        float *Y) {
    vsSub(N, A, B, Y);
}

template <data_type_t data_type>
inline void vMul(cblas_int N,
        const data_t<data_type> *A, const data_t<data_type> *B,
        data_t<data_type> *Y);

template <>
inline void vMul<data_type::f32>(cblas_int N,
        const float *A, const float *B,
        float *Y) {
    vsMul(N, A, B, Y);
}

template <data_type_t data_type>
inline void vDiv(cblas_int N,
        const data_t<data_type> *A, const data_t<data_type> *B,
        data_t<data_type> *Y);

template <>
inline void vDiv<data_type::f32>(cblas_int N,
        const float *A, const float *B,
        float *Y) {
    vsDiv(N, A, B, Y);
}

template <data_type_t data_type>
inline void vSqr(cblas_int N,
        const data_t<data_type> *A,
        data_t<data_type> *Y);

template <>
inline void vSqr<data_type::f32>(cblas_int N,
        const float *A,
        float *Y) {
    vsSqr(N, A, Y);
}

template <data_type_t data_type>
inline void vExp(cblas_int N,
        const data_t<data_type> *A,
        data_t<data_type> *Y);

template <>
inline void vExp<data_type::f32>(cblas_int N,
        const float *A,
        float *Y) {
    vsExp(N, A, Y);
}

template <data_type_t data_type>
inline void vTanh(cblas_int N,
        const data_t<data_type> *A,
        data_t<data_type> *Y);

template <>
inline void vTanh<data_type::f32>(cblas_int N,
        const float *A,
        float *Y) {
    vsTanh(N, A, Y);
}

}
#endif //USE_VML


}
}
}

