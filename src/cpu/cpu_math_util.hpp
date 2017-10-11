/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include <math.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"

#if defined(_OPENMP)
#if _OPENMP < 201307
#define OMP_SIMD omp declare simd
#define OMP_FOR_SIMD omp parallel for
#else
#define OMP_SIMD omp simd
#define OMP_FOR_SIMD omp parallel for simd
#endif // 
#endif // _OPENMP

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t data_type>
using data_t = typename prec_traits<data_type>::type;

#ifdef USE_MKL
#include "mkl_cblas.h"
#include "mkl_trans.h"

typedef MKL_INT cblas_int;

namespace cpu_blas {

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
inline data_t<data_type>* cblas_gemm_alloc(CBLAS_IDENTIFIER identifier,
        const cblas_int m, const cblas_int n, const cblas_int k);

template <>
inline float* cblas_gemm_alloc<data_type::f32>(CBLAS_IDENTIFIER identifier,
        const cblas_int m, const cblas_int n, const cblas_int k)
{ return cblas_sgemm_alloc(identifier, m, n, k); }

template <data_type_t data_type>
inline void cblas_gemm_pack(const CBLAS_LAYOUT Layout,
        const CBLAS_IDENTIFIER identifier, const CBLAS_TRANSPOSE trans,
        const cblas_int m, const cblas_int n, const cblas_int k,
        const data_t<data_type> alpha, const data_t<data_type> *src,
        const cblas_int ld, data_t<data_type> *dest);


template <>
inline void cblas_gemm_pack<data_type::f32>(const CBLAS_LAYOUT Layout,
        const CBLAS_IDENTIFIER identifier, const CBLAS_TRANSPOSE trans,
        const cblas_int m, const cblas_int n, const cblas_int k,
        const float alpha, const float *src, const cblas_int ld, float *dest)
{ cblas_sgemm_pack(Layout, identifier, trans, m, n, k, alpha, src, ld, dest); }

template <data_type_t data_type>
inline void cblas_gemm_compute(const CBLAS_LAYOUT Layout,
        const cblas_int transa, const cblas_int transb,
        const cblas_int m, const cblas_int n, const cblas_int k,
        const data_t<data_type> *a, const cblas_int lda,
        const data_t<data_type> *b, const cblas_int ldb,
        const data_t<data_type> beta,
        data_t<data_type> *c, const cblas_int ldc);

template <>
inline void cblas_gemm_compute<data_type::f32>(const CBLAS_LAYOUT Layout,
        const cblas_int transa, const cblas_int transb,
        const cblas_int m, const cblas_int n, const cblas_int k,
        const float *a, const cblas_int lda,
        const float *b, const cblas_int ldb,
        const float beta,
        float *c, const cblas_int ldc) {
    cblas_sgemm_compute(Layout, transa, transb,
        m, n, k, a, lda, b, ldb, beta, c, ldc);
}

template <data_type_t data_type>
inline void cblas_gemm_free(data_t<data_type> *dst);

template <>
inline void cblas_gemm_free<data_type::f32>(float *dst)
{ cblas_sgemm_free(dst); }


template <data_type_t data_type>
inline void cblas_axpy(cblas_int N, data_t<data_type> alpha,
        const data_t<data_type> *X, cblas_int incx, data_t<data_type> *Y,
        cblas_int incy);

template <>
inline void cblas_axpy<data_type::f32>(cblas_int N, float alpha,
        const float *X, cblas_int incx, float *Y, cblas_int incy)
{ cblas_saxpy(N, alpha, X, incx, Y, incy); }

template <data_type_t data_type>
inline void cblas_scal(cblas_int N, data_t<data_type> a, data_t<data_type> *X,
        cblas_int incx);

template <>
inline void cblas_scal<data_type::f32>(cblas_int N, float a, float *X,
        cblas_int incx)
{ cblas_sscal(N, a, X, incx); }

template <data_type_t data_type>
inline void cblas_copy(cblas_int N, const data_t<data_type> *X, cblas_int incx,
        data_t<data_type> *Y, cblas_int incy);

template <>
inline void cblas_copy<data_type::f32>(cblas_int N, const float *X,
        cblas_int incx, float *Y, cblas_int incy)
{ cblas_scopy(N, X, incx, Y, incy); }

}

namespace cpu_trans {

template <data_type_t data_type>
using data_t = typename prec_traits<data_type>::type;

template <data_type_t data_type>
inline void omatcopy(char ordering, char trans, cblas_int rows, cblas_int cols,
        data_t<data_type> alpha, const data_t<data_type> *A, cblas_int lda,
        data_t<data_type> *B, cblas_int ldb);

template <>
inline void omatcopy<data_type::f32>(char ordering, char trans, cblas_int rows,
        cblas_int cols, float alpha, const float *A, cblas_int lda, float *B,
        cblas_int ldb)
{ mkl_somatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb); }

}

#endif // USE_MKL

template <data_type_t data_type>
#if defined(_OPENMP)
#pragma OMP_SIMD
#endif
inline data_t<data_type> Pow(data_t<data_type> A, data_t<data_type> B);

template <>
inline float Pow<data_type::f32>(float A, float B) { return powf(A, B); }

template <data_type_t data_type>
#if defined(_OPENMP)
#pragma OMP_SIMD
#endif
inline data_t<data_type> Sigmoid(data_t<data_type> A);

template <>
inline float Sigmoid<data_type::f32>(float A) { return 1 / (1 + expf(-A)); }

template <data_type_t data_type>
#if defined(_OPENMP)
#pragma OMP_SIMD
#endif
inline data_t<data_type> Tanh(data_t<data_type> A);

template <>
inline float Tanh<data_type::f32>(float A) { return tanhf(A); }

}
}
}
