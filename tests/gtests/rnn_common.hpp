/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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

#ifndef MKLDNN_RNN_COMMON_HPP
#define MKLDNN_RNN_COMMON_HPP

#include "mkldnn.hpp"

enum {
    RELU = mkldnn_rnn_relu,
    TANH = mkldnn_rnn_tanh,
    LSTM = mkldnn_rnn_lstm,
    GRU = mkldnn_rnn_gru
};

enum { UNIDIRECT = 1, BIDIRECT = 2 };

enum { LINEAR = 1, SKIP = 2 };

enum { NOTRANS = 1, TRANS = 2 };
struct test_lstm_desc_t {
    size_t state_size, input_size;
    size_t seq_length, num_layers;
    size_t batch_size;
    int alg_kind;
    int direction;
    int input_mode;
    int state_outputs;
};
struct test_gru_desc_t {
    size_t state_size, input_size;
    size_t seq_length, num_layers;
    size_t batch_size;
    int alg_kind;
    int direction;
    int input_mode;
    int state_outputs;
};
struct test_rnn_desc_t {
    size_t state_size, input_size;
    size_t seq_length, num_layers;
    size_t batch_size;
    int alg_kind;
    int direction;
    int input_mode;
    int state_outputs;
};

template <typename data_t>
inline void transpose(const data_t *src, data_t *dst, const int M, const int N)
{
    data_t **src_a = new data_t *[M];
    data_t **dst_a = new data_t *[N];
    for (int i = 0; i < M; i++)
        src_a[i] = (data_t *)src + N * i;
    for (int i = 0; i < N; i++)
        dst_a[i] = (data_t *)dst + M * i;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            dst_a[n][m] = src_a[m][n];
        }
    }

    delete[] src_a;
    delete[] dst_a;
}

template <typename data_t>
inline void directcopy(const data_t *src, data_t *dst, const int M, const int N)
{
#pragma omp parallel for
    for (int n = 0; n < N * M; n++) {
        dst[n] = src[n];
    }
}

template <typename data_t>
inline void axpycopy(const data_t *src, data_t *dst, const int M, const int N)
{
#pragma omp parallel for
    for (int n = 0; n < N * M; n++) {
        dst[n] += src[n];
    }
}

template <typename data_t>
inline void gemm(const int transA, const int transB, const data_t *A,
        const data_t *B, data_t *C, const int M, const int N, const int K,
        const data_t beta)
{
    int m, n, k;
    if (beta == 0) {
        for (m = 0; m < M * N; m++) {
            C[m] = static_cast<data_t>(0.);
        }
    }
    for (k = 0; k < K; k++) {
        for (m = 0; m < M; m++) {
            for (n = 0; n < N; n++) {
                if (transA == NOTRANS && transB == NOTRANS)
                    C[m * N + n] += A[m * K + k] * B[k * N + n];
                if (transA == TRANS && transB == NOTRANS)
                    C[m * N + n] += A[k * M + m] * B[k * N + n];
                if (transA == NOTRANS && transB == TRANS)
                    C[m * N + n] += A[m * K + k] * B[n * K + k];
                if (transA == TRANS && transB == TRANS)
                    C[m * N + n] += A[k * M + m] * B[n * K + k];
            }
        }
    }
}

template<typename T, typename U>
inline void array_set(T *arr, const U& val, size_t size) {
    for (size_t i = 0; i < size; ++i) arr[i] = val;
}

namespace mkldnn {
struct lstm_test_params {
    prop_kind aprop_kind;
    engine::kind engine_kind;
    algorithm aalgorithm;
    direction adirection;
    input_mode ainput_mode;
    memory::format rnx_format;
    test_lstm_desc_t test_ld;
};
struct gru_test_params {
    prop_kind aprop_kind;
    engine::kind engine_kind;
    algorithm aalgorithm;
    direction adirection;
    input_mode ainput_mode;
    memory::format rnx_format;
    test_gru_desc_t test_ld;
};
struct rnn_test_params {
    prop_kind aprop_kind;
    engine::kind engine_kind;
    algorithm aalgorithm;
    direction adirection;
    input_mode ainput_mode;
    memory::format rnx_format;
    test_rnn_desc_t test_rd;
};
}

#endif
