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

#include <cmath>
#include <cblas.h>
#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"

#include "utils.hpp"

namespace mkldnn {

enum {
    RELU = 1,
    TANH = 2,
    LSTM = 3,
    GRU = 4
};

enum {
    UNIDIRECT = 1,
    BIDIRECT = 2
};

enum {
    LINEAR = 1,
    SKIP =2
};

enum {
    NOTRANS = 1,
    TRANS =2
};
struct test_lstm_desc_t {
    size_t state_size, input_size;
    size_t seq_length, num_layers;
    size_t batch_size;
    int alg_kind;
    int direction;
    int input_mode;
};

template <typename data_t>
inline void transpose(const data_t *src, data_t *dst, const int M, const int N) {
    data_t **src_a = new data_t *[M];
    data_t **dst_a = new data_t *[N];
    for (int i=0; i< M; i++) src_a[i] = (data_t *)src + N*i;
    for (int i=0; i< N; i++) dst_a[i] = (data_t *)dst + M*i;
    for (int m = 0; m<M; m++) {
        for(int n=0; n<N; n++) {
            dst_a[n][m] = src_a[m][n];
        }
    }
}

template <typename data_t>
inline void directcopy(const data_t *src, data_t *dst, const int M, const int N) {
#pragma omp parallel for
    for(int n = 0; n<N*M; n++) {
        dst[n] = src[n];
    }
}

template <typename data_t>
inline void gemm(const int transA, const int transB, const data_t *A, const data_t *B, data_t *C, const int M, const int N, const int K) {
    int m, n, k;
    for (m = 0; m< M*N; m++) {
        C[m] = static_cast<data_t>(0.);
    }
    for (k = 0; k < K; k++) {
        for (m = 0; m< M; m++) {
            for (n = 0; n < N; n++) {
                if (transA == NOTRANS && transB == NOTRANS )
                    C[m * N + n] += A[m * K + k] * B[k * N + n];
                if (transA == TRANS && transB == NOTRANS)
                    C[m * N + n] += A[k * M + m] * B[k * N + n];
                if (transA == NOTRANS && transB == TRANS)
                    C[m * N + n] += A[m*K + k] * B[n * K +k];
                if (transA == TRANS && transB == TRANS)
                    C[m * N + n] += A[k * M + m] * B[n * K +k];
            }   
        }
    }
}

template <typename data_t>
void compute_ref_lstm_fwd(const test_lstm_desc_t &ld,
        const memory::desc &x_d,
        const memory::desc &hx_d,
        const memory::desc &y_d,
        const memory::desc &weights_d,
        const memory &x,
        const memory &hx,
        const memory &cx,
        const memory &weights,
        const memory &y,
        const memory &hy,
        const memory &cy,
        const memory &ws)
{
    using namespace mkldnn::impl::utils;

    data_t *x_ptr = (data_t *)x.get_data_handle();
    data_t *hx_ptr = (data_t *)hx.get_data_handle();
    data_t *cx_ptr = (data_t *)cx.get_data_handle();
    data_t *weights_ptr = (data_t *)weights.get_data_handle();
    data_t *y_ptr = (data_t *)y.get_data_handle();
    data_t *hy_ptr = (data_t *)hy.get_data_handle();
    data_t *cy_ptr = (data_t *)cy.get_data_handle();
    data_t *ws_ptr = (data_t *)ws.get_data_handle();

    const size_t state_size = ld.state_size;
    const size_t input_size = ld.input_size;
    const size_t seq_length = ld.seq_length;
    const size_t num_layers = ld.num_layers;
    const size_t batch_size = ld.batch_size;

    const size_t w1_size = state_size * (state_size + input_size + 2) * 4;
    const size_t wx_size = state_size * (state_size + state_size + 2) * 4;
    const size_t h_size = batch_size * state_size;
    const size_t x_size = batch_size * input_size;
    const size_t h_nlayer_size = h_size * num_layers;
    const size_t gates_size = h_size * 4;
    const size_t gates_nlayer_size = gates_size * num_layers;
    const size_t gates_space_size = gates_nlayer_size * seq_length;
    const size_t hout_space_size = h_nlayer_size * seq_length;

    size_t bsize = (input_size > state_size) ? input_size : state_size;
    size_t tmp1 = bsize + state_size + 2;
    size_t tmp2 = state_size * 4;
    size_t btmp = (tmp1 > tmp2) ? tmp1 : tmp2;
    size_t temp_size = btmp * batch_size;
    data_t *fts_ = new data_t[temp_size];

    const size_t gates_space_off = 0;
    const size_t hout_space_off = gates_space_size;
    const size_t c_space_off = hout_space_off + hout_space_size;
    for (size_t t = 0; t < seq_length; t++) {
      for (size_t l = 0; l < num_layers; l++) {
        // Hin
        if (t == 0 && l == 0) {
            transpose<data_t>(x_ptr, fts_, batch_size, input_size);
            transpose<data_t>(hx_ptr, fts_ + x_size, batch_size, state_size);
            array_set(fts_ + x_size + h_size,
                1.0, 2 * batch_size);
        } else if (t == 0 && l > 0) {
            directcopy<data_t>(ws_ptr + hout_space_off + (l - 1) * h_size,
                fts_, state_size, batch_size);
            transpose<data_t>(hx_ptr + l * h_size, fts_ + x_size, batch_size, state_size);
            array_set(fts_ + x_size + h_size,
                1.0, 2 * batch_size);
        } else if (t > 0 && l == 0) {
            transpose<data_t>(x_ptr + t * x_size, fts_, batch_size, input_size);
            directcopy<data_t>(ws_ptr + hout_space_off + (t - 1) * h_nlayer_size, 
                fts_ + x_size, state_size, batch_size);
            array_set(fts_ + x_size + h_size,
                1.0, 2 * batch_size);
        } else if (t > 0 && l > 0) {
            directcopy<data_t>(ws_ptr + hout_space_off + (l - 1) * h_size + t * h_nlayer_size,
                fts_, state_size, batch_size);
            directcopy<data_t>(ws_ptr + hout_space_off + l * h_size + (t - 1) * h_nlayer_size,
                fts_ + x_size, state_size, batch_size);
            array_set(fts_ + x_size + h_size,
                1.0, 2 * batch_size);
        }
        size_t w_base_offset = 0;
        if (l > 0) {
          w_base_offset = w1_size + (l - 1) * wx_size;
        }

        size_t in_size = (l == 0) ? input_size : state_size;
        gemm<data_t>(TRANS, NOTRANS, 
            weights_ptr + w_base_offset, fts_,
            ws_ptr + gates_space_off + l * gates_size + t * gates_nlayer_size,
            4 * state_size, batch_size,
            in_size + state_size + 2);
        for(int h = 0; h < h_size; h++) {
            data_t it = ws_ptr[gates_space_off + l * gates_size + t * gates_nlayer_size + h];
            data_t ft = ws_ptr[gates_space_off + l * gates_size + t * gates_nlayer_size + h_size + h];
            data_t ot = ws_ptr[gates_space_off + l * gates_size + t * gates_nlayer_size + 2 * h_size + h];
            data_t gt = ws_ptr[gates_space_off + l * gates_size + t * gates_nlayer_size + 3 * h_size + h];
            it = 1/(1+exp(-it));
            ft = 1/(1+exp(-ft));
            ot = 1/(1+exp(-ot));
            gt = tanh(gt);
            data_t c_t_1;
            if (t == 0) { 
                c_t_1 = cx_ptr[l*h_size + h];
            } else {
                c_t_1 = ws_ptr[c_space_off + l * h_size + (t - 1) * h_nlayer_size + h];
            }
            data_t ct = c_t_1 * ft + gt * it;
            data_t ht = ot * tanh(ct);
            ws_ptr[hout_space_off + l * h_size + t * h_nlayer_size + h] = ht;
            ws_ptr[hout_space_off + l * h_size + t * h_nlayer_size + h] = ht;
            ws_ptr[c_space_off + l * h_size + t * h_nlayer_size + h] = ct;
        }

        // save output
        if (l == num_layers - 1) {
            transpose<data_t>(ws_ptr + hout_space_off +
                        (h_nlayer_size - h_size) + t * h_nlayer_size,
                        y_ptr + t * h_size,
                        state_size, batch_size);
        }

        if (t == (seq_length - 1)) {
            if(hy_ptr != nullptr)
            transpose<data_t>(ws_ptr + hout_space_off +
                          (seq_length - 1) * h_nlayer_size + l * h_size,
                          hy_ptr + l * h_size,
                          state_size, batch_size);
            if(cy_ptr != nullptr)
            transpose<data_t>(ws_ptr + c_space_off +
                          (seq_length - 1) * h_nlayer_size + l * h_size,
                          cy_ptr + l * h_size,
                          state_size, batch_size);
          }
        }
      }

}

struct lstm_fwd_test_params {
    prop_kind aprop_kind;
    const engine::kind engine_kind;
    algorithm aalgorithm;
    direction adirection;
    input_mode ainput_mode;
    memory::format rnx_format;
    test_lstm_desc_t test_ld;
};

template <typename data_t>
class lstm_forward_test : public ::testing::TestWithParam<lstm_fwd_test_params> {
protected:
    virtual void SetUp()
    {
        using namespace mkldnn::impl::utils;
        lstm_fwd_test_params p
                = ::testing::TestWithParam<lstm_fwd_test_params>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_TRUE(p.aprop_kind == prop_kind::forward_training
                || p.aprop_kind == prop_kind::forward_scoring);
        auto eng = engine(p.engine_kind, 0);
        memory::data_type data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);
        test_lstm_desc_t ld = p.test_ld;
        const size_t w1_size = ld.state_size * (ld.state_size + ld.input_size + 2) * 4;
        const size_t wx_size = ld.state_size * (ld.state_size + ld.state_size + 2) * 4;
        const size_t total_w = ld.num_layers == 1 ? w1_size : (w1_size + (ld.num_layers -1) * wx_size);
        auto l_x_desc = create_md({ static_cast<int>(ld.seq_length), static_cast<int>(ld.batch_size), static_cast<int>(ld.input_size) },
                data_type, p.rnx_format);
        auto l_hx_desc = create_md({ static_cast<int>(ld.num_layers), static_cast<int>(ld.batch_size), static_cast<int>(ld.state_size) },
                data_type, p.rnx_format);
        auto l_y_desc = create_md({ static_cast<int>(ld.seq_length), static_cast<int>(ld.batch_size), static_cast<int>(ld.state_size) },
                data_type, p.rnx_format);
        auto l_weights_desc = create_md({ static_cast<int>(total_w) }, data_type, memory::format::x);
        auto rnn_desc = rnn_forward::desc(p.aprop_kind, p.aalgorithm, p.adirection, 
            p.ainput_mode, ld.state_size, ld.num_layers, ld.seq_length, l_x_desc,
            l_hx_desc, l_y_desc, l_weights_desc);

        auto rnn_prim_desc = rnn_forward::primitive_desc(rnn_desc, eng);
        auto x_primitive_desc = memory::primitive_desc(l_x_desc, eng);
        auto hx_primitive_desc = memory::primitive_desc(l_hx_desc, eng);
        auto cx_primitive_desc = memory::primitive_desc(l_hx_desc, eng);
        auto y_primitive_desc = memory::primitive_desc(l_y_desc, eng);
        auto hy_primitive_desc = memory::primitive_desc(l_hx_desc, eng);   
        auto cy_primitive_desc = memory::primitive_desc(l_hx_desc, eng);
        auto weights_primitive_desc = memory::primitive_desc(l_weights_desc, eng);
        auto workspace_primitive_desc = rnn_prim_desc.workspace_primitive_desc();
        auto x_size = x_primitive_desc.get_size();
        auto hx_size = hx_primitive_desc.get_size();
        auto cx_size = cx_primitive_desc.get_size();
        auto y_size = y_primitive_desc.get_size();
        auto hy_size = hy_primitive_desc.get_size();
        auto cy_size = cy_primitive_desc.get_size();
        auto weights_size = weights_primitive_desc.get_size();
        auto workspace_size = workspace_primitive_desc.get_size();

        // TODO: free
        data_t *x_data = new data_t[x_size/sizeof(data_t)];
        data_t *hx_data = new data_t[hx_size/sizeof(data_t)];
        data_t *cx_data = new data_t[cx_size/sizeof(data_t)];
        data_t *y_data = new data_t[y_size/sizeof(data_t)];
        data_t *hy_data = new data_t[hy_size/sizeof(data_t)];
        data_t *cy_data = new data_t[cy_size/sizeof(data_t)];
        data_t *weights_data = new data_t[weights_size/sizeof(data_t)];
        data_t *workspace_data = new data_t[workspace_size/sizeof(data_t)];

        data_t *ref_y_data = new data_t[y_size/sizeof(data_t)];
        data_t *ref_hy_data = new data_t[hy_size/sizeof(data_t)];
        data_t *ref_cy_data = new data_t[cy_size/sizeof(data_t)];

        auto l_x = memory(x_primitive_desc, x_data);
        auto l_hx = memory(hx_primitive_desc, hx_data);
        auto l_cx = memory(cx_primitive_desc, cx_data);
        auto l_y = memory(y_primitive_desc, y_data);
        auto l_hy = memory(x_primitive_desc, hy_data);
        auto l_cy = memory(x_primitive_desc, cy_data);
        auto l_weights = memory(weights_primitive_desc, weights_data);
        auto l_ws = memory(workspace_primitive_desc, workspace_data);

        auto l_ref_y = memory(y_primitive_desc, ref_y_data);
        auto l_ref_hy = memory(hy_primitive_desc, ref_hy_data);
        auto l_ref_cy = memory(cy_primitive_desc, ref_cy_data); 

        // Only true for dense format
        fill_data<data_t>(l_x.get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)l_x.get_data_handle());
        fill_data<data_t>(l_hx.get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)l_hx.get_data_handle());
        fill_data<data_t>(l_cx.get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)l_cx.get_data_handle());
        fill_data<data_t>(l_weights.get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)l_weights.get_data_handle());

        // Execute
        std::vector<primitive> pipeline;
        auto s = stream(stream::kind::lazy);
        auto l = rnn_forward(rnn_prim_desc, l_x, l_hx, l_cx, l_weights, l_y, l_hy, l_cy, l_ws);
        pipeline.push_back(l);
        s.submit(pipeline).wait();
        
        compute_ref_lstm_fwd<data_t>(ld, l_x_desc, l_hx_desc, l_y_desc, l_weights_desc,
            l_x, l_hx, l_cx, l_weights, l_ref_y, l_ref_hy, l_ref_cy, l_ws);
        compare_data<data_t>(l_ref_y, l_y);
        compare_data<data_t>(l_ref_hy, l_hy);
        compare_data<data_t>(l_ref_cy, l_cy);

        delete [] x_data;
        delete [] hx_data;
        delete [] cx_data;
        delete [] y_data;
        delete [] hy_data;
        delete [] cy_data;
        delete [] weights_data;
        delete [] workspace_data;
        delete [] ref_y_data;
        delete [] ref_hy_data;
        delete [] ref_cy_data;
    }
};

using lstm_forward_test_float = lstm_forward_test<float>;
using lstm_fwd_test_params_float = lstm_fwd_test_params;

TEST_P(lstm_forward_test_float, TestsRNN)
{
}

INSTANTIATE_TEST_CASE_P(TestRNNForward_train, lstm_forward_test_float,
        ::testing::Values(
            lstm_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::rnn_lstm, direction::rnn_unidirectional,
            input_mode::rnn_linear_input, memory::format::rnx,
            { 128, 128, 10, 4, 64, LSTM, UNIDIRECT, LINEAR } }
            ));

}
