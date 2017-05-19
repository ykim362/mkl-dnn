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

#include "mkldnn_test_common.hpp"
#include "rnn_common.hpp"
#include "utils.hpp"
#include "gtest/gtest.h"
#include <sys/time.h>

namespace mkldnn {

double l_total = 0.0;
double flops = 0.0;

template <typename data_t>
class lstm_test : public ::testing::TestWithParam<lstm_test_params> {
private:
    std::shared_ptr<memory> x;
    std::shared_ptr<memory> hx;
    std::shared_ptr<memory> cx;
    std::shared_ptr<memory> dx;
    std::shared_ptr<memory> dhx;
    std::shared_ptr<memory> dcx;
    std::shared_ptr<memory> y;
    std::shared_ptr<memory> hy;
    std::shared_ptr<memory> cy;
    std::shared_ptr<memory> dy;
    std::shared_ptr<memory> dhy;
    std::shared_ptr<memory> dcy;
    std::shared_ptr<memory> weights;
    std::shared_ptr<memory> dweights;
    std::shared_ptr<memory> workspace;
    std::shared_ptr<memory> ref_y;
    std::shared_ptr<memory> ref_hy;
    std::shared_ptr<memory> ref_cy;
    std::shared_ptr<memory> ref_dx;
    std::shared_ptr<memory> ref_dhx;
    std::shared_ptr<memory> ref_dcx;
    std::shared_ptr<memory> ref_dweights;
    std::shared_ptr<memory::desc> x_desc;
    std::shared_ptr<memory::desc> hx_desc;
    std::shared_ptr<memory::desc> y_desc;
    std::shared_ptr<memory::desc> weights_desc;
    std::shared_ptr<rnn_forward::primitive_desc> rnn_fwd_prim_desc;
    std::shared_ptr<rnn_backward::primitive_desc> rnn_bwd_prim_desc;
    lstm_test_params p;
    std::shared_ptr<engine> eng;
    memory::data_type data_type;
    bool with_workspace;

protected:
    virtual void SetUp()
    {
        using namespace mkldnn::impl::utils;
        p = ::testing::TestWithParam<lstm_test_params>::GetParam();
        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_TRUE(p.aalgorithm == algorithm::rnn_lstm);
        ASSERT_TRUE(p.adirection == direction::rnn_unidirectional
                || p.adirection == direction::rnn_bidirectional);
        ASSERT_TRUE(p.ainput_mode == input_mode::rnn_linear_input);
        eng.reset(new engine(p.engine_kind, 0));
        data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);
        test_lstm_desc_t ld = p.test_ld;
        with_workspace = p.aprop_kind == prop_kind::forward_training;
        size_t dir = (p.adirection == direction::rnn_unidirectional) ? 1 : 2;

        const size_t w1_size
                = ld.state_size * (ld.state_size + ld.input_size + 2) * 4;
        const size_t wx_size
                = ld.state_size * (ld.state_size + ld.state_size + 2) * 4;
        const size_t total_w = ld.num_layers == 1 ?
                dir * w1_size :
                dir * (w1_size + (ld.num_layers - 1) * wx_size);

        x_desc.reset(new memory::desc({ static_cast<int>(ld.seq_length),
                                              static_cast<int>(ld.batch_size),
                                              static_cast<int>(ld.input_size) },
                data_type, p.rnx_format));
        hx_desc.reset(
                new memory::desc({ static_cast<int>(ld.num_layers),
                                         static_cast<int>(ld.batch_size),
                                         static_cast<int>(ld.state_size) },
                        data_type, p.rnx_format));
        y_desc.reset(new memory::desc(
                { static_cast<int>(ld.seq_length),
                        static_cast<int>(ld.batch_size),
                        static_cast<int>(ld.state_size * dir) },
                data_type, p.rnx_format));
        weights_desc.reset(new memory::desc(
                { static_cast<int>(total_w) }, data_type, memory::format::x));
        x.reset(new memory({ *x_desc, *eng }));
        hx.reset(new memory({ *hx_desc, *eng }));
        cx.reset(new memory({ *hx_desc, *eng }));
        dx.reset(new memory({ *x_desc, *eng }));
        dhx.reset(new memory({ *hx_desc, *eng }));
        dcx.reset(new memory({ *hx_desc, *eng }));
        y.reset(new memory({ *y_desc, *eng }));
        hy.reset(new memory({ *hx_desc, *eng }));
        cy.reset(new memory({ *hx_desc, *eng }));
        dy.reset(new memory({ *y_desc, *eng }));
        dhy.reset(new memory({ *hx_desc, *eng }));
        dcy.reset(new memory({ *hx_desc, *eng }));
        weights.reset(new memory({ *weights_desc, *eng }));
        dweights.reset(new memory({ *weights_desc, *eng }));

        ref_y.reset(new memory({ *y_desc, *eng }));
        ref_hy.reset(new memory({ *hx_desc, *eng }));
        ref_cy.reset(new memory({ *hx_desc, *eng }));
        ref_dx.reset(new memory({ *x_desc, *eng }));
        ref_dhx.reset(new memory({ *hx_desc, *eng }));
        ref_dcx.reset(new memory({ *hx_desc, *eng }));
        ref_dweights.reset(new memory({ *weights_desc, *eng }));

        int iters = 20;
        flops = 2.0 * (4.0 * (double)p.test_rd.state_size)
                * (double)p.test_rd.batch_size
                * (2.0 * (double)p.test_rd.state_size + 2.0)
                * (double)p.test_rd.seq_length;

        // warm-up
        for (int _it = 0; _it < 5; _it++) {
            Forward();
            // Backward();
        }

        l_total = 0.0;
        for (int _it = 0; _it < iters; _it++) {
            Forward();
            // Backward();
        }

        // l_total = sec(l_start, l_end);
        printf("LSTM FWD benchmark - gflops = %.5g , time [s] = %.5g, GFLOPS = "
               "%.5g\n",
                flops * 1e-9, l_total / iters, iters * flops / l_total / 1e9);
    }

    void Forward()
    {
        auto rnn_fwd_desc = rnn_forward::desc(p.aprop_kind, p.aalgorithm,
                p.adirection, p.ainput_mode, p.test_ld.state_size,
                p.test_ld.num_layers, p.test_ld.seq_length,
                p.test_ld.state_outputs, *x_desc, *hx_desc, *y_desc,
                *weights_desc);
        rnn_fwd_prim_desc.reset(
                new rnn_forward::primitive_desc(rnn_fwd_desc, *eng));
        fill_data<data_t>(x->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)x->get_data_handle());
        fill_data<data_t>(hx->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)hx->get_data_handle());
        fill_data<data_t>(cx->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)cx->get_data_handle());
        fill_data<data_t>(
                weights->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)weights->get_data_handle());
        // Execute
        std::vector<primitive> pipeline;
        auto s = stream(stream::kind::lazy);

        struct timeval l_start, l_end;
        gettimeofday(&l_start, NULL);

        if (with_workspace) {
            auto workspace_primitive_desc
                    = rnn_fwd_prim_desc->workspace_primitive_desc();
            workspace.reset(new memory(workspace_primitive_desc));
        }
        auto l = rnn_forward(*rnn_fwd_prim_desc, x.get(), hx.get(), cx.get(),
                weights.get(), y.get(), hy.get(), cy.get(), workspace.get());
        pipeline.push_back(l);
        s.submit(pipeline).wait();

        gettimeofday(&l_end, NULL);
        l_total += (l_end.tv_sec - l_start.tv_sec)
                + (l_end.tv_usec - l_start.tv_usec) / 1000000.0;
    }

    void Backward()
    {
        auto pk = prop_kind::backward;
        auto rnn_bwd_desc = rnn_backward::desc(pk, p.aalgorithm, p.adirection,
                p.ainput_mode, p.test_ld.state_size, p.test_ld.num_layers,
                p.test_ld.seq_length, p.test_ld.state_outputs, *x_desc,
                *hx_desc, *y_desc, *weights_desc);
        rnn_bwd_prim_desc.reset(new rnn_backward::primitive_desc(
                rnn_bwd_desc, *eng, *rnn_fwd_prim_desc));
        fill_data<data_t>(dy->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)dy->get_data_handle());
        fill_data<data_t>(dhy->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)dhy->get_data_handle());
        fill_data<data_t>(dcy->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)dcy->get_data_handle());
        fill_data<data_t>(
                dweights->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)dweights->get_data_handle());
        fill_data<data_t>(
                ref_dweights->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)ref_dweights->get_data_handle());
        // Execute
        std::vector<primitive> pipeline;
        auto s = stream(stream::kind::lazy);

        struct timeval l_start, l_end;
        gettimeofday(&l_start, NULL);

        auto l = rnn_backward(*rnn_bwd_prim_desc, x.get(), hx.get(), cx.get(),
                dy.get(), dhy.get(), dcy.get(), weights.get(), workspace.get(),
                dx.get(), dhx.get(), dcx.get(), dweights.get());
        pipeline.push_back(l);
        s.submit(pipeline).wait();

        gettimeofday(&l_end, NULL);
        l_total += (l_end.tv_sec - l_start.tv_sec)
                + (l_end.tv_usec - l_start.tv_usec) / 1000000.0;
    }
};

using lstm_test_float = lstm_test<float>;
using lstm_test_params_float = lstm_test_params;

TEST_P(lstm_test_float, TestsLSTM)
{
}

INSTANTIATE_TEST_CASE_P(TestRNNForward0, lstm_test_float,
        ::testing::Values(
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 512, 512, 25, 1, 16, LSTM, UNIDIRECT, LINEAR, 0 } },
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 512, 512, 25, 1, 32, LSTM, UNIDIRECT, LINEAR, 0 } },
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 512, 512, 25, 1, 64, LSTM, UNIDIRECT, LINEAR, 0 } },
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 512, 512, 25, 1, 128, LSTM, UNIDIRECT, LINEAR, 0 } },
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 1024, 1024, 25, 1, 16, LSTM, UNIDIRECT, LINEAR, 0 } },
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 1024, 1024, 25, 1, 32, LSTM, UNIDIRECT, LINEAR, 0 } },
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 1024, 1024, 25, 1, 64, LSTM, UNIDIRECT, LINEAR, 0 } },
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 1024, 1024, 25, 1, 128, LSTM, UNIDIRECT, LINEAR,
                                0 } },
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 2048, 2048, 25, 1, 16, LSTM, UNIDIRECT, LINEAR, 0 } },
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 2048, 2048, 25, 1, 32, LSTM, UNIDIRECT, LINEAR, 0 } },
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 2048, 2048, 25, 1, 64, LSTM, UNIDIRECT, LINEAR, 0 } },
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 2048, 2048, 25, 1, 128, LSTM, UNIDIRECT, LINEAR,
                                0 } },
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 4096, 4096, 25, 1, 16, LSTM, UNIDIRECT, LINEAR, 0 } },
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 4096, 4096, 25, 1, 32, LSTM, UNIDIRECT, LINEAR, 0 } },
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 4096, 4096, 25, 1, 64, LSTM, UNIDIRECT, LINEAR, 0 } },
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 4096, 4096, 25, 1, 128, LSTM, UNIDIRECT, LINEAR,
                                0 } }));
} // namespace mkldnn
