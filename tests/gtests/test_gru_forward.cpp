 /**
  * @file      test_gru_forward.cpp
  * @author    zhangshu(shu.zhang@intel.com)
  * @date      2017-12-06 09:41:24
  * @brief
  **/
#include "mkldnn_test_common.hpp"
#include "rnn_common.hpp"
#include "gtest/gtest.h"
namespace mkldnn {
typedef test_rnn_desc_t test_gru_desc_t;
typedef rnn_test_params test_gru_params_t;

template <typename data_t>
inline data_t sigmoid(data_t x) {
    return 1.0f / (1.0f + exp(-x));
}
/**
 * @brief:      single layer gru forward computation
 *
 * @params:     T:  seq_length   
 *              N:  batch_size
 *              I:  input_size
 *              H:  hidden_size
 *              bid: if bid is true, then time step is tn --> t0, 
 *                   else time step is t0 --> tn
 *              x:  [T, N, I]  input features
 *              hx: [N, H] initial hidden state
 *              weights: include wx[I, 3H], wh[H, 3H], bx[3H] and bh[3H]
 *              y:  [T, N, H] output features for each time step
 *              hy: [N, H] output feature for the last time step
 * 
 * @formula_list:   rt = sigmoid(xt * Wrx + brx + ht-1 * Wrh + brh)
 *                  zt = sigmoid(xt * Wzx + bzx + ht-1 * Wzh + bzh)
 *                  nt = tanh(xt * Wnx + bnx + rt * (ht-1 * Wnh + bnh))
 *                  ht = (1 - zt) * nt + zt * ht-1
 *
 */
template <typename data_t>
void grucell_fwd(const int T,
                 const int N,
                 const int I,
                 const int H,
                 bool bid,
                 const data_t* x,
                 const data_t* hx,
                 const data_t* weights,
                 data_t* y,
                 data_t* hy) {
    const data_t *wx = weights;
    const data_t *wh = wx + I * H * 3;
    const data_t *bx = wh + H * H * 3;
    const data_t *bh = bx + H * 3;
    const int col = H * 3;
    // need a buf to save [rt|zt|nt], size:[N, 3H]
    data_t *buf = new data_t[N * col];
    data_t *hbuf = new data_t[N * col];

    const data_t *brx = bx;
    const data_t *bzx = brx + H;
    const data_t *bnx = bzx + H;
    const data_t *brh = bh;
    const data_t *bzh = brh + H;
    const data_t *bnh = bzh + H;

    const data_t *xt = x;
    const data_t *ht_pre = hx;
    // t0 --> tn
    if (!bid) {
        for (int i = 0; i < T; ++i) {
            data_t *rt = buf;
            data_t *zt = rt + H;
            data_t *nt = zt + H;
            data_t *rh = hbuf;
            data_t *zh = rh + H;
            data_t *nh = zh + H;
            gemm<data_t>(NOTRANS, NOTRANS,     xt, wx,  buf, N, col, I, 0);
            gemm<data_t>(NOTRANS, NOTRANS, ht_pre, wh, hbuf, N, col, H, 0);
            for (int j = 0; j < N; ++j) {
                int row = i * N + j;
                for (int k = 0; k < H; ++k) {
                    rt[k] = sigmoid(rt[k] + brx[k] + rh[k] + brh[k]);
                    zt[k] = sigmoid(zt[k] + bzx[k] + zh[k] + bzh[k]);
                    nt[k] =    tanh(nt[k] + bnx[k] + rt[k] * (nh[k] + bnh[k]));
                    hy[j * H + k] = (1 - zt[k]) * nt[k]
                                    + zt[k] * ht_pre[j * H + k];
                    y[row * H + k] = hy[j * H + k];
                }
                rt += col;
                zt += col;
                nt += col;
                rh += col;
                zh += col;
                nh += col;
            }
            xt += N * I;
            ht_pre = hy;
        }
    } else {
    // tn --> t0
        xt = x + (T - 1) * N * I;
        for (int i = T - 1; i >= 0; --i) {
            data_t *rt = buf;
            data_t *zt = rt + H;
            data_t *nt = zt + H;
            data_t *rh = hbuf;
            data_t *zh = rh + H;
            data_t *nh = zh + H;
            gemm<data_t>(NOTRANS, NOTRANS,     xt, wx,  buf, N, col, I, 0);
            gemm<data_t>(NOTRANS, NOTRANS, ht_pre, wh, hbuf, N, col, H, 0);
            for (int j = 0; j < N; ++j) {
                int row = i * N + j;
                for (int k = 0; k < H; ++k) {
                    rt[k] = sigmoid(rt[k] + brx[k] + rh[k] + brh[k]);
                    zt[k] = sigmoid(zt[k] + bzx[k] + zh[k] + bzh[k]);
                    nt[k] =    tanh(nt[k] + bnx[k] + rt[k] * (nh[k] + bnh[k]));
                    hy[j * H + k] = (1 - zt[k]) * nt[k]
                                    + zt[k] * ht_pre[j * H + k];
                    y[row * H + k] = hy[j * H + k];
                }
                rt += col;
                zt += col;
                nt += col;
                rh += col;
                zh += col;
                nh += col;
            }
            xt -= N * I;
            ht_pre = hy;
        }
    }
    delete []buf;
    delete []hbuf;
}

/**
 * @brief: The test function to compute gru_forward.
 */
template <typename data_t>
void compute_ref_gru_fwd(const test_gru_desc_t &gd,
        const memory &x,
        const memory &hx,
        const memory &weights,
        const memory &y,
        const memory &hy) {
    const int T = static_cast<int>(gd.seq_length);
    const int N = static_cast<int>(gd.batch_size);
    const int I = static_cast<int>(gd.input_size);
    const int H = static_cast<int>(gd.state_size);
    const int nd = gd.direction;  // num_direction
    const int nl = gd.num_layers;
    data_t *x_ptr = static_cast<data_t*>(x.get_data_handle());
    data_t *hx_ptr = static_cast<data_t*>(hx.get_data_handle());
    data_t *weights_ptr = static_cast<data_t*>(weights.get_data_handle());
    data_t *y_ptr = static_cast<data_t*>(y.get_data_handle());
    data_t *hy_ptr = static_cast<data_t*>(hy.get_data_handle());
    // if bidirect, need a tmp memory to save reorder y [nd * T, N, H]
    data_t *reorder_y = y_ptr;
    data_t *y_bid = NULL;
    if (nd == BIDIRECT) {
        reorder_y = new data_t[T * N * H * nd];
        y_bid = reorder_y + T * N * H;
    }
    // multi layer
    for (int l = 0; l < nl; ++l) {
        int input_size = (l == 0) ? I : nd * H;
        grucell_fwd(T, N, input_size, H, false,
                    x_ptr, hx_ptr, weights_ptr, reorder_y, hy_ptr);
        hx_ptr += N * H;
        weights_ptr +=  (input_size + H + 2) * H * 3;
        hy_ptr += N * H;
        // bidirectional
        if (nd == BIDIRECT) {
            grucell_fwd(T, N, input_size, H, true,
                        x_ptr, hx_ptr, weights_ptr, y_bid, hy_ptr);
            hx_ptr += N * H;
            weights_ptr +=  (input_size + H + 2) * H * 3;
            hy_ptr += N * H;
            // y_ptr:[T, N, H * nd], reorder_y:[nd * T, N, H]
            for (int i = 0; i < T; ++i) {
                for (int j = 0; j < N; ++j) {
                    int row = i * N + j;
                    for (int k = 0; k < H; ++k) {
                        y_ptr[row * H * nd + k] = reorder_y[row * H + k];
                    }
                    for (int k = 0; k < H; ++k) {
                        y_ptr[row * H * nd + H + k] =
                                        reorder_y[(row + T * N) * H + k];
                    }
                }
            }
        }
        x_ptr = y_ptr;
    }
    if (nd == BIDIRECT) {
        delete []reorder_y;
    }
}

template <typename data_t>
class gru_forward_test : public ::testing::TestWithParam<test_gru_params_t> {
private:
    std::shared_ptr<memory> workspace;
    std::shared_ptr<memory::desc> x_desc;
    std::shared_ptr<memory::desc> hx_desc;
    std::shared_ptr<memory::desc> weights_desc;
    std::shared_ptr<memory::desc> y_desc;
    std::shared_ptr<memory> x;
    std::shared_ptr<memory> hx;
    std::shared_ptr<memory> weights;
    std::shared_ptr<memory> hy;
    std::shared_ptr<memory> y;
    std::shared_ptr<rnn_forward::primitive_desc> rnn_fwd_prim_desc;
    bool with_workspace;
    // testcase's out
    std::shared_ptr<memory> ref_hy;
    std::shared_ptr<memory> ref_y;
    // testcase's params
    test_gru_params_t gru_param;

protected:
    virtual void SetUp() {
        gru_param = ::testing::TestWithParam <test_gru_params_t>::GetParam();

        ASSERT_TRUE(gru_param.engine_kind == engine::kind::cpu);
        ASSERT_TRUE(gru_param.aprop_kind == prop_kind::forward_inference);
        ASSERT_TRUE(gru_param.aalgorithm == algorithm::rnn_gru);
        ASSERT_TRUE(gru_param.adirection == direction::rnn_unidirectional ||
                    gru_param.adirection == direction::rnn_bidirectional);
        ASSERT_TRUE(gru_param.ainput_mode == input_mode::rnn_skip_input);
        auto eng = engine(gru_param.engine_kind, 0);
        memory::data_type data_type_all = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type_all, mkldnn::memory::data_type::f32);
        test_gru_desc_t &gd = gru_param.test_rd;
        with_workspace = gru_param.aprop_kind == prop_kind::forward_training;
        const int T = static_cast<int>(gd.seq_length);
        const int N = static_cast<int>(gd.batch_size);
        const int I = static_cast<int>(gd.input_size);
        const int H = static_cast<int>(gd.state_size);
        const int nd = static_cast<int>(gd.direction);
        const int nl = static_cast<int>(gd.num_layers);
        // size1 & size2 equals wx_size + wh_size + bx_size + bh_size
        const int size1 = (I + H + 2) * H * 3 * nd;  // first layer
        const int size2 = (nd*H + H + 2) * H * 3 * nd;  // other layers
        x_desc.reset(new memory::desc({T, N, I},
                     data_type_all, gru_param.rnx_format));
        hx_desc.reset(new memory::desc({nl * nd, N, H},
                      data_type_all, gru_param.rnx_format));
        weights_desc.reset(new memory::desc({size1 + (nl - 1) * size2},
                           data_type_all, memory::format::x));
        y_desc.reset(new memory::desc({T, N, H * nd},
                     data_type_all, gru_param.rnx_format));
        x.reset(new memory({*x_desc, eng}));
        hx.reset(new memory({*hx_desc, eng}));
        weights.reset(new memory({*weights_desc, eng}));
        y.reset(new memory({*y_desc, eng}));
        hy.reset(new memory({*hx_desc, eng}));
        ref_y.reset(new memory({*y_desc, eng}));
        ref_hy.reset(new memory({*hx_desc, eng}));
        Forward(eng);
    }
    void Forward(engine eng) {
        auto rnn_fwd_desc = rnn_forward::desc(
                gru_param.aprop_kind,
                gru_param.aalgorithm,
                gru_param.adirection,
                gru_param.ainput_mode,
                gru_param.test_rd.state_size,
                gru_param.test_rd.num_layers,
                gru_param.test_rd.seq_length,
                gru_param.test_rd.state_outputs,
                *x_desc, *hx_desc, *y_desc, *weights_desc);
        rnn_fwd_prim_desc.reset(
                new rnn_forward::primitive_desc(rnn_fwd_desc, eng));
        // random fill inputs and param
        fill_data<data_t>(x->get_primitive_desc().get_size() / sizeof(data_t),
                static_cast<data_t*>(x->get_data_handle()));
        fill_data<data_t>(hx->get_primitive_desc().get_size() / sizeof(data_t),
                static_cast<data_t*>(hx->get_data_handle()));
        fill_data<data_t>(
                weights->get_primitive_desc().get_size() / sizeof(data_t),
                static_cast<data_t*>(weights->get_data_handle()));
        // Execute
        std::vector<primitive> pipeline;
        auto s = stream(stream::kind::lazy);
        if (with_workspace) {
            auto workspace_primitive_desc =
                                rnn_fwd_prim_desc->workspace_primitive_desc();
            workspace.reset(new memory(workspace_primitive_desc));
        }
        if (gru_param.test_rd.state_outputs) {
            auto l = rnn_forward(*rnn_fwd_prim_desc, *x, *hx, *weights,
                    *y, *hy, *workspace);
            pipeline.push_back(l);
            s.submit(pipeline).wait();
        } else {
            auto l = rnn_forward(*rnn_fwd_prim_desc, *x, *hx,
                (const primitive::at &)*weights, *y, *workspace);
            pipeline.push_back(l);
            s.submit(pipeline).wait();
        }
        compute_ref_gru_fwd<data_t>(gru_param.test_rd,
                                    *x, *hx, *weights, *ref_y, *ref_hy);
        if (gru_param.test_rd.state_outputs) {
            compare_data<data_t>(*ref_y, *y);
            compare_data<data_t>(*ref_hy, *hy);
        } else {
            compare_data<data_t>(*ref_y, *y);
        }
    }
};

using gru_forward_test_float = gru_forward_test<float>;

TEST_P(gru_forward_test_float, TestsRNN)
{}
INSTANTIATE_TEST_CASE_P(
    TestRNNForward0,
    gru_forward_test_float,
    ::testing::Values(
        test_gru_params_t {
            prop_kind::forward_inference,
            engine::kind::cpu,
            algorithm::rnn_gru,
            direction::rnn_unidirectional,
            input_mode::rnn_skip_input,
            memory::format::rnx,
            { 128, 128, 10, 1, 32, GRU, UNIDIRECT, SKIP, 0 }
        },
        test_gru_params_t {
            prop_kind::forward_inference,
            engine::kind::cpu,
            algorithm::rnn_gru,
            direction::rnn_unidirectional,
            input_mode::rnn_skip_input,
            memory::format::rnx,
            { 800, 800, 10, 3, 32, GRU, UNIDIRECT, SKIP, 0 }
        },
        test_gru_params_t {
            prop_kind::forward_inference,
            engine::kind::cpu,
            algorithm::rnn_gru,
            direction::rnn_bidirectional,
            input_mode::rnn_skip_input,
            memory::format::rnx,
            { 128, 128, 10, 1, 32, GRU, BIDIRECT, SKIP, 0 }
        },
        test_gru_params_t {
            prop_kind::forward_inference,
            engine::kind::cpu,
            algorithm::rnn_gru,
            direction::rnn_bidirectional,
            input_mode::rnn_skip_input,
            memory::format::rnx,
            { 800, 800, 10, 3, 32, GRU, BIDIRECT, SKIP, 0 }
        }
    )
);

INSTANTIATE_TEST_CASE_P(
    TestRNNForward1,
    gru_forward_test_float,
    ::testing::Values(
        test_gru_params_t{prop_kind::forward_inference,
                          engine::kind::cpu,
                          algorithm::rnn_gru,
                          direction::rnn_unidirectional,
                          input_mode::rnn_skip_input,
                          memory::format::rnx,
                          { 128, 128, 10, 1, 32, GRU, UNIDIRECT, SKIP, 1 }},
        test_gru_params_t{prop_kind::forward_inference,
                          engine::kind::cpu,
                          algorithm::rnn_gru,
                          direction::rnn_unidirectional,
                          input_mode::rnn_skip_input,
                          memory::format::rnx,
                          { 800, 800, 10, 3, 32, GRU, UNIDIRECT, SKIP, 1 }},
        test_gru_params_t{prop_kind::forward_inference,
                          engine::kind::cpu,
                          algorithm::rnn_gru,
                          direction::rnn_bidirectional,
                          input_mode::rnn_skip_input,
                          memory::format::rnx,
                          { 128, 128, 10, 1, 32, GRU, BIDIRECT, SKIP, 1 }},
        test_gru_params_t{prop_kind::forward_inference,
                          engine::kind::cpu,
                          algorithm::rnn_gru,
                          direction::rnn_bidirectional,
                          input_mode::rnn_skip_input,
                          memory::format::rnx,
                          { 800, 800, 10, 3, 32, GRU, BIDIRECT, SKIP, 1 }}
    )
);

}  // namespace mkldnn
