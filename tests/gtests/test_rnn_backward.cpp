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
#include "gtest/gtest.h"
#include "utils.hpp"

namespace mkldnn {

template <typename data_t>
void compute_ref_lstm_bwd(const test_lstm_desc_t &ld, const memory::desc &x_d,
                          const memory::desc &hx_d, const memory::desc &y_d,
                          const memory::desc &weights_d, const memory &x,
                          const memory &hx, const memory &cx, const memory &dy,
                          const memory &dhy, const memory &dcy,
                          const memory &weights, const memory &ws,
                          const memory &dx, const memory &dhx,
                          const memory &dcx, const memory &dweights) {
  using namespace mkldnn::impl::utils;
  data_t *x_ptr = (data_t *)x.get_data_handle();
  data_t *hx_ptr = (data_t *)hx.get_data_handle();
  data_t *cx_ptr = (data_t *)cx.get_data_handle();
  data_t *dy_ptr = (data_t *)dy.get_data_handle();
  data_t *dhy_ptr = (data_t *)dhy.get_data_handle();
  data_t *dcy_ptr = (data_t *)dcy.get_data_handle();
  data_t *weights_ptr = (data_t *)weights.get_data_handle();
  data_t *ws_ptr = (data_t *)ws.get_data_handle();
  data_t *dx_ptr = (data_t *)dx.get_data_handle();
  data_t *dhx_ptr = (data_t *)dhx.get_data_handle();
  data_t *dcx_ptr = (data_t *)dcx.get_data_handle();
  data_t *dweights_ptr = (data_t *)dweights.get_data_handle();

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
  size_t temp_size =
      btmp * batch_size + gates_space_size + (2 * hout_space_size);
  data_t *ts_ = new data_t[temp_size];
  memset(ts_, 0, temp_size * sizeof(data_t));

  const size_t gates_space_off = 0;
  const size_t hout_space_off = gates_space_size;
  const size_t c_space_off = hout_space_off + hout_space_size;

  const size_t dgates_space_off = 0;
  const size_t dhout_space_off = dgates_space_off + gates_space_size;
  const size_t dc_space_off = dhout_space_off + hout_space_size;
  const size_t temp_space_off = dc_space_off + hout_space_size;

#pragma omp parallel for
  for (size_t seq = 0; seq < seq_length; seq++) {
    transpose<data_t>(dy_ptr + seq * h_size,
                      ts_ + dhout_space_off + (h_nlayer_size - h_size) +
                          seq * h_nlayer_size,
                      batch_size, state_size);
  }

  if (num_layers > 1) {
#pragma omp parallel for
    for (size_t ly = 0; ly < (num_layers - 1); ly++) {
      transpose<data_t>(dhy_ptr + ly * h_size,
                        ts_ + dhout_space_off +
                            (seq_length - 1) * h_nlayer_size + ly * h_size,
                        batch_size, state_size);
    }
  }
#pragma omp parallel for
  for (size_t ly = 0; ly < num_layers; ly++) {
    transpose<data_t>(dcy_ptr + ly * h_size,
                      ts_ + dc_space_off + (seq_length - 1) * h_nlayer_size +
                          ly * h_size,
                      batch_size, state_size);
  }

  for (int t = (seq_length - 1); t >= 0; t--) {
    for (int l = (num_layers - 1); l >= 0; l--) {
      if (t == 0) {
        transpose<data_t>(cx_ptr + l * h_size, ts_ + temp_space_off, batch_size,
                          state_size);
      }
      data_t it, ft, ot, gt, c_t_1, ct, dct, dht;
      data_t dit, dft, dot, dgt, dc_t_1;
      for (int h = 0; h < h_size; h++) {
        it = ws_ptr
            [gates_space_off + l * gates_size + t * gates_nlayer_size + h];
        ft = ws_ptr[gates_space_off + l * gates_size + t * gates_nlayer_size +
                    h_size + h];
        ot = ws_ptr[gates_space_off + l * gates_size + t * gates_nlayer_size +
                    2 * h_size + h];
        gt = ws_ptr[gates_space_off + l * gates_size + t * gates_nlayer_size +
                    3 * h_size + h];
        if (t == 0)
          c_t_1 = ts_[temp_space_off + h];
        else
          c_t_1 =
              ws_ptr[c_space_off + l * h_size + (t - 1) * h_nlayer_size + h];
        ct = ws_ptr[c_space_off + l * h_size + t * h_nlayer_size + h];
        dct = ts_[dc_space_off + l * h_size + t * h_nlayer_size + h];
        dht = ts_[dhout_space_off + l * h_size + t * h_nlayer_size + h];
        dct += (1 - pow(tanh(ct), 2)) * dht * ot;
        dc_t_1 = dct * ft;
        dit = dct * gt;
        dft = dct * c_t_1;
        dot = dht * tanh(ct);
        dgt = dct * it;
        dit = dit * it * (1 - it);
        dft = dft * ft * (1 - ft);
        dot = dot * ot * (1 - ot);
        dgt = dgt * (1 - pow(gt, 2));

        if (t == 0)
          ts_[temp_space_off + h_size + h] = dc_t_1;
        else
          ts_[dc_space_off + l * h_size + (t - 1) * h_nlayer_size + h] = dc_t_1;
        ts_[dgates_space_off + l * gates_size + t * gates_nlayer_size + h] =
            dit;
        ts_[dgates_space_off + l * gates_size + t * gates_nlayer_size + h_size +
            h] = dft;
        ts_[dgates_space_off + l * gates_size + t * gates_nlayer_size +
            2 * h_size + h] = dot;
        ts_[dgates_space_off + l * gates_size + t * gates_nlayer_size +
            3 * h_size + h] = dgt;
      }
      if (t == 0) {
        transpose<data_t>(ts_ + temp_space_off + h_size, dcx_ptr + l * h_size,
                          state_size, batch_size);
      }

      if ((t > 0) && (l > 0)) {
        // HinX
        directcopy<data_t>(ws_ptr + hout_space_off + (l - 1) * h_size +
                               t * h_nlayer_size,
                           ts_ + temp_space_off, state_size, batch_size);
        directcopy<data_t>(
            ws_ptr + hout_space_off + l * h_size + (t - 1) * h_nlayer_size,
            ts_ + temp_space_off + h_size, state_size, batch_size);
        array_set(ts_ + temp_space_off + 2 * h_size, 1.0, 2 * batch_size);
      } else if ((t == 0) && (l > 0)) {
        directcopy<data_t>(ws_ptr + hout_space_off + (l - 1) * h_size,
                           ts_ + temp_space_off, state_size, batch_size);
        transpose<data_t>(hx_ptr + l * h_size, ts_ + temp_space_off + h_size,
                          batch_size, state_size);
        array_set(ts_ + temp_space_off + 2 * h_size, 1.0, 2 * batch_size);
      } else if ((l == 0) && (t > 0)) {
        transpose<data_t>(x_ptr + t * x_size, ts_ + temp_space_off, batch_size,
                          input_size);
        directcopy<data_t>(ws_ptr + hout_space_off + (t - 1) * h_nlayer_size,
                           ts_ + temp_space_off + x_size, state_size,
                           batch_size);
        array_set(ts_ + temp_space_off + x_size + h_size, 1.0, 2 * batch_size);
      } else {
        transpose<data_t>(x_ptr, ts_ + temp_space_off, batch_size, input_size);
        transpose<data_t>(hx_ptr, ts_ + temp_space_off + x_size, batch_size,
                          state_size);
        array_set(ts_ + temp_space_off + x_size + h_size, 1.0, 2 * batch_size);
      }

      size_t in_size = (l == 0) ? input_size : state_size;
      size_t woffset = 0;
      if (l == 0) {
        woffset = 0;
      } else {
        woffset = w1_size + (l - 1) * wx_size;
      }
      gemm<data_t>(NOTRANS, TRANS, ts_ + temp_space_off,
                   ts_ + dgates_space_off + l * gates_size +
                       t * gates_nlayer_size,
                   dweights_ptr + woffset, in_size + state_size + 2,
                   4 * state_size, batch_size, 1);
      gemm<data_t>(NOTRANS, NOTRANS, weights_ptr + woffset,
                   ts_ + dgates_space_off + l * gates_size +
                       t * gates_nlayer_size,
                   ts_ + temp_space_off, in_size + state_size + 2, batch_size,
                   4 * state_size, 0);
      if ((t > 0) && (l > 0)) {
        directcopy<data_t>(ts_ + temp_space_off + h_size,
                           ts_ + dhout_space_off + l * h_size +
                               (t - 1) * h_nlayer_size,
                           state_size, batch_size);
      } else if ((l == 0) && (t > 0)) {
        transpose<data_t>(ts_ + temp_space_off, dx_ptr + t * x_size, input_size,
                          batch_size);
        directcopy<data_t>(ts_ + temp_space_off + x_size,
                           ts_ + dhout_space_off + (t - 1) * h_nlayer_size,
                           state_size, batch_size);
      } else if ((t == 0) && (l > 0)) {
        transpose<data_t>(ts_ + temp_space_off + h_size, dhx_ptr + l * h_size,
                          state_size, batch_size);
      } else {
        transpose<data_t>(ts_ + temp_space_off, dx_ptr, input_size, batch_size);
        transpose<data_t>(ts_ + temp_space_off + x_size, dhx_ptr, state_size,
                          batch_size);
      }
    }
  }
}

template <typename data_t>
class lstm_backward_test : public ::testing::TestWithParam<lstm_test_params> {
protected:
  virtual void SetUp() {
    using namespace mkldnn::impl::utils;
    lstm_test_params p = ::testing::TestWithParam<lstm_test_params>::GetParam();

    ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
    ASSERT_TRUE(p.aprop_kind == prop_kind::backward);
    auto eng = engine(p.engine_kind, 0);
    memory::data_type data_type = data_traits<data_t>::data_type;
    ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);
    test_lstm_desc_t ld = p.test_ld;

    const size_t w1_size =
        ld.state_size * (ld.state_size + ld.input_size + 2) * 4;
    const size_t wx_size =
        ld.state_size * (ld.state_size + ld.state_size + 2) * 4;
    const size_t total_w = ld.num_layers == 1
                               ? w1_size
                               : (w1_size + (ld.num_layers - 1) * wx_size);
    auto l_x_desc = create_md({ static_cast<int>(ld.seq_length),
                                static_cast<int>(ld.batch_size),
                                static_cast<int>(ld.input_size) },
                              data_type, p.rnx_format);
    auto l_hx_desc = create_md({ static_cast<int>(ld.num_layers),
                                 static_cast<int>(ld.batch_size),
                                 static_cast<int>(ld.state_size) },
                               data_type, p.rnx_format);
    auto l_y_desc = create_md({ static_cast<int>(ld.seq_length),
                                static_cast<int>(ld.batch_size),
                                static_cast<int>(ld.state_size) },
                              data_type, p.rnx_format);
    auto l_weights_desc =
        create_md({ static_cast<int>(total_w) }, data_type, memory::format::x);
    auto rnn_desc = rnn_backward::desc(p.aprop_kind, p.aalgorithm, p.adirection,
                                       p.ainput_mode, ld.state_size,
                                       ld.num_layers, ld.seq_length, l_x_desc,
                                       l_hx_desc, l_y_desc, l_weights_desc);

    auto rnn_prim_desc = rnn_backward::primitive_desc(rnn_desc, eng);
    auto x_primitive_desc = memory::primitive_desc(l_x_desc, eng);
    auto hx_primitive_desc = memory::primitive_desc(l_hx_desc, eng);
    auto cx_primitive_desc = memory::primitive_desc(l_hx_desc, eng);
    auto dy_primitive_desc = memory::primitive_desc(l_y_desc, eng);
    auto dhy_primitive_desc = memory::primitive_desc(l_hx_desc, eng);
    auto dcy_primitive_desc = memory::primitive_desc(l_hx_desc, eng);
    auto weights_primitive_desc = memory::primitive_desc(l_weights_desc, eng);
    auto workspace_primitive_desc = rnn_prim_desc.workspace_primitive_desc();
    auto dx_primitive_desc = memory::primitive_desc(l_x_desc, eng);
    auto dhx_primitive_desc = memory::primitive_desc(l_hx_desc, eng);
    auto dcx_primitive_desc = memory::primitive_desc(l_hx_desc, eng);
    auto dweights_primitive_desc = memory::primitive_desc(l_weights_desc, eng);

    auto x_size = x_primitive_desc.get_size();
    auto hx_size = hx_primitive_desc.get_size();
    auto cx_size = cx_primitive_desc.get_size();
    auto dy_size = dy_primitive_desc.get_size();
    auto dhy_size = dhy_primitive_desc.get_size();
    auto dcy_size = dcy_primitive_desc.get_size();
    auto weights_size = weights_primitive_desc.get_size();
    auto workspace_size = workspace_primitive_desc.get_size();
    auto dx_size = dx_primitive_desc.get_size();
    auto dhx_size = dhx_primitive_desc.get_size();
    auto dcx_size = dcx_primitive_desc.get_size();
    auto dweights_size = dweights_primitive_desc.get_size();

    data_t *x_data = new data_t[x_size / sizeof(data_t)];
    data_t *hx_data = new data_t[hx_size / sizeof(data_t)];
    data_t *cx_data = new data_t[cx_size / sizeof(data_t)];
    data_t *dy_data = new data_t[dy_size / sizeof(data_t)];
    data_t *dhy_data = new data_t[dhy_size / sizeof(data_t)];
    data_t *dcy_data = new data_t[dcy_size / sizeof(data_t)];
    data_t *weights_data = new data_t[weights_size / sizeof(data_t)];
    data_t *workspace_data = new data_t[workspace_size / sizeof(data_t)];
    data_t *dx_data = new data_t[dx_size / sizeof(data_t)];
    data_t *dhx_data = new data_t[dhx_size / sizeof(data_t)];
    data_t *dcx_data = new data_t[dcx_size / sizeof(data_t)];
    data_t *dweights_data = new data_t[dweights_size / sizeof(data_t)];

    data_t *ref_dx_data = new data_t[dx_size / sizeof(data_t)];
    data_t *ref_dhx_data = new data_t[dhx_size / sizeof(data_t)];
    data_t *ref_dcx_data = new data_t[dcx_size / sizeof(data_t)];
    data_t *ref_dw_data = new data_t[dweights_size / sizeof(data_t)];

    auto l_x = memory(x_primitive_desc, x_data);
    auto l_hx = memory(hx_primitive_desc, hx_data);
    auto l_cx = memory(cx_primitive_desc, cx_data);
    auto l_dy = memory(dy_primitive_desc, dy_data);
    auto l_dhy = memory(dhy_primitive_desc, dhy_data);
    auto l_dcy = memory(dcy_primitive_desc, dcy_data);
    auto l_weights = memory(weights_primitive_desc, weights_data);
    auto l_ws = memory(workspace_primitive_desc, workspace_data);
    auto l_dx = memory(dx_primitive_desc, dx_data);
    auto l_dhx = memory(dhx_primitive_desc, dhx_data);
    auto l_dcx = memory(dcx_primitive_desc, dcx_data);
    auto l_dweights = memory(weights_primitive_desc, dweights_data);

    auto l_ref_dx = memory(dx_primitive_desc, ref_dx_data);
    auto l_ref_dhx = memory(dhx_primitive_desc, ref_dhx_data);
    auto l_ref_dcx = memory(dcx_primitive_desc, ref_dcx_data);
    auto l_ref_dweights = memory(dweights_primitive_desc, ref_dw_data);

    // Only true for dense format
    fill_data<data_t>(l_x.get_primitive_desc().get_size() / sizeof(data_t),
                      (data_t *)l_x.get_data_handle());
    fill_data<data_t>(l_hx.get_primitive_desc().get_size() / sizeof(data_t),
                      (data_t *)l_hx.get_data_handle());
    fill_data<data_t>(l_cx.get_primitive_desc().get_size() / sizeof(data_t),
                      (data_t *)l_cx.get_data_handle());
    fill_data<data_t>(l_dy.get_primitive_desc().get_size() / sizeof(data_t),
                      (data_t *)l_dy.get_data_handle());
    fill_data<data_t>(l_dhy.get_primitive_desc().get_size() / sizeof(data_t),
                      (data_t *)l_dhy.get_data_handle());
    fill_data<data_t>(l_dcy.get_primitive_desc().get_size() / sizeof(data_t),
                      (data_t *)l_dcy.get_data_handle());
    fill_data<data_t>(l_weights.get_primitive_desc().get_size() /
                          sizeof(data_t),
                      (data_t *)l_weights.get_data_handle());
    fill_data<data_t>(l_ws.get_primitive_desc().get_size() / sizeof(data_t),
                      (data_t *)l_ws.get_data_handle());
    fill_data<data_t>(l_dweights.get_primitive_desc().get_size() /
                          sizeof(data_t),
                      (data_t *)l_dweights.get_data_handle());
    fill_data<data_t>(l_ref_dweights.get_primitive_desc().get_size() /
                          sizeof(data_t),
                      (data_t *)l_ref_dweights.get_data_handle());

    // Execute
    std::vector<primitive> pipeline;
    auto s = stream(stream::kind::lazy);
    auto l = rnn_backward(rnn_prim_desc, l_x, l_hx, l_cx, l_dy, l_dhy, l_dcy,
                          l_weights, l_ws, l_dx, l_dhx, l_dcx, l_dweights);
    pipeline.push_back(l);
    s.submit(pipeline).wait();

    compute_ref_lstm_bwd<data_t>(ld, l_x_desc, l_hx_desc, l_y_desc,
                                 l_weights_desc, l_x, l_hx, l_cx, l_dy, l_dhy,
                                 l_dcy, l_weights, l_ws, l_ref_dx, l_ref_dhx,
                                 l_ref_dcx, l_ref_dweights);
    compare_data<data_t>(l_ref_dx, l_dx);
    compare_data<data_t>(l_ref_dhx, l_dhx);
    compare_data<data_t>(l_ref_dcx, l_dcx);
    compare_data<data_t>(l_ref_dweights, l_dweights);

    delete[] x_data;
    delete[] hx_data;
    delete[] cx_data;
    delete[] dy_data;
    delete[] dhy_data;
    delete[] dcy_data;
    delete[] weights_data;
    delete[] workspace_data;
    delete[] dx_data;
    delete[] dhx_data;
    delete[] dcx_data;
    delete[] dweights_data;

    delete[] ref_dx_data;
    delete[] ref_dhx_data;
    delete[] ref_dcx_data;
    delete[] ref_dw_data;
  }
};

using lstm_backward_test_float = lstm_backward_test<float>;
using lstm_test_params_float = lstm_test_params;

TEST_P(lstm_backward_test_float, TestsRNN) {}

INSTANTIATE_TEST_CASE_P(TestRNNBackward, lstm_backward_test_float,
                        ::testing::Values(lstm_test_params_float{
                          prop_kind::backward,
                          engine::kind::cpu,
                          algorithm::rnn_lstm,
                          direction::rnn_unidirectional,
                          input_mode::rnn_linear_input,
                          memory::format::rnx,
                          { 8, 8, 4, 4, 4, LSTM, UNIDIRECT, LINEAR }
                        }));
}
