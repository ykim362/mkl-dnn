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

namespace mkldnn {
template <typename data_t>
void compute_ref_lstm_fwd(const test_lstm_desc_t &ld, const memory::desc &x_d,
        const memory::desc &hx_d, const memory::desc &y_d,
        const memory::desc &weights_d, const memory &x, const memory &hx,
        const memory &cx, const memory &weights, const memory &y,
        const memory &hy, const memory &cy) {
    using namespace mkldnn::impl::utils;

    data_t *x_ptr = (data_t *)x.get_data_handle();
    data_t *hx_ptr = (data_t *)hx.get_data_handle();
    data_t *cx_ptr = (data_t *)cx.get_data_handle();
    data_t *weights_ptr = (data_t *)weights.get_data_handle();
    data_t *y_ptr = (data_t *)y.get_data_handle();
    data_t *hy_ptr = (data_t *)hy.get_data_handle();
    data_t *cy_ptr = (data_t *)cy.get_data_handle();

    const size_t state_size = ld.state_size;
    const size_t input_size = ld.input_size;
    const size_t seq_length = ld.seq_length;
    const size_t num_layers = ld.num_layers;
    const size_t batch_size = ld.batch_size;
    const size_t direction = ld.direction;
    const size_t total_layers = num_layers * direction;
    const size_t w1_size = state_size * (state_size + input_size + 2) * 4;
    const size_t wx_size = state_size * (state_size + state_size + 2) * 4;
    const size_t h_size = batch_size * state_size;
    const size_t x_size = batch_size * input_size;
    const size_t h_nlayer_size = h_size * num_layers;
    const size_t gates_size = h_size * 4;
    const size_t gates_nlayer_size = gates_size * num_layers;
    const size_t gates_space_size = gates_nlayer_size * seq_length * direction;
    const size_t hout_space_size = h_nlayer_size * seq_length * direction;

    const size_t ws_size = gates_space_size + hout_space_size * 2;
    data_t *ws_ptr = new data_t[ws_size];

    size_t bsize = (input_size > state_size) ? input_size : state_size;
    size_t tmp1 = bsize + state_size + 2;
    size_t tmp2 = state_size * 4;
    size_t btmp = (tmp1 > tmp2) ? tmp1 : tmp2;
    size_t temp_size = btmp * batch_size;
    data_t *ts_ = new data_t[temp_size];

    const size_t gates_space_off = 0;
    const size_t hout_space_off = gates_space_size;
    const size_t c_space_off = hout_space_off + hout_space_size;
    size_t w_off = 0;
    size_t in_size = 0;
    size_t wa = w1_size + (num_layers - 1) * wx_size;
    size_t dl, rl, roff, rt;

    for (int l = 0; l < total_layers; l++) {
        dl = l / num_layers;
        rl = l % num_layers;
        roff = (rl == 0) ? 0 : (w1_size + (rl - 1) * wx_size);
        w_off = wa * dl + roff;
        in_size = (rl == 0) ? input_size : state_size;
        if (l / num_layers == 0) {
            for (int t = 0; t < seq_length; t++) {
                // Hin
                if (t == 0 && l == 0) {
                    transpose<data_t>(x_ptr, ts_, batch_size, input_size);
                    transpose<data_t>(
                            hx_ptr, ts_ + x_size, batch_size, state_size);
                    array_set(ts_ + x_size + h_size, 1.0, 2 * batch_size);
                } else if (t == 0 && l > 0) {
                    directcopy<data_t>(
                            ws_ptr + hout_space_off + (l - 1) * h_size, ts_,
                            state_size, batch_size);
                    transpose<data_t>(hx_ptr + l * h_size, ts_ + h_size,
                            batch_size, state_size);
                    array_set(ts_ + h_size + h_size, 1.0, 2 * batch_size);
                } else if (t > 0 && l == 0) {
                    transpose<data_t>(
                            x_ptr + t * x_size, ts_, batch_size, input_size);
                    directcopy<data_t>(
                            ws_ptr + hout_space_off + (t - 1) * h_nlayer_size,
                            ts_ + x_size, state_size, batch_size);
                    array_set(ts_ + x_size + h_size, 1.0, 2 * batch_size);
                } else if (t > 0 && l > 0) {
                    directcopy<data_t>(ws_ptr + hout_space_off
                                    + (l - 1) * h_size + t * h_nlayer_size,
                            ts_, state_size, batch_size);
                    directcopy<data_t>(ws_ptr + hout_space_off + l * h_size
                                    + (t - 1) * h_nlayer_size,
                            ts_ + h_size, state_size, batch_size);
                    array_set(ts_ + h_size + h_size, 1.0, 2 * batch_size);
                }
                gemm<data_t>(TRANS, NOTRANS, weights_ptr + w_off, ts_,
                        ws_ptr + gates_space_off + l * gates_size
                                + t * gates_nlayer_size,
                        4 * state_size, batch_size, in_size + state_size + 2,
                        0);
                if (t == 0)
                    transpose<data_t>(
                            cx_ptr + l * h_size, ts_, batch_size, state_size);
                for (int h = 0; h < h_size; h++) {
                    data_t it = ws_ptr[gates_space_off + l * gates_size
                            + t * gates_nlayer_size + h];
                    data_t ft = ws_ptr[gates_space_off + l * gates_size
                            + t * gates_nlayer_size + h_size + h];
                    data_t ot = ws_ptr[gates_space_off + l * gates_size
                            + t * gates_nlayer_size + 2 * h_size + h];
                    data_t gt = ws_ptr[gates_space_off + l * gates_size
                            + t * gates_nlayer_size + 3 * h_size + h];
                    it = 1 / (1 + exp(-it));
                    ft = 1 / (1 + exp(-ft));
                    ot = 1 / (1 + exp(-ot));
                    gt = tanh(gt);
                    data_t c_t_1;
                    if (t == 0) {
                        c_t_1 = ts_[h];
                    } else {
                        c_t_1 = ws_ptr[c_space_off + l * h_size
                                + (t - 1) * h_nlayer_size + h];
                    }
                    data_t ct = c_t_1 * ft + gt * it;
                    data_t ht = ot * tanh(ct);
                    ws_ptr[hout_space_off + l * h_size + t * h_nlayer_size + h]
                            = ht;
                    ws_ptr[c_space_off + l * h_size + t * h_nlayer_size + h]
                            = ct;
                }
                // save output
                if (l == num_layers - 1) {
                    transpose<data_t>(ws_ptr + hout_space_off
                                    + (h_nlayer_size - h_size)
                                    + t * h_nlayer_size,
                            y_ptr + t * h_size * direction, state_size,
                            batch_size);
                }

                if (direction == 1 && t == (seq_length - 1)) {
                    if (hy_ptr != nullptr)
                        transpose<data_t>(ws_ptr + hout_space_off
                                        + (seq_length - 1) * h_nlayer_size
                                        + l * h_size,
                                hy_ptr + l * h_size, state_size, batch_size);
                    if (cy_ptr != nullptr)
                        transpose<data_t>(ws_ptr + c_space_off
                                        + (seq_length - 1) * h_nlayer_size
                                        + l * h_size,
                                cy_ptr + l * h_size, state_size, batch_size);
                }
            }
        } else if (l / num_layers == 1) {
            for (int t = (seq_length - 1); t >= 0; t--) {
                rt = 2 * seq_length - t - 1;
                if (rl == 0) {
                    transpose<data_t>(
                            x_ptr + t * x_size, ts_, batch_size, input_size);
                    directcopy<data_t>(
                            ws_ptr + hout_space_off + (rt - 1) * h_nlayer_size,
                            ts_ + x_size, state_size, batch_size);
                    array_set(ts_ + x_size + h_size, 1.0, 2 * batch_size);
                } else if (rl > 0) {
                    directcopy<data_t>(ws_ptr + hout_space_off
                                    + (rl - 1) * h_size + rt * h_nlayer_size,
                            ts_, state_size, batch_size);
                    directcopy<data_t>(ws_ptr + hout_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            ts_ + h_size, state_size, batch_size);
                    array_set(ts_ + h_size + h_size, 1.0, 2 * batch_size);
                }
                gemm<data_t>(TRANS, NOTRANS, weights_ptr + w_off, ts_,
                        ws_ptr + gates_space_off + rl * gates_size
                                + rt * gates_nlayer_size,
                        4 * state_size, batch_size, in_size + state_size + 2,
                        0);
                for (int h = 0; h < h_size; h++) {
                    data_t it = ws_ptr[gates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + h];
                    data_t ft = ws_ptr[gates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + h_size + h];
                    data_t ot = ws_ptr[gates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + 2 * h_size + h];
                    data_t gt = ws_ptr[gates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + 3 * h_size + h];
                    it = 1 / (1 + exp(-it));
                    ft = 1 / (1 + exp(-ft));
                    ot = 1 / (1 + exp(-ot));
                    gt = tanh(gt);
                    data_t c_t_1 = ws_ptr[c_space_off + rl * h_size
                            + (rt - 1) * h_nlayer_size + h];
                    data_t ct = c_t_1 * ft + gt * it;
                    data_t ht = ot * tanh(ct);
                    ws_ptr[hout_space_off + rl * h_size + rt * h_nlayer_size
                            + h]
                            = ht;
                    ws_ptr[c_space_off + rl * h_size + rt * h_nlayer_size + h]
                            = ct;
                }
                // save output
                if (rl == num_layers - 1) {
                    transpose<data_t>(ws_ptr + hout_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            y_ptr + t * h_size * direction + h_size, state_size,
                            batch_size);
                }

                if (direction == 2 && t == 0) {
                    if (hy_ptr != nullptr)
                        transpose<data_t>(ws_ptr + hout_space_off
                                        + rt * h_nlayer_size + rl * h_size,
                                hy_ptr + rl * h_size, state_size, batch_size);
                    if (cy_ptr != nullptr)
                        transpose<data_t>(ws_ptr + c_space_off
                                        + rt * h_nlayer_size + rl * h_size,
                                cy_ptr + rl * h_size, state_size, batch_size);
                }
            }
        }
    }
}
template <typename data_t>
void compute_ref_lstm_bwd(const test_lstm_desc_t &ld, const memory::desc &x_d,
        const memory::desc &hx_d, const memory::desc &y_d,
        const memory::desc &weights_d, const memory &x, const memory &hx,
        const memory &cx, const memory &dy, const memory &dhy,
        const memory &dcy, const memory &weights, const memory &ws,
        const memory &dx, const memory &dhx, const memory &dcx,
        const memory &dweights) {
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
    const size_t direction = ld.direction;
    const size_t total_layers = num_layers * direction;
    const size_t w1_size = state_size * (state_size + input_size + 2) * 4;
    const size_t wx_size = state_size * (state_size + state_size + 2) * 4;
    const size_t h_size = batch_size * state_size;
    const size_t x_size = batch_size * input_size;
    const size_t h_nlayer_size = h_size * num_layers;
    const size_t gates_size = h_size * 4;
    const size_t gates_nlayer_size = gates_size * num_layers;
    const size_t gates_space_size = gates_nlayer_size * seq_length * direction;
    const size_t hout_space_size = h_nlayer_size * seq_length * direction;

    size_t bsize = (input_size > state_size) ? input_size : state_size;
    size_t tmp1 = bsize + state_size + 2;
    size_t tmp2 = state_size * 4;
    size_t btmp = (tmp1 > tmp2) ? tmp1 : tmp2;
    size_t temp_size
            = btmp * batch_size + gates_space_size + (2 * hout_space_size);
    data_t *ts_ = new data_t[temp_size];
    memset(ts_, 0, temp_size * sizeof(data_t));

    const size_t gates_space_off = 0;
    const size_t hout_space_off = gates_space_size;
    const size_t c_space_off = hout_space_off + hout_space_size;

    const size_t dgates_space_off = 0;
    const size_t dhout_space_off = dgates_space_off + gates_space_size;
    const size_t dc_space_off = dhout_space_off + hout_space_size;
    const size_t temp_space_off = dc_space_off + hout_space_size;
    size_t w_off = 0;
    size_t in_size = 0;
    size_t wa = w1_size + (num_layers - 1) * wx_size;
    size_t dl, rl, roff, rt;

    for (size_t seq = 0; seq < seq_length; seq++) {
        transpose<data_t>(dy_ptr + seq * h_size * direction,
                ts_ + dhout_space_off + (h_nlayer_size - h_size)
                        + seq * h_nlayer_size,
                batch_size, state_size);
        if (direction == 2)
            transpose<data_t>(dy_ptr + seq * h_size * direction + h_size,
                    ts_ + dhout_space_off + (h_nlayer_size - h_size)
                            + (seq_length * direction - 1 - seq)
                                    * h_nlayer_size,
                    batch_size, state_size);
    }
    if (ld.state_outputs) {
        for (size_t ly = 0; ly < num_layers; ly++) {
            transpose<data_t>(dhy_ptr + ly * h_size, ts_ + temp_space_off,
                    batch_size, state_size);
            axpycopy<data_t>(ts_ + temp_space_off, ts_ + dhout_space_off
                            + (seq_length * direction - 1) * h_nlayer_size
                            + ly * h_size,
                    batch_size, state_size);
        }
        for (size_t ly = 0; ly < num_layers; ly++) {
            transpose<data_t>(dcy_ptr + ly * h_size, ts_ + dc_space_off
                            + (seq_length * direction - 1) * h_nlayer_size
                            + ly * h_size,
                    batch_size, state_size);
        }
    }
    for (int l = (total_layers - 1); l >= 0; l--) {
        dl = l / num_layers;
        rl = l % num_layers;
        roff = (rl == 0) ? 0 : (w1_size + (rl - 1) * wx_size);
        w_off = wa * dl + roff;
        in_size = (rl == 0) ? input_size : state_size;
        if (dl == 1) {
            for (int t = 0; t < seq_length; t++) {
                rt = 2 * seq_length - t - 1;
                data_t it, ft, ot, gt, c_t_1, ct, dct, dht;
                data_t dit, dft, dot, dgt, dc_t_1;
                for (int h = 0; h < h_size; h++) {
                    it = ws_ptr[gates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + h];
                    ft = ws_ptr[gates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + h_size + h];
                    ot = ws_ptr[gates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + 2 * h_size + h];
                    gt = ws_ptr[gates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + 3 * h_size + h];
                    c_t_1 = ws_ptr[c_space_off + rl * h_size
                            + (rt - 1) * h_nlayer_size + h];
                    ct = ws_ptr[c_space_off + rl * h_size + rt * h_nlayer_size
                            + h];
                    dct = ts_[dc_space_off + rl * h_size + rt * h_nlayer_size
                            + h];
                    dht = ts_[dhout_space_off + rl * h_size + rt * h_nlayer_size
                            + h];
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
                    ts_[dc_space_off + rl * h_size + (rt - 1) * h_nlayer_size
                            + h]
                            = dc_t_1;
                    ts_[dgates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + h]
                            = dit;
                    ts_[dgates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + h_size + h]
                            = dft;
                    ts_[dgates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + 2 * h_size + h]
                            = dot;
                    ts_[dgates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + 3 * h_size + h]
                            = dgt;
                }
                if (rl == 0) {
                    transpose<data_t>(x_ptr + t * x_size, ts_ + temp_space_off,
                            batch_size, input_size);
                    directcopy<data_t>(
                            ws_ptr + hout_space_off + (rt - 1) * h_nlayer_size,
                            ts_ + temp_space_off + x_size, state_size,
                            batch_size);
                    array_set(ts_ + temp_space_off + x_size + h_size, 1.0,
                            2 * batch_size);
                } else if (rl > 0) {
                    directcopy<data_t>(ws_ptr + hout_space_off
                                    + (rl - 1) * h_size + rt * h_nlayer_size,
                            ts_ + temp_space_off, state_size, batch_size);
                    directcopy<data_t>(ws_ptr + hout_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            ts_ + temp_space_off + h_size, state_size,
                            batch_size);
                    array_set(ts_ + temp_space_off + h_size + h_size, 1.0,
                            2 * batch_size);
                }

                gemm<data_t>(NOTRANS, TRANS, ts_ + temp_space_off,
                        ts_ + dgates_space_off + rl * gates_size
                                + rt * gates_nlayer_size,
                        dweights_ptr + w_off, in_size + state_size + 2,
                        4 * state_size, batch_size, 1);

                gemm<data_t>(NOTRANS, NOTRANS, weights_ptr + w_off,
                        ts_ + dgates_space_off + rl * gates_size
                                + rt * gates_nlayer_size,
                        ts_ + temp_space_off, in_size + state_size + 2,
                        batch_size, 4 * state_size, 0);
                if (rl == 0) {
                    transpose<data_t>(ts_ + temp_space_off, dx_ptr + t * x_size,
                            input_size, batch_size);
                    axpycopy<data_t>(ts_ + temp_space_off + x_size,
                            ts_ + dhout_space_off + (rt - 1) * h_nlayer_size,
                            state_size, batch_size);
                } else if (rl > 0) {
                    axpycopy<data_t>(ts_ + temp_space_off, ts_ + dhout_space_off
                                    + (rl - 1) * h_size + rt * h_nlayer_size,
                            state_size, batch_size);
                    axpycopy<data_t>(ts_ + temp_space_off + h_size,
                            ts_ + dhout_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            state_size, batch_size);
                }
            }
        } else if (dl == 0) {
            for (int t = (seq_length - 1); t >= 0; t--) {
                rt = t;
                if (rt == 0) {
                    transpose<data_t>(cx_ptr + l * h_size, ts_ + temp_space_off,
                            batch_size, state_size);
                }
                data_t it, ft, ot, gt, c_t_1, ct, dct, dht;
                data_t dit, dft, dot, dgt, dc_t_1;
                for (int h = 0; h < h_size; h++) {
                    it = ws_ptr[gates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + h];
                    ft = ws_ptr[gates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + h_size + h];
                    ot = ws_ptr[gates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + 2 * h_size + h];
                    gt = ws_ptr[gates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + 3 * h_size + h];
                    if (t == 0)
                        c_t_1 = ts_[temp_space_off + h];
                    else
                        c_t_1 = ws_ptr[c_space_off + rl * h_size
                                + (rt - 1) * h_nlayer_size + h];
                    ct = ws_ptr[c_space_off + rl * h_size + rt * h_nlayer_size
                            + h];
                    dct = ts_[dc_space_off + rl * h_size + rt * h_nlayer_size
                            + h];
                    dht = ts_[dhout_space_off + rl * h_size + rt * h_nlayer_size
                            + h];
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
                    if (rt == 0)
                        ts_[temp_space_off + h_size + h] = dc_t_1;
                    else
                        ts_[dc_space_off + rl * h_size
                                + (rt - 1) * h_nlayer_size + h]
                                = dc_t_1;
                    ts_[dgates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + h]
                            = dit;
                    ts_[dgates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + h_size + h]
                            = dft;
                    ts_[dgates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + 2 * h_size + h]
                            = dot;
                    ts_[dgates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + 3 * h_size + h]
                            = dgt;
                }
                if (rt == 0) {
                    transpose<data_t>(ts_ + temp_space_off + h_size,
                            dcx_ptr + l * h_size, state_size, batch_size);
                }

                if ((rt > 0) && (rl > 0)) {
                    // HinX
                    directcopy<data_t>(ws_ptr + hout_space_off
                                    + (rl - 1) * h_size + rt * h_nlayer_size,
                            ts_ + temp_space_off, state_size, batch_size);
                    directcopy<data_t>(ws_ptr + hout_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            ts_ + temp_space_off + h_size, state_size,
                            batch_size);
                    array_set(ts_ + temp_space_off + 2 * h_size, 1.0,
                            2 * batch_size);
                } else if ((rt == 0) && (rl > 0)) {
                    directcopy<data_t>(
                            ws_ptr + hout_space_off + (rl - 1) * h_size,
                            ts_ + temp_space_off, state_size, batch_size);
                    transpose<data_t>(hx_ptr + l * h_size,
                            ts_ + temp_space_off + h_size, batch_size,
                            state_size);
                    array_set(ts_ + temp_space_off + 2 * h_size, 1.0,
                            2 * batch_size);
                } else if ((rl == 0) && (rt > 0)) {
                    transpose<data_t>(x_ptr + rt * x_size, ts_ + temp_space_off,
                            batch_size, input_size);
                    directcopy<data_t>(
                            ws_ptr + hout_space_off + (rt - 1) * h_nlayer_size,
                            ts_ + temp_space_off + x_size, state_size,
                            batch_size);
                    array_set(ts_ + temp_space_off + x_size + h_size, 1.0,
                            2 * batch_size);
                } else {
                    transpose<data_t>(x_ptr, ts_ + temp_space_off, batch_size,
                            input_size);
                    transpose<data_t>(hx_ptr, ts_ + temp_space_off + x_size,
                            batch_size, state_size);
                    array_set(ts_ + temp_space_off + x_size + h_size, 1.0,
                            2 * batch_size);
                }
                gemm<data_t>(NOTRANS, TRANS, ts_ + temp_space_off,
                        ts_ + dgates_space_off + rl * gates_size
                                + rt * gates_nlayer_size,
                        dweights_ptr + w_off, in_size + state_size + 2,
                        4 * state_size, batch_size, 1);

                gemm<data_t>(NOTRANS, NOTRANS, weights_ptr + w_off,
                        ts_ + dgates_space_off + rl * gates_size
                                + rt * gates_nlayer_size,
                        ts_ + temp_space_off, in_size + state_size + 2,
                        batch_size, 4 * state_size, 0);
                if ((rt > 0) && (rl > 0)) {
                    axpycopy<data_t>(ts_ + temp_space_off, ts_ + dhout_space_off
                                    + (rl - 1) * h_size + rt * h_nlayer_size,
                            state_size, batch_size);
                    axpycopy<data_t>(
                            ts_ + temp_space_off + in_size * batch_size,
                            ts_ + dhout_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            state_size, batch_size);
                } else if ((rl == 0) && (rt > 0)) {
                    transpose<data_t>(ts_ + temp_space_off,
                            dx_ptr + rt * x_size, input_size, batch_size);
                    axpycopy<data_t>(
                            ts_ + temp_space_off + in_size * batch_size,
                            ts_ + dhout_space_off + (rt - 1) * h_nlayer_size,
                            state_size, batch_size);
                } else if ((rt == 0) && (rl > 0)) {
                    axpycopy<data_t>(ts_ + temp_space_off, ts_ + dhout_space_off
                                    + (rl - 1) * h_size + rt * h_nlayer_size,
                            state_size, batch_size);
                    transpose<data_t>(
                            ts_ + temp_space_off + in_size * batch_size,
                            dhx_ptr + l * h_size, state_size, batch_size);
                } else {
                    transpose<data_t>(ts_ + temp_space_off, dx_ptr, input_size,
                            batch_size);
                    transpose<data_t>(
                            ts_ + temp_space_off + in_size * batch_size,
                            dhx_ptr, state_size, batch_size);
                }
            }
        }
    }
}

template <typename data_t>
class lstm_backward_test : public ::testing::TestWithParam<lstm_test_params> {
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
    virtual void SetUp() {
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
        const size_t total_w = ld.num_layers == 1 ? dir * w1_size : dir
                        * (w1_size + (ld.num_layers - 1) * wx_size);

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
        for (int i = 0; i < 5; i++) {
            Forward();
            Backward();
        }
    }

    void Forward() {
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
        if (with_workspace) {
            auto workspace_primitive_desc
                    = rnn_fwd_prim_desc->workspace_primitive_desc();
            workspace.reset(new memory(workspace_primitive_desc));
        }
        auto l = rnn_forward(*rnn_fwd_prim_desc, x.get(), hx.get(), cx.get(),
                weights.get(), y.get(), hy.get(), cy.get(), workspace.get());
        pipeline.push_back(l);
        s.submit(pipeline).wait();

        compute_ref_lstm_fwd<data_t>(p.test_ld, *x_desc, *hx_desc, *y_desc,
                *weights_desc, *x, *hx, *cx, *weights, *ref_y, *ref_hy,
                *ref_cy);
        if (p.test_ld.state_outputs) {
            compare_data<data_t>(*ref_y, *y);
            compare_data<data_t>(*ref_hy, *hy);
            compare_data<data_t>(*ref_cy, *cy);
        } else
            compare_data<data_t>(*ref_y, *y);
    }

    void Backward() {
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
        auto l = rnn_backward(*rnn_bwd_prim_desc, x.get(), hx.get(), cx.get(),
                dy.get(), dhy.get(), dcy.get(), weights.get(), workspace.get(),
                dx.get(), dhx.get(), dcx.get(), dweights.get());
        pipeline.push_back(l);
        s.submit(pipeline).wait();

        compute_ref_lstm_bwd<data_t>(p.test_ld, *x_desc, *hx_desc, *y_desc,
                *weights_desc, *x, *hx, *cx, *dy, *dhy, *dcy, *weights,
                *workspace, *ref_dx, *ref_dhx, *ref_dcx, *ref_dweights);

        compare_data<data_t>(*ref_dx, *dx);
        compare_data<data_t>(*ref_dhx, *dhx);
        compare_data<data_t>(*ref_dcx, *dcx);
        compare_data<data_t>(*ref_dweights, *dweights);
    }
};

using lstm_backward_test_float = lstm_backward_test<float>;
using lstm_test_params_float = lstm_test_params;

TEST_P(lstm_backward_test_float, TestsRNN) {}

INSTANTIATE_TEST_CASE_P(
        TestRNNBackward0, lstm_backward_test_float,
        ::testing::Values(
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 128, 128, 10, 4, 32, LSTM, UNIDIRECT, LINEAR, 1 } },
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_bidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 128, 128, 10, 4, 16, LSTM, BIDIRECT, LINEAR, 1 } }));

INSTANTIATE_TEST_CASE_P(
        TestRNNBackward1, lstm_backward_test_float,
        ::testing::Values(
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 128, 128, 10, 4, 32, LSTM, UNIDIRECT, LINEAR, 0 } },
                lstm_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_lstm,
                        direction::rnn_bidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 128, 128, 10, 4, 16, LSTM, BIDIRECT, LINEAR, 0 } }));
}
