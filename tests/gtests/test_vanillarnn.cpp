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

#include "mkldnn_test_common.hpp"
#include "rnn_common.hpp"
#include "gtest/gtest.h"

namespace mkldnn {
template <typename data_t>
void compute_ref_rnn_fwd(const test_rnn_desc_t &rd, const memory::desc &x_d,
        const memory::desc &hx_d, const memory::desc &y_d,
        const memory::desc &weights_d, const memory &x, const memory &hx,
        const memory &weights, const memory &y, const memory &hy)
{
    data_t *x_ptr = (data_t *)x.get_data_handle();
    data_t *hx_ptr = (data_t *)hx.get_data_handle();
    data_t *weights_ptr = (data_t *)weights.get_data_handle();
    data_t *y_ptr = (data_t *)y.get_data_handle();
    data_t *hy_ptr = (data_t *)hy.get_data_handle();

    const int state_size = rd.state_size;
    const int input_size = rd.input_size;
    const int seq_length = rd.seq_length;
    const int num_layers = rd.num_layers;
    const int batch_size = rd.batch_size;
    const int direction = rd.direction;
    const int alg_kind = rd.alg_kind;
    const int total_layers = num_layers * direction;
    const int w1_size = state_size * (state_size + input_size + 2);
    const int wx_size = state_size * (state_size + state_size + 2);
    const int h_size = batch_size * state_size;
    const int x_size = batch_size * input_size;
    const int h_nlayer_size = h_size * num_layers;
    const int gates_size = h_size;
    const int gates_nlayer_size = gates_size * num_layers;
    const int gates_space_size = gates_nlayer_size * seq_length * direction;
    const int hout_space_size = h_nlayer_size * seq_length * direction;

    const int ws_size = gates_space_size + hout_space_size;
    data_t *ws_ptr = new data_t[ws_size];

    int bsize = (input_size > state_size) ? input_size : state_size;
    int tmp = bsize + state_size + 2;
    int temp_size = tmp * batch_size;
    data_t *ts_ = new data_t[temp_size];

    const int gates_space_off = 0;
    const int hout_space_off = gates_space_size;
    int in_size = 0;
    int wa = w1_size + (num_layers - 1) * wx_size;
    int dl, rl, rt;

    data_t* reordered_w = 
        new data_t[((input_size > state_size) ? input_size : state_size +
            state_size + 2) * state_size];

    for (int l = 0; l < total_layers; l++) {
        dl = l / num_layers;
        rl = l % num_layers;
        in_size = (rl == 0) ? input_size : state_size;

        // Wx
        size_t offset = (rl == 0) ? 0 : 
            (input_size*state_size + (rl - 1) * (state_size*state_size)) +
            rl * (state_size*state_size);
        for (size_t ii = 0; ii < state_size; ii++) {
            for (size_t jj = 0; jj < in_size; jj++) {
                reordered_w[ii*(in_size + state_size + 2) + jj] = 
                    weights_ptr[offset + ii*in_size + jj];
            }
        }

        // Wh
        offset += in_size*state_size;
        for (size_t ii = 0; ii < state_size; ii++) {
            for (size_t jj = 0; jj < state_size; jj++) {
                reordered_w[ii*(in_size + state_size + 2) + in_size + jj] = 
                    weights_ptr[offset + ii*state_size + jj];
            }
        }

        // bx
        offset = (input_size+state_size)*state_size;
        if (num_layers > 1)
            offset += (num_layers - 1)*2*state_size*state_size + rl*2*state_size;
        for (size_t ii = 0; ii < state_size; ii++) {
            for (size_t jj = 0; jj < 2; jj++) {
                reordered_w[ii*(in_size + state_size + 2) + in_size + state_size + jj] = 
                    weights_ptr[offset + ii + jj*state_size];
            }
        }

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
                gemm<data_t>(NOTRANS, NOTRANS, reordered_w, ts_,
                        ws_ptr + gates_space_off + l * gates_size
                                + t * gates_nlayer_size,
                        state_size, batch_size, in_size + state_size + 2, 0);
                for (int h = 0; h < h_size; h++) {
                    data_t it = ws_ptr[gates_space_off + l * gates_size
                            + t * gates_nlayer_size + h];
                    if (alg_kind == rnn_relu) {
                        if (it > 0) {
                            ws_ptr[hout_space_off + l * h_size
                                    + t * h_nlayer_size + h]
                                    = it;
                        } else {
                            ws_ptr[hout_space_off + l * h_size
                                    + t * h_nlayer_size + h]
                                    = 0;
                        }
                    } else if (alg_kind == rnn_tanh) {
                        ws_ptr[hout_space_off + l * h_size + t * h_nlayer_size
                                + h]
                                = tanh(it);
                    }
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
                gemm<data_t>(NOTRANS, NOTRANS, reordered_w, ts_,
                        ws_ptr + gates_space_off + rl * gates_size
                                + rt * gates_nlayer_size,
                        state_size, batch_size, in_size + state_size + 2, 0);
                for (int h = 0; h < h_size; h++) {
                    data_t it = ws_ptr[gates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + h];
                    if (alg_kind == rnn_relu) {
                        if (it > 0) {
                            ws_ptr[hout_space_off + rl * h_size
                                    + rt * h_nlayer_size + h]
                                    = it;
                        } else {
                            ws_ptr[hout_space_off + rl * h_size
                                    + rt * h_nlayer_size + h]
                                    = 0;
                        }
                    } else if (alg_kind == rnn_tanh) {
                        ws_ptr[hout_space_off + rl * h_size + rt * h_nlayer_size
                                + h]
                                = tanh(it);
                    }
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
                }
            }
        }
    }

    delete[] reordered_w;
}

template <typename data_t>
void compute_ref_rnn_bwd(const test_rnn_desc_t &rd, const memory::desc &x_d,
        const memory::desc &hx_d, const memory::desc &y_d,
        const memory::desc &weights_d, const memory &x, const memory &hx,
        const memory &dy, const memory &dhy, const memory &weights,
        const memory &ws, const memory &dx, const memory &dhx,
        const memory &dweights)
{
    data_t *x_ptr = (data_t *)x.get_data_handle();
    data_t *hx_ptr = (data_t *)hx.get_data_handle();
    data_t *dy_ptr = (data_t *)dy.get_data_handle();
    data_t *dhy_ptr = (data_t *)dhy.get_data_handle();
    data_t *weights_ptr = (data_t *)weights.get_data_handle();
    data_t *ws_ptr = (data_t *)ws.get_data_handle();
    data_t *dx_ptr = (data_t *)dx.get_data_handle();
    data_t *dhx_ptr = (data_t *)dhx.get_data_handle();
    data_t *dweights_ptr = (data_t *)dweights.get_data_handle();

    const int state_size = rd.state_size;
    const int input_size = rd.input_size;
    const int seq_length = rd.seq_length;
    const int num_layers = rd.num_layers;
    const int batch_size = rd.batch_size;
    const int direction = rd.direction;
    const int alg_kind = rd.alg_kind;
    const int total_layers = num_layers * direction;
    const int w1_size = state_size * (state_size + input_size + 2);
    const int wx_size = state_size * (state_size + state_size + 2);
    const int h_size = batch_size * state_size;
    const int x_size = batch_size * input_size;
    const int h_nlayer_size = h_size * num_layers;
    const int gates_size = h_size;
    const int gates_nlayer_size = gates_size * num_layers;
    const int gates_space_size = gates_nlayer_size * seq_length * direction;
    const int hout_space_size = h_nlayer_size * seq_length * direction;

    int bsize = (input_size > state_size) ? input_size : state_size;
    int tmp = bsize + state_size + 2;
    int temp_size = tmp * batch_size + gates_space_size + hout_space_size;
    data_t *ts_ = new data_t[temp_size];
    memset(ts_, 0, temp_size * sizeof(data_t));

    const int gates_space_off = 0;
    const int hout_space_off = gates_space_size;

    const int dgates_space_off = 0;
    const int dhout_space_off = dgates_space_off + gates_space_size;
    const int temp_space_off = dhout_space_off + hout_space_size;
    int w_off = 0;
    int in_size = 0;
    int wa = w1_size + (num_layers - 1) * wx_size;
    int dl, rl, rt;

    data_t* reordered_w = 
        new data_t[((input_size > state_size) ? input_size : state_size +
            state_size + 2) * state_size];

#pragma omp parallel for
    for (int seq = 0; seq < seq_length; seq++) {
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
    if (rd.state_outputs) {
#pragma omp parallel for
        for (int ly = 0; ly < num_layers; ly++) {
            transpose<data_t>(dhy_ptr + ly * h_size,
                    ts_ + temp_space_off + ly * h_size, batch_size, state_size);
            axpycopy<data_t>(ts_ + temp_space_off + ly * h_size,
                    ts_ + dhout_space_off
                            + (seq_length * direction - 1) * h_nlayer_size
                            + ly * h_size,
                    batch_size, state_size);
        }
    }

    for (int l = (total_layers - 1); l >= 0; l--) {
        dl = l / num_layers;
        rl = l % num_layers;
        in_size = (rl == 0) ? input_size : state_size;

        // Wx
        size_t offset = (rl == 0) ? 0 : 
            (input_size*state_size + (rl - 1) * (state_size*state_size)) +
            rl * (state_size*state_size);
        for (size_t ii = 0; ii < state_size; ii++) {
            for (size_t jj = 0; jj < in_size; jj++) {
                reordered_w[ii*(in_size + state_size + 2) + jj] = 
                    weights_ptr[offset + ii*in_size + jj];
            }
        }

        // Wh
        offset += in_size*state_size;
        for (size_t ii = 0; ii < state_size; ii++) {
            for (size_t jj = 0; jj < state_size; jj++) {
                reordered_w[ii*(in_size + state_size + 2) + in_size + jj] = 
                    weights_ptr[offset + ii*state_size + jj];
            }
        }

        // bx
        offset = (input_size+state_size)*state_size;
        if (num_layers > 1)
            offset += (num_layers - 1)*2*state_size*state_size + rl*2*state_size;
        for (size_t ii = 0; ii < state_size; ii++) {
            for (size_t jj = 0; jj < 2; jj++) {
                reordered_w[ii*(in_size + state_size + 2) + in_size + state_size + jj] = 
                    weights_ptr[offset + ii + jj*state_size];
            }
        }

        if (dl == 1) {
            for (int t = 0; t < seq_length; t++) {
                rt = 2 * seq_length - t - 1;
                data_t it, dht;
                for (int h = 0; h < h_size; h++) {
                    dht = ts_[dhout_space_off + rl * h_size + rt * h_nlayer_size
                            + h];
                    it = ws_ptr[gates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + h];
                    if (alg_kind == rnn_relu) {
                        if (it > 0) {
                            ts_[dgates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size + h]
                                    = dht;
                        } else {
                            ts_[dgates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size + h]
                                    = 0;
                        }
                    } else if (alg_kind == rnn_tanh) {
                        ts_[dgates_space_off + rl * gates_size
                                + rt * gates_nlayer_size + h]
                                = (1.0 - pow(it, 2)) * dht;
                    }
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
                        state_size, batch_size, 1);

                gemm<data_t>(TRANS, NOTRANS, reordered_w,
                        ts_ + dgates_space_off + rl * gates_size
                                + rt * gates_nlayer_size,
                        ts_ + temp_space_off, in_size + state_size + 2,
                        batch_size, state_size, 0);
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
                data_t it, dht;
                for (int h = 0; h < h_size; h++) {
                    it = ws_ptr[gates_space_off + rl * gates_size
                            + rt * gates_nlayer_size + h];
                    dht = ts_[dhout_space_off + rl * h_size + rt * h_nlayer_size
                            + h];
                    if (alg_kind == rnn_relu) {
                        if (it > 0) {
                            ts_[dgates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size + h]
                                    = dht;
                        } else {
                            ts_[dgates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size + h]
                                    = 0;
                        }
                    } else if (alg_kind == rnn_tanh) {
                        ts_[dgates_space_off + rl * gates_size
                                + rt * gates_nlayer_size + h]
                                = (1.0 - pow(it, 2)) * dht;
                    }
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
                        state_size, batch_size, 1);

                gemm<data_t>(TRANS, NOTRANS, reordered_w,
                        ts_ + dgates_space_off + rl * gates_size
                                + rt * gates_nlayer_size,
                        ts_ + temp_space_off, in_size + state_size + 2,
                        batch_size, state_size, 0);

                if ((rt > 0) && (rl > 0)) {
                    axpycopy<data_t>(ts_ + temp_space_off, ts_ + dhout_space_off
                                    + (rl - 1) * h_size + rt * h_nlayer_size,
                            state_size, batch_size);
                    axpycopy<data_t>(ts_ + temp_space_off + h_size,
                            ts_ + dhout_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            state_size, batch_size);
                } else if ((rl == 0) && (rt > 0)) {
                    transpose<data_t>(ts_ + temp_space_off,
                            dx_ptr + rt * x_size, input_size, batch_size);
                    axpycopy<data_t>(ts_ + temp_space_off + x_size,
                            ts_ + dhout_space_off + (rt - 1) * h_nlayer_size,
                            state_size, batch_size);
                } else if ((rt == 0) && (rl > 0)) {
                    axpycopy<data_t>(ts_ + temp_space_off, ts_ + dhout_space_off
                                    + (rl - 1) * h_size + rt * h_nlayer_size,
                            state_size, batch_size);
                    transpose<data_t>(ts_ + temp_space_off + h_size,
                            dhx_ptr + l * h_size, state_size, batch_size);
                } else {
                    transpose<data_t>(ts_ + temp_space_off, dx_ptr, input_size,
                            batch_size);
                    transpose<data_t>(ts_ + temp_space_off + x_size, dhx_ptr,
                            state_size, batch_size);
                }
            }
        }
    }

    delete[] reordered_w;
}

template <typename data_t>
class rnn_backward_test : public ::testing::TestWithParam<rnn_test_params> {
private:
    std::shared_ptr<memory> x;
    std::shared_ptr<memory> hx;
    std::shared_ptr<memory> dx;
    std::shared_ptr<memory> dhx;
    std::shared_ptr<memory> y;
    std::shared_ptr<memory> hy;
    std::shared_ptr<memory> dy;
    std::shared_ptr<memory> dhy;
    std::shared_ptr<memory> weights;
    std::shared_ptr<memory> dweights;
    std::shared_ptr<memory> workspace;
    std::shared_ptr<memory> ref_y;
    std::shared_ptr<memory> ref_hy;
    std::shared_ptr<memory> ref_dx;
    std::shared_ptr<memory> ref_dhx;
    std::shared_ptr<memory> ref_weights;
    std::shared_ptr<memory> ref_dweights;
    std::shared_ptr<memory::desc> x_desc;
    std::shared_ptr<memory::desc> hx_desc;
    std::shared_ptr<memory::desc> y_desc;
    std::shared_ptr<memory::desc> weights_desc;
    std::shared_ptr<rnn_forward::primitive_desc> rnn_fwd_prim_desc;
    std::shared_ptr<rnn_backward::primitive_desc> rnn_bwd_prim_desc;
    rnn_test_params p;
    std::shared_ptr<engine> eng;
    memory::data_type data_type;
    bool with_workspace;

protected:
    virtual void SetUp()
    {
        // using namespace mkldnn::impl::utils;
        p = ::testing::TestWithParam<rnn_test_params>::GetParam();
        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_TRUE(p.aalgorithm == algorithm::rnn_relu
                || p.aalgorithm == algorithm::rnn_tanh);
        ASSERT_TRUE(p.adirection == direction::rnn_unidirectional
                || p.adirection == direction::rnn_bidirectional);
        ASSERT_TRUE(p.ainput_mode == input_mode::rnn_linear_input);
        eng.reset(new engine(p.engine_kind, 0));
        data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);
        test_rnn_desc_t rd = p.test_rd;
        with_workspace = p.aprop_kind == prop_kind::forward_training;
        int dir = (p.adirection == direction::rnn_unidirectional) ? 1 : 2;

        const int w1_size
                = rd.state_size * (rd.state_size + rd.input_size + 2);
        const int wx_size
                = rd.state_size * (rd.state_size + rd.state_size + 2);
        const int total_w = rd.num_layers == 1 ? dir * w1_size : dir
                        * (w1_size + (rd.num_layers - 1) * wx_size);

        x_desc.reset(new memory::desc({ static_cast<int>(rd.seq_length),
                                              static_cast<int>(rd.batch_size),
                                              static_cast<int>(rd.input_size) },
                data_type, p.rnx_format));
        hx_desc.reset(
                new memory::desc({ static_cast<int>(rd.num_layers),
                                         static_cast<int>(rd.batch_size),
                                         static_cast<int>(rd.state_size) },
                        data_type, p.rnx_format));
        y_desc.reset(new memory::desc(
                { static_cast<int>(rd.seq_length),
                        static_cast<int>(rd.batch_size),
                        static_cast<int>(rd.state_size * dir) },
                data_type, p.rnx_format));
        weights_desc.reset(new memory::desc(
                { static_cast<int>(total_w) }, data_type, memory::format::x));
        x.reset(new memory({ *x_desc, *eng }));
        hx.reset(new memory({ *hx_desc, *eng }));
        dx.reset(new memory({ *x_desc, *eng }));
        dhx.reset(new memory({ *hx_desc, *eng }));
        y.reset(new memory({ *y_desc, *eng }));
        hy.reset(new memory({ *hx_desc, *eng }));
        dy.reset(new memory({ *y_desc, *eng }));
        dhy.reset(new memory({ *hx_desc, *eng }));
        weights.reset(new memory({ *weights_desc, *eng }));
        dweights.reset(new memory({ *weights_desc, *eng }));

        ref_y.reset(new memory({ *y_desc, *eng }));
        ref_hy.reset(new memory({ *hx_desc, *eng }));
        ref_dx.reset(new memory({ *x_desc, *eng }));
        ref_dhx.reset(new memory({ *hx_desc, *eng }));
        ref_dweights.reset(new memory({ *weights_desc, *eng }));

        Forward();
        Backward();
    }

    void Forward()
    {
        auto rnn_fwd_desc = rnn_forward::desc(p.aprop_kind, p.aalgorithm,
                p.adirection, p.ainput_mode, p.test_rd.state_size,
                p.test_rd.num_layers, p.test_rd.seq_length,
                p.test_rd.state_outputs, *x_desc, *hx_desc, *y_desc,
                *weights_desc);
        rnn_fwd_prim_desc.reset(
                new rnn_forward::primitive_desc(rnn_fwd_desc, *eng));
        fill_data<data_t>(x->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)x->get_data_handle());
        fill_data<data_t>(hx->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)hx->get_data_handle());
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

        if (p.test_rd.state_outputs) {
            auto l = rnn_forward(*rnn_fwd_prim_desc, *x, *hx, *weights, *y,
                    *hy, *workspace);
            pipeline.push_back(l);
            s.submit(pipeline).wait();
        } else {
            auto l = rnn_forward(*rnn_fwd_prim_desc, *x, *hx, *weights, *y,
                    *workspace);
            pipeline.push_back(l);
            s.submit(pipeline).wait();
        }

        compute_ref_rnn_fwd<data_t>(p.test_rd, *x_desc, *hx_desc, *y_desc,
                *weights_desc, *x, *hx, *weights, *ref_y, *ref_hy);
        if (p.test_rd.state_outputs) {
            compare_data_woinfnan<data_t>(*ref_y, *y);
            compare_data_woinfnan<data_t>(*ref_hy, *hy);
        } else
            compare_data_woinfnan<data_t>(*ref_y, *y);
    }

    void Backward()
    {
        auto pk = prop_kind::backward;
        auto rnn_bwd_desc = rnn_backward::desc(pk, p.aalgorithm, p.adirection,
                p.ainput_mode, p.test_rd.state_size, p.test_rd.num_layers,
                p.test_rd.seq_length, p.test_rd.state_outputs, *x_desc,
                *hx_desc, *y_desc, *weights_desc);
        rnn_bwd_prim_desc.reset(new rnn_backward::primitive_desc(
                rnn_bwd_desc, *eng, *rnn_fwd_prim_desc));
        fill_data<data_t>(dy->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)dy->get_data_handle());
        fill_data<data_t>(dhy->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)dhy->get_data_handle());
        fill_data<data_t>(
                dweights->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)dweights->get_data_handle());
        fill_data<data_t>(
                ref_dweights->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)ref_dweights->get_data_handle());
        // Execute
        std::vector<primitive> pipeline;
        auto s = stream(stream::kind::lazy);
        if (p.test_rd.state_outputs) {
            auto l = rnn_backward(*rnn_bwd_prim_desc, *x, *hx, *dy, *dhy,
                    *weights, *workspace, *dx, *dhx, *dweights);
            pipeline.push_back(l);
            s.submit(pipeline).wait();
        } else {
            auto l = rnn_backward(*rnn_bwd_prim_desc, *x, *hx, *dy, *weights,
                    *workspace, *dx, *dhx, *dweights);
            pipeline.push_back(l);
            s.submit(pipeline).wait();
        }
        compute_ref_rnn_bwd<data_t>(p.test_rd, *x_desc, *hx_desc, *y_desc,
                *weights_desc, *x, *hx, *dy, *dhy, *weights, *workspace,
                *ref_dx, *ref_dhx, *ref_dweights);
    }
};

using rnn_backward_test_float = rnn_backward_test<float>;
using rnn_test_params_float = rnn_test_params;

TEST_P(rnn_backward_test_float, TestsRNN)
{
}

INSTANTIATE_TEST_CASE_P(
        TestRNNBackward0, rnn_backward_test_float,
        ::testing::Values(
                rnn_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_relu,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 4, 4, 2, 2, 2, RELU, UNIDIRECT, LINEAR, 1 } },
                rnn_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_relu,
                        direction::rnn_bidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 4, 4, 2, 2, 2, RELU, BIDIRECT, LINEAR, 1 } }));

INSTANTIATE_TEST_CASE_P(
        TestRNNBackward1, rnn_backward_test_float,
        ::testing::Values(
                rnn_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_relu,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 4, 4, 2, 2, 2, RELU, UNIDIRECT, LINEAR, 0 } },
                rnn_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_relu,
                        direction::rnn_bidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 4, 4, 1, 2, 2, RELU, BIDIRECT, LINEAR, 0 } }));
INSTANTIATE_TEST_CASE_P(
        TestRNNBackward2, rnn_backward_test_float,
        ::testing::Values(
                rnn_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_tanh,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 4, 4, 2, 2, 2, TANH, UNIDIRECT, LINEAR, 1 } },
                rnn_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_tanh,
                        direction::rnn_bidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 4, 4, 2, 2, 2, TANH, BIDIRECT, LINEAR, 1 } }));

INSTANTIATE_TEST_CASE_P(
        TestRNNBackward3, rnn_backward_test_float,
        ::testing::Values(
                rnn_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_tanh,
                        direction::rnn_unidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 4, 4, 2, 2, 2, TANH, UNIDIRECT, LINEAR, 0 } },
                rnn_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, algorithm::rnn_tanh,
                        direction::rnn_bidirectional,
                        input_mode::rnn_linear_input, memory::format::rnx,
                        { 4, 4, 1, 2, 2, TANH, BIDIRECT, LINEAR, 0 } }));
}
