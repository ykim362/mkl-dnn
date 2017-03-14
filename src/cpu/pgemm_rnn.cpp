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

#include <iostream>
#include "c_types_map.hpp"
#include "type_helpers.hpp"

#include "pgemm_rnn.hpp"
#include "cpu_math_util.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

#define L1_BLK_SIZE 8096
enum {
    NOTRANS = 1,
    TRANS = 2
};

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::alg_kind;
using namespace mkldnn::impl::data_type;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::primitive_kind;
using namespace mkldnn::impl::utils;
#ifdef USE_MKL
using namespace mkldnn::impl::cpu::cpu_blas;
using namespace mkldnn::impl::cpu::cpu_trans;
using namespace mkldnn::impl::cpu::cpu_vml;
#endif

template <typename data_t>
inline void vsigmoid(data_t *X, data_t *tmp1, size_t Len) {
#ifdef USE_MKL
    cblas_scal<data_trait<data_t>::data_type>(Len, -1.0, X, 1);
    vExp<data_trait<data_t>::data_type>(Len, X, X);
    vAdd<data_trait<data_t>::data_type>(Len, tmp1, X, X);
    vDiv<data_trait<data_t>::data_type>(Len, tmp1, X, X);
#endif
}

template <typename data_t>
inline void lstm_fwd_ele_wise(data_t *Gates, const data_t *Ct_1, data_t *Ct,
                            data_t *Ht, data_t *tmp1, data_t *tmp,
                            size_t Length) {
#ifdef USE_MKL
    size_t iter = Length / L1_BLK_SIZE;
    size_t rem = Length % L1_BLK_SIZE;
    size_t compute_block = 0;
    if (rem != 0) iter += 1;
    #pragma omp parallel for
    for(size_t i=0; i < iter; i++) {
        if(iter == 0 || (rem != 0 && i == (iter - 1))) 
            compute_block = rem;
        else
            compute_block = L1_BLK_SIZE;
        vsigmoid<data_t>(Gates, tmp1, compute_block);
        vsigmoid<data_t>(Gates + Length, tmp1, compute_block);
        vsigmoid<data_t>(Gates + 2 * Length, tmp1, compute_block);
        vTanh<data_trait<data_t>::data_type>(compute_block, Gates + 3 * Length,
                                       Gates + 3 * Length);
        vMul<data_trait<data_t>::data_type>(compute_block, Gates + Length, Ct_1, Ct);
        vMul<data_trait<data_t>::data_type>(compute_block, Gates, Gates + 3 * Length, tmp);
        vAdd<data_trait<data_t>::data_type>(compute_block, Ct, tmp, Ct);
        vTanh<data_trait<data_t>::data_type>(compute_block, Ct, tmp);
        vMul<data_trait<data_t>::data_type>(compute_block, Gates + 2 * Length, tmp, Ht);
    }
#endif
}

template <typename data_t>
inline void
lstm_fwd_prop_single(const size_t input_size, const size_t state_size,
                    const size_t batch_size, const data_t *x, int tranx,
                    const data_t *ht_1, int tranht_1, const data_t *ct_1,
                    int tranct_1, const data_t *w, data_t *ht, data_t *ct,
                    data_t *gates, data_t *tmp1, data_t *tmp) {
#ifdef USE_MKL
    auto x_size = input_size * batch_size;
    auto h_size = state_size * batch_size;
    if (tranx == TRANS) {
        omatcopy<data_trait<data_t>::data_type>(
            'R', 'T', batch_size, input_size, 1.0, x, input_size, tmp, batch_size);
    } else {
        cblas_copy<data_trait<data_t>::data_type>(x_size, x, 1, tmp, 1);
    }
    if (tranht_1 == TRANS) {
        omatcopy<data_trait<data_t>::data_type>('R', 'T', batch_size, state_size,
                                                1.0, ht_1, state_size, tmp + x_size,
                                                batch_size);
    } else {
        cblas_copy<data_trait<data_t>::data_type>(h_size, ht_1, 1, tmp + x_size, 1);
    }
    array_set(tmp + x_size + h_size, 1.0, 2 * batch_size);
    cblas_gemm_compute<data_trait<data_t>::data_type>(
        CblasRowMajor, CblasPacked, CblasNoTrans, 4 * state_size, batch_size,
        input_size + state_size + 2, w, 4 * state_size, tmp, batch_size, 0.0,
        gates, batch_size);
    if (tranct_1 == TRANS) {
        omatcopy<data_trait<data_t>::data_type>('R', 'T', batch_size, state_size,
                                                1.0, ct_1, state_size, tmp,
                                                batch_size);
        lstm_fwd_ele_wise<data_t>(gates, tmp, ct, ht, tmp1, tmp + h_size, h_size);
    } else {
        lstm_fwd_ele_wise<data_t>(gates, ct_1, ct, ht, tmp1, tmp, h_size);
    }
#endif
}

template <typename data_t>
inline void lstm_bwd_ele_wise(const data_t *Gates, data_t *dGates,
                            const data_t *Ct_1, data_t *dCt_1,
                            const data_t *Ct, data_t *dCt, const data_t *dHt,
                            data_t *tmp1, data_t *tmp, size_t Length) {
#ifdef USE_MKL
    // formula: do[t] = dh[t] * tanh(c[t])
    vTanh<data_trait<data_t>::data_type>(Length, Ct, tmp);
    vMul<data_trait<data_t>::data_type>(Length, tmp, dHt, dGates + 2 * Length);

    // formula: dc[t] += (1-tanh(c[t] **2) * dh[t] * o[t]
    vSqr<data_trait<data_t>::data_type>(Length, tmp, tmp);
    vSub<data_trait<data_t>::data_type>(Length, tmp1, tmp, tmp);
    vMul<data_trait<data_t>::data_type>(Length, tmp, Gates + 2 * Length, tmp);
    vMul<data_trait<data_t>::data_type>(Length, tmp, dHt, tmp);
    vAdd<data_trait<data_t>::data_type>(Length, tmp, dCt, dCt);

    // formula: df[t] = dc[t] * c[t-1]
    // formula: dc[t-1] = dc[t] * f[t]
    vMul<data_trait<data_t>::data_type>(Length, dCt, Ct_1, dGates + Length);
    vMul<data_trait<data_t>::data_type>(Length, dCt, Gates + Length, dCt_1);

    // di[t] = dc[t] * g[t]
    vMul<data_trait<data_t>::data_type>(Length, dCt, Gates + 3 * Length, dGates);

    // dg[t] = dc[t] * i[t]
    vMul<data_trait<data_t>::data_type>(Length, dCt, Gates, dGates + 3 * Length);

    // dg'[t] = dg[t] * (1 - g[t]**2)
    vSqr<data_trait<data_t>::data_type>(Length, Gates + 3 * Length, tmp);
    vSub<data_trait<data_t>::data_type>(Length, tmp1, tmp, tmp);
    vMul<data_trait<data_t>::data_type>(Length, tmp, dGates + 3 * Length,
                                      dGates + 3 * Length);

    // di'[t] = di[t] * i[t] * (1-i[t])
    // df'[t] = df[t] * f[t] * (1-f[t])
    // do'[t] = do[t] * o[t] * (1-o[t])
    vSub<data_trait<data_t>::data_type>(Length * 3, tmp1, Gates, tmp);
    vMul<data_trait<data_t>::data_type>(Length * 3, Gates, tmp, tmp);
    vMul<data_trait<data_t>::data_type>(Length * 3, dGates, tmp, dGates);
#endif
}

template <typename data_t>
inline void lstm_bwd_prop_single(
        const size_t input_size, const size_t state_size, const size_t batch_size,
        const data_t *x, int tranx, const data_t *ht_1, int tranht_1,
        const data_t *ct_1, int tranct_1, const data_t *ct, const data_t *w,
        const data_t *gates, data_t *dht, data_t *dct, data_t *dw, data_t *dx,
        data_t *dht_1, data_t *dct_1, data_t *dgates, data_t *tmp1, data_t *tmp) {
#ifdef USE_MKL
    auto x_size = input_size * batch_size;
    auto h_size = state_size * batch_size;
    if (tranct_1 == TRANS) {
        omatcopy<data_trait<data_t>::data_type>('R', 'T', batch_size, state_size,
                                                1.0, ct_1, state_size, tmp,
                                                batch_size);
        lstm_bwd_ele_wise<data_t>(gates, dgates, tmp, tmp + h_size, ct, dct, dht,
                                                tmp1, tmp + 2 * h_size, h_size);
        omatcopy<data_trait<data_t>::data_type>('R', 'T', state_size, batch_size,
                                                1.0, tmp + h_size, batch_size,
                                                dct_1, state_size);
    } else {
        lstm_bwd_ele_wise<data_t>(gates, dgates, ct_1, dct_1, ct, dct, dht, tmp1,
                                  tmp, h_size);
    }
    if (tranx == TRANS) {
        omatcopy<data_trait<data_t>::data_type>(
            'R', 'T', batch_size, input_size, 1.0, x, input_size, tmp, batch_size);
    } else {
        cblas_copy<data_trait<data_t>::data_type>(x_size, x, 1, tmp, 1);
    }
    if (tranht_1 == TRANS) {
        omatcopy<data_trait<data_t>::data_type>('R', 'T', batch_size, state_size,
                                                1.0, ht_1, state_size, tmp + x_size,
                                                batch_size);
    } else {
        cblas_copy<data_trait<data_t>::data_type>(h_size, ht_1, 1, tmp + x_size, 1);
    }
    array_set(tmp + x_size + h_size, 1.0, 2 * batch_size);
    cblas_gemm<data_trait<data_t>::data_type>(
        CblasRowMajor, CblasNoTrans, CblasTrans, (input_size + state_size + 2),
        4 * state_size, batch_size, 1.0, tmp, batch_size, dgates, batch_size, 1.0,
        dw, 4 * state_size);
    cblas_gemm_compute<data_trait<data_t>::data_type>(
        CblasRowMajor, CblasPacked, CblasNoTrans, (input_size + state_size + 2),
        batch_size, 4 * state_size, w, 4 * state_size, dgates, batch_size, 0.0,
        tmp, batch_size);
    if (tranx == TRANS) {
        omatcopy<data_trait<data_t>::data_type>(
            'R', 'T', input_size, batch_size, 1.0, tmp, batch_size, dx, input_size);
    } else {
        cblas_copy<data_trait<data_t>::data_type>(x_size, tmp, 1, dx, 1);
    }
    if (tranht_1 == TRANS) {
        omatcopy<data_trait<data_t>::data_type>('R', 'T', state_size, batch_size,
                                                1.0, tmp + x_size, batch_size,
                                                dht_1, state_size);
    } else {
        cblas_copy<data_trait<data_t>::data_type>(h_size, tmp + x_size, 1, dht_1,
                                              1);
    }
#endif
}

template <impl::data_type_t data_type>
void pgemm_rnn_fwd_t<data_type>::execute_forward() {
#ifdef USE_MKL
    auto x = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto hx = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto cx = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto w = reinterpret_cast<const data_t *>(this->input_memory(3));
    const int state_outputs = conf_.state_outputs();
    data_t *y = nullptr;
    data_t *hy = nullptr;
    data_t *cy = nullptr;
    data_t *ws = nullptr;
    if (state_outputs) {
        y = reinterpret_cast<data_t *>(this->memory(0));
        hy = reinterpret_cast<data_t *>(this->memory(1));
        cy = reinterpret_cast<data_t *>(this->memory(2));
        ws = reinterpret_cast<data_t *>(this->memory(3));
    } else {
        y = reinterpret_cast<data_t *>(this->memory(0));
        ws = reinterpret_cast<data_t *>(this->memory(1));
    }
    const size_t seq_length = conf_.tau();
    const size_t num_layers = conf_.layers();
    const size_t batch_size = conf_.batch();
    const size_t input_size = conf_.input_size();
    const size_t state_size = conf_.hidden_size();
    const size_t direction = conf_.direction();
    const size_t total_layers = num_layers * direction;
    const size_t w1_size = conf_.w1_size();
    const size_t wx_size = conf_.wx_size();
    const size_t h_size = conf_.h_size();
    const size_t x_size = conf_.x_size();
    const size_t h_nlayer_size = conf_.h_nlayer_size();
    const size_t gates_size = conf_.gates_size();
    const size_t gates_nlayer_size = conf_.gates_nlayer_size();
    const size_t gates_space_size = conf_.gates_space_size();
    const size_t hout_space_size = conf_.hout_space_size();
    const size_t c_space_size = conf_.c_space_size();
    const memory_desc_wrapper ws_d(conf_.workspace_pd());
    const size_t gates_space_off = 0;
    const size_t hout_space_off = gates_space_off + gates_space_size;
    const size_t c_space_off = hout_space_off + hout_space_size;
    size_t tmp_space_off = 0;
    size_t w_off = 0;
    size_t in_size = 0;
    size_t tmp1_off = 0;
    size_t wa = w1_size + (num_layers - 1) * wx_size;
    size_t dl, rl, roff, rt;
    data_t *ws_ptr;
    if (ws) {
        ws_ptr = ws;
        tmp1_off = 0;
        tmp_space_off = 3 * h_size;
    } else {
        ws_ptr = ts_;
        tmp1_off = c_space_off + c_space_size;
        tmp_space_off = c_space_off + c_space_size + 3 * h_size;
    }
    auto bsize = (input_size > state_size) ? input_size : state_size;
    auto tmp1 = bsize + state_size + 2;
    auto tmp2 = state_size * 4;
    auto tmp = (tmp1 > tmp2) ? tmp1 : tmp2;
    auto ts_size_ = tmp * batch_size + 3 * h_size;
    if (conf_.desc()->prop_kind != forward_training) {
        ts_size_ += conf_.workspace_size();
    }
    memset(ts_, 0, ts_size_ * sizeof(data_t));
    array_set(ts_ + tmp1_off, 1.0, 3 * h_size);
    data_t **weights_pack = new data_t *[total_layers];

    for (int l = 0; l < total_layers; l++) {
        dl = l / num_layers;
        rl = l % num_layers;
        roff = (rl == 0) ? 0 : (w1_size + (rl - 1) * wx_size);
        w_off = wa * dl + roff;
        in_size = (rl == 0) ? input_size : state_size;
        // pack weights
        weights_pack[l] = cblas_gemm_alloc<data_type>(
            CblasAMatrix, 4 * state_size, batch_size, (in_size + state_size + 2));
        cblas_gemm_pack<data_type>(CblasRowMajor, CblasAMatrix, CblasTrans,
                                    4 * state_size, batch_size,
                                    (in_size + state_size + 2), 1.0, w + w_off,
                                    4 * state_size, weights_pack[l]);
    }
    for (int l = 0; l < total_layers; l++) {
        dl = l / num_layers;
        rl = l % num_layers;
        roff = (rl == 0) ? 0 : (w1_size + (rl - 1) * wx_size);
        w_off = wa * dl + roff;
        if (dl == 0) {
            for (int t = 0; t < seq_length; t++) {
                rt = t;
                if (rt == 0 && rl == 0) {
                    lstm_fwd_prop_single<data_t>(
                        input_size, state_size, batch_size, x, TRANS, hx, TRANS, cx,
                        TRANS, weights_pack[l],
                        ws_ptr + hout_space_off + rl * h_size + rt * h_nlayer_size,
                        ws_ptr + c_space_off + rl * h_size + rt * h_nlayer_size,
                        ws_ptr + gates_space_off + rl * gates_size +
                          rt * gates_nlayer_size,
                        ts_ + tmp1_off, ts_ + tmp_space_off);
                } else if (rt == 0 && rl > 0) {
                    lstm_fwd_prop_single<data_t>(
                        state_size, state_size, batch_size,
                        ws_ptr + hout_space_off + (rl - 1) * h_size, NOTRANS,
                        hx + l * h_size, TRANS, cx + l * h_size, TRANS, weights_pack[l],
                        ws_ptr + hout_space_off + rl * h_size + rt * h_nlayer_size,
                        ws_ptr + c_space_off + rl * h_size + rt * h_nlayer_size,
                        ws_ptr + gates_space_off + rl * gates_size +
                            rt * gates_nlayer_size,
                        ts_ + tmp1_off, ts_ + tmp_space_off);
                } else if (rt > 0 && rl == 0) {
                    lstm_fwd_prop_single<data_t>(
                        state_size, state_size, batch_size, x + t * x_size, TRANS,
                        ws_ptr + hout_space_off + (rt - 1) * h_nlayer_size, NOTRANS,
                        ws_ptr + c_space_off + (rt - 1) * h_nlayer_size, NOTRANS,
                        weights_pack[l],
                        ws_ptr + hout_space_off + rl * h_size + rt * h_nlayer_size,
                        ws_ptr + c_space_off + rl * h_size + rt * h_nlayer_size,
                        ws_ptr + gates_space_off + rl * gates_size +
                            rt * gates_nlayer_size,
                        ts_ + tmp1_off, ts_ + tmp_space_off);
                } else if (rt > 0 && rl > 0) {
                    lstm_fwd_prop_single<data_t>(
                        state_size, state_size, batch_size,
                        ws_ptr + hout_space_off + (rl - 1) * h_size + rt * h_nlayer_size,
                        NOTRANS,
                        ws_ptr + hout_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        NOTRANS,
                        ws_ptr + c_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        NOTRANS, weights_pack[l],
                        ws_ptr + hout_space_off + rl * h_size + rt * h_nlayer_size,
                        ws_ptr + c_space_off + rl * h_size + rt * h_nlayer_size,
                        ws_ptr + gates_space_off + rl * gates_size +
                            rt * gates_nlayer_size,
                        ts_ + tmp1_off, ts_ + tmp_space_off);
                }
                // save y output
                if (rl == (num_layers - 1)) {
                    omatcopy<data_type>(
                        'R', 'T', state_size, batch_size, 1.0,
                        ws_ptr + hout_space_off + rl * h_size + rt * h_nlayer_size,
                        batch_size, y + t * h_size * direction, state_size);
                }
                if (direction == 1 && rt == (seq_length - 1)) {
                    if (hy != nullptr)
                        omatcopy<data_type>('R', 'T', state_size, batch_size, 1.0,
                                        ws_ptr + hout_space_off + rl * h_size +
                                            rt * h_nlayer_size,
                                        batch_size, hy + l * h_size, state_size);
                    if (cy != nullptr)
                        omatcopy<data_type>('R', 'T', state_size, batch_size, 1.0,
                                        ws_ptr + c_space_off + rl * h_size +
                                            rt * h_nlayer_size,
                                        batch_size, cy + l * h_size, state_size);
                }
            }
        } else if (dl == 1) {
            for (int t = (seq_length - 1); t >= 0; t--) {
                rt = 2 * seq_length - t - 1;
                if (rl == 0) {
                    lstm_fwd_prop_single<data_t>(
                        input_size, state_size, batch_size, x + t * x_size, TRANS,
                        ws_ptr + hout_space_off + (rt - 1) * h_nlayer_size, NOTRANS,
                        ws_ptr + c_space_off + (rt - 1) * h_nlayer_size, NOTRANS,
                        weights_pack[l],
                        ws_ptr + hout_space_off + rl * h_size + rt * h_nlayer_size,
                        ws_ptr + c_space_off + rl * h_size + rt * h_nlayer_size,
                        ws_ptr + gates_space_off + rl * gates_size +
                            rt * gates_nlayer_size,
                        ts_ + tmp1_off, ts_ + tmp_space_off);
                } else if (rl > 0) {
                    lstm_fwd_prop_single<data_t>(
                        state_size, state_size, batch_size,
                        ws_ptr + hout_space_off + (rl - 1) * h_size + rt * h_nlayer_size,
                        NOTRANS,
                        ws_ptr + hout_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        NOTRANS,
                        ws_ptr + c_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        NOTRANS, weights_pack[l],
                        ws_ptr + hout_space_off + rl * h_size + rt * h_nlayer_size,
                        ws_ptr + c_space_off + rl * h_size + rt * h_nlayer_size,
                        ws_ptr + gates_space_off + rl * gates_size +
                            rt * gates_nlayer_size,
                        ts_ + tmp1_off, ts_ + tmp_space_off);
                }
                // save y output
                if (rl == (num_layers - 1)) {
                    omatcopy<data_type>(
                        'R', 'T', state_size, batch_size, 1.0,
                        ws_ptr + hout_space_off + rl * h_size + rt * h_nlayer_size,
                        batch_size, y + t * h_size * direction + h_size, state_size);
                }
                if (direction == 2 && t == 0) {
                    if (hy != nullptr)
                        omatcopy<data_type>('R', 'T', state_size, batch_size, 1.0,
                                    ws_ptr + hout_space_off + rl * h_size +
                                        rt * h_nlayer_size,
                                    batch_size, hy + rl * h_size, state_size);
                    if (cy != nullptr)
                        omatcopy<data_type>('R', 'T', state_size, batch_size, 1.0,
                                    ws_ptr + c_space_off + rl * h_size +
                                        rt * h_nlayer_size,
                                    batch_size, cy + rl * h_size, state_size);
                }
            }
        }
    }
    for (int nl = 0; nl < total_layers; nl++) {
        cblas_gemm_free<data_type>(weights_pack[nl]);
    }
#endif // USE_MKL
}

template <impl::data_type_t data_type>
void pgemm_rnn_bwd_t<data_type>::execute_backward() {
#ifdef USE_MKL
    auto x = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto hx = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto cx = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dy = reinterpret_cast<const data_t *>(this->input_memory(3));
    int state_outputs = conf_.state_outputs();
    const data_t *dhy;
    const data_t *dcy;
    const data_t *w;
    const data_t *ws;
    if (state_outputs) {
        dhy = reinterpret_cast<const data_t *>(this->input_memory(4));
        dcy = reinterpret_cast<const data_t *>(this->input_memory(5));
        w = reinterpret_cast<const data_t *>(this->input_memory(6));
        ws = reinterpret_cast<const data_t *>(this->input_memory(7));
    } else {
        w = reinterpret_cast<const data_t *>(this->input_memory(4));
        ws = reinterpret_cast<const data_t *>(this->input_memory(5));
        dhy = nullptr;
        dcy = nullptr;
    }

    auto dx = reinterpret_cast<data_t *>(this->memory(0));
    auto dhx = reinterpret_cast<data_t *>(this->memory(1));
    auto dcx = reinterpret_cast<data_t *>(this->memory(2));
    auto dw = reinterpret_cast<data_t *>(this->memory(3));

    const size_t seq_length = conf_.tau();
    const size_t num_layers = conf_.layers();
    const size_t batch_size = conf_.batch();
    const size_t input_size = conf_.input_size();
    const size_t state_size = conf_.hidden_size();
    const size_t direction = conf_.direction();
    const size_t total_layers = num_layers * direction;
    const size_t w1_size = conf_.w1_size();
    const size_t wx_size = conf_.wx_size();
    const size_t h_size = conf_.h_size();
    const size_t x_size = conf_.x_size();
    const size_t h_nlayer_size = conf_.h_nlayer_size();
    const size_t gates_size = conf_.gates_size();
    const size_t gates_nlayer_size = conf_.gates_nlayer_size();
    const size_t gates_space_size = conf_.gates_space_size();
    const size_t hout_space_size = conf_.hout_space_size();
    const size_t c_space_size = conf_.c_space_size();

    const memory_desc_wrapper ws_d(conf_.workspace_pd());
    const size_t gates_space_off = ws_d.blk_off(0);
    const size_t hout_space_off = ws_d.blk_off(gates_space_size);
    const size_t c_space_off = ws_d.blk_off(hout_space_off + hout_space_size);

    const size_t dgates_space_off = 0;
    const size_t dhout_space_off = dgates_space_off + gates_space_size;
    const size_t dc_space_off = dhout_space_off + hout_space_size;
    const size_t tmp1_off = dc_space_off + c_space_size;
    const size_t tmp_space_off = tmp1_off + 3 * h_size;
    auto bsize = (input_size > state_size) ? input_size : state_size;
    auto tmp1 = bsize + state_size + 2;
    auto tmp2 = state_size * 4;
    auto tmp = (tmp1 > tmp2) ? tmp1 : tmp2;
    auto ts_size_ = tmp * batch_size + gates_space_size +
                    hout_space_size + c_space_size +
                    4 * h_size;
    memset(ts_, 0, ts_size_ * sizeof(data_t));
    array_set(ts_ + tmp1_off, 1.0, 3 * h_size);
    size_t w_off = 0;
    size_t in_size = 0;
    size_t wa = w1_size + (num_layers - 1) * wx_size;
    size_t dl, rl, roff, rt;

    #pragma omp parallel for
    for (size_t seq = 0; seq < seq_length; seq++) {
        omatcopy<data_type>('R', 'T', batch_size, state_size, 1.0,
                            dy + seq * h_size * direction, state_size,
                            ts_ + dhout_space_off + (h_nlayer_size - h_size) +
                                seq * h_nlayer_size,
                            batch_size);
        if (direction == 2)
          omatcopy<data_type>('R', 'T', batch_size, state_size, 1.0,
                            dy + seq * h_size * direction + h_size, state_size,
                            ts_ + dhout_space_off + (h_nlayer_size - h_size) +
                                (seq_length * direction - 1 - seq) *
                                    h_nlayer_size,
                            batch_size);
    }
    if (state_outputs) {
        if (num_layers > 1) {
#pragma omp parallel for
            for (size_t ly = 0; ly < (num_layers - 1); ly++) {
                omatcopy<data_type>('R', 'T', batch_size, state_size, 1.0,
                                    dhy + ly * h_size, state_size,
                                    ts_ + dhout_space_off +
                                        (seq_length * direction - 1) * h_nlayer_size +
                                        ly * h_size,
                                    batch_size);
            }
        }
#pragma omp parallel for
        for (size_t ly = 0; ly < num_layers; ly++) {
            omatcopy<data_type>(
                    'R', 'T', batch_size, state_size, 1.0, dcy + ly * h_size, state_size,
                    ts_ + dc_space_off + (seq_length * direction - 1) * h_nlayer_size +
                        ly * h_size,
                    batch_size);
        }
    }
    data_t **weights_pack = new data_t *[total_layers];
    for (int l = 0; l < total_layers; l++) {
        dl = l / num_layers;
        rl = l % num_layers;
        roff = (rl == 0) ? 0 : (w1_size + (rl - 1) * wx_size);
        w_off = wa * dl + roff;
        in_size = (rl == 0) ? input_size : state_size;
        // pack weights
        weights_pack[l] = cblas_gemm_alloc<data_type>(
            CblasAMatrix, 4 * state_size, batch_size, (in_size + state_size + 2));
        cblas_gemm_pack<data_type>(CblasRowMajor, CblasAMatrix, CblasNoTrans,
                                (in_size + state_size + 2), batch_size,
                                4 * state_size, 1.0, w + w_off, 4 * state_size,
                                weights_pack[l]);
    }

    for (int l = (total_layers - 1); l >= 0; l--) {
        dl = l / num_layers;
        rl = l % num_layers;
        roff = (rl == 0) ? 0 : (w1_size + (rl - 1) * wx_size);
        w_off = wa * dl + roff;
        // std::cout << "woff = " << w_off << std::endl;
        if (dl == 1) {
            for (int t = 0; t < seq_length; t++) {
                rt = 2 * seq_length - t - 1;
                if (rl > 0) {
                    lstm_bwd_prop_single(
                        state_size, state_size, batch_size,
                        ws + hout_space_off + (rl - 1) * h_size + rt * h_nlayer_size,
                        NOTRANS,
                        ws + hout_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        NOTRANS,
                        ws + c_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        NOTRANS, ws + c_space_off + rl * h_size + rt * h_nlayer_size,
                        weights_pack[l],
                        ws + gates_space_off + rl * gates_size + rt * gates_nlayer_size,
                        ts_ + dhout_space_off + rl * h_size + rt * h_nlayer_size,
                        ts_ + dc_space_off + rl * h_size + rt * h_nlayer_size, dw + w_off,
                        ts_ + dhout_space_off + (rl - 1) * h_size + rt * h_nlayer_size,
                        ts_ + dhout_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        ts_ + dc_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        ts_ + dgates_space_off + rl * gates_size + rt * gates_nlayer_size,
                        ts_ + tmp1_off, ts_ + tmp_space_off);
                } else if (rl == 0) {
                    lstm_bwd_prop_single(
                        input_size, state_size, batch_size, x + t * x_size, TRANS,
                        ws + hout_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        NOTRANS,
                        ws + c_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        NOTRANS, ws + c_space_off + rl * h_size + rt * h_nlayer_size,
                        weights_pack[l],
                        ws + gates_space_off + rl * gates_size + rt * gates_nlayer_size,
                        ts_ + dhout_space_off + rl * h_size + rt * h_nlayer_size,
                        ts_ + dc_space_off + rl * h_size + rt * h_nlayer_size, dw + w_off,
                        dx + t * x_size,
                        ts_ + dhout_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        ts_ + dc_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        ts_ + dgates_space_off + rl * gates_size + rt * gates_nlayer_size,
                        ts_ + tmp1_off, ts_ + tmp_space_off);
                }
            }
        } else if (dl == 0) {
            for (int t = (seq_length - 1); t >= 0; t--) {
                rt = t;
                if (rl > 0 && rt > 0) {
                    lstm_bwd_prop_single(
                        state_size, state_size, batch_size,
                        ws + hout_space_off + (rl - 1) * h_size + rt * h_nlayer_size,
                        NOTRANS,
                        ws + hout_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        NOTRANS,
                        ws + c_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        NOTRANS, ws + c_space_off + rl * h_size + rt * h_nlayer_size,
                        weights_pack[l],
                        ws + gates_space_off + rl * gates_size + rt * gates_nlayer_size,
                        ts_ + dhout_space_off + rl * h_size + rt * h_nlayer_size,
                        ts_ + dc_space_off + rl * h_size + rt * h_nlayer_size, dw + w_off,
                        ts_ + dhout_space_off + (rl - 1) * h_size + rt * h_nlayer_size,
                        ts_ + dhout_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        ts_ + dc_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        ts_ + dgates_space_off + rl * gates_size + rt * gates_nlayer_size,
                        ts_ + tmp1_off, ts_ + tmp_space_off);
                } else if (rl == 0 && rt > 0) {
                    lstm_bwd_prop_single(
                        input_size, state_size, batch_size, x + t * x_size, TRANS,
                        ws + hout_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        NOTRANS,
                        ws + c_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        NOTRANS, ws + c_space_off + rl * h_size + rt * h_nlayer_size,
                        weights_pack[l],
                        ws + gates_space_off + rl * gates_size + rt * gates_nlayer_size,
                        ts_ + dhout_space_off + rl * h_size + rt * h_nlayer_size,
                        ts_ + dc_space_off + rl * h_size + rt * h_nlayer_size, dw + w_off,
                        dx + t * x_size,
                        ts_ + dhout_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        ts_ + dc_space_off + rl * h_size + (rt - 1) * h_nlayer_size,
                        ts_ + dgates_space_off + rl * gates_size + rt * gates_nlayer_size,
                        ts_ + tmp1_off, ts_ + tmp_space_off);
                } else if (rl > 0 && rt == 0) {
                    lstm_bwd_prop_single(
                        state_size, state_size, batch_size,
                        ws + hout_space_off + (rl - 1) * h_size + rt * h_nlayer_size,
                        NOTRANS, hx + l * h_size, TRANS, cx + l * h_size, TRANS,
                        ws + c_space_off + rl * h_size + rt * h_nlayer_size,
                        weights_pack[l],
                        ws + gates_space_off + rl * gates_size + rt * gates_nlayer_size,
                        ts_ + dhout_space_off + rl * h_size + rt * h_nlayer_size,
                        ts_ + dc_space_off + rl * h_size + rt * h_nlayer_size, dw + w_off,
                        ts_ + dhout_space_off + (rl - 1) * h_size + rt * h_nlayer_size,
                        dhx + l * h_size, dcx + l * h_size,
                        ts_ + dgates_space_off + rl * gates_size + rt * gates_nlayer_size,
                        ts_ + tmp1_off, ts_ + tmp_space_off);
                } else if (rl == 0 && rt == 0) {
                    lstm_bwd_prop_single(
                        input_size, state_size, batch_size, x + t * x_size, TRANS,
                        hx + l * h_size, TRANS, cx + l * h_size, TRANS,
                        ws + c_space_off + rl * h_size + rt * h_nlayer_size,
                        weights_pack[l],
                        ws + gates_space_off + rl * gates_size + rt * gates_nlayer_size,
                        ts_ + dhout_space_off + rl * h_size + rt * h_nlayer_size,
                        ts_ + dc_space_off + rl * h_size + rt * h_nlayer_size, dw + w_off,
                        dx + t * x_size, dhx + l * h_size, dcx + l * h_size,
                        ts_ + dgates_space_off + rl * gates_size + rt * gates_nlayer_size,
                        ts_ + tmp1_off, ts_ + tmp_space_off);
                }
            }
        }
    }
    for (int nl = 0; nl < total_layers; nl++) {
        cblas_gemm_free<data_type>(weights_pack[nl]);
    }
#endif // USE_MKL
}

template struct pgemm_rnn_fwd_t<data_type::f32>;
template struct pgemm_rnn_bwd_t<data_type::f32>;
}
}
}
