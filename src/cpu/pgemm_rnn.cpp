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

#include "c_types_map.hpp"
#include "type_helpers.hpp"

#include "pgemm_rnn.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

enum { NOTRANS = 1, TRANS = 2 };

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
#endif

#if defined(_OPENMP)
#if _OPENMP < 201307
#define OMP_SIMD omp declare simd
#define OMP_FOR_SIMD omp parallel for
#else
#define OMP_SIMD omp simd
#define OMP_FOR_SIMD omp parallel for simd
#endif // _OPENMP < 201307
#endif // _OPENMP

template <typename data_t>
#if defined(_OPENMP)
#pragma OMP_SIMD
#endif
inline void lstm_fwd_ele_wise(data_t *Gates, const data_t *Ct_1, data_t *Ct,
        data_t *Ht, int Length)
{
#if defined(_OPENMP)
#pragma OMP_FOR_SIMD
#endif
    for (int i = 0; i < Length; i++) {
        Gates[i] = Sigmoid<data_traits<data_t>::data_type>(Gates[i]);
        Gates[Length + i]
                = Sigmoid<data_traits<data_t>::data_type>(Gates[Length + i]);
        Gates[2 * Length + i] = Sigmoid<data_traits<data_t>::data_type>(
                Gates[2 * Length + i]);
        Gates[3 * Length + i]
                = Tanh<data_traits<data_t>::data_type>(Gates[3 * Length + i]);
        Ct[i] = Ct_1[i] * Gates[Length + i] + Gates[i] * Gates[3 * Length + i];
        Ht[i] = Gates[2 * Length + i]
                * Tanh<data_traits<data_t>::data_type>(Ct[i]);
    }
}

template <typename data_t>
#if defined(_OPENMP)
#pragma OMP_SIMD
#endif
inline void lstm_bwd_ele_wise(const data_t *Gates, data_t *dGates,
        const data_t *Ct_1, data_t *dCt_1, const data_t *Ct, data_t *dCt,
        const data_t *dHt, int Length)
{
#if defined(_OPENMP)
#pragma OMP_FOR_SIMD
#endif
    for (int i = 0; i < Length; i++) {
        dCt[i] += (1 - Pow<data_traits<data_t>::data_type>(
                               Tanh<data_traits<data_t>::data_type>(Ct[i]), 2))
                * dHt[i] * Gates[2 * Length + i];
        dCt_1[i] = dCt[i] * Gates[Length + i];
        dGates[i] = dCt[i] * Gates[3 * Length + i] * Gates[i] * (1 - Gates[i]);
        dGates[Length + i] = dCt[i] * Ct_1[i] * Gates[Length + i]
                * (1 - Gates[Length + i]);
        dGates[2 * Length + i] = dHt[i]
                * Tanh<data_traits<data_t>::data_type>(Ct[i])
                * Gates[2 * Length + i] * (1 - Gates[2 * Length + i]);
        dGates[3 * Length + i] = dCt[i] * Gates[i]
                * (1 - Gates[3 * Length + i] * Gates[3 * Length + i]);
    }
}

template <typename data_t>
#if defined(_OPENMP)
#pragma OMP_SIMD
#endif
inline void lstm_fwd_prop_single(const int input_size,
        const int state_size, const int batch_size, const data_t *x,
        int tranx, const data_t *ht_1, int tranht_1, const data_t *ct_1,
        int tranct_1, const data_t *w, data_t *ht, data_t *ct, data_t *gates,
        data_t *tmp)
{
#ifdef USE_MKL
    auto x_size = input_size * batch_size;
    auto h_size = state_size * batch_size;
    if (tranx == TRANS) {
        omatcopy<data_traits<data_t>::data_type>('R', 'T', batch_size,
                input_size, 1.0, x, input_size, tmp, batch_size);
    } else {
#if defined(_OPENMP)
#pragma OMP_FOR_SIMD
#endif
        for (int ii = 0; ii < x_size; ++ii)
            tmp[ii] = x[ii];
    }
    if (tranht_1 == TRANS) {
        omatcopy<data_traits<data_t>::data_type>('R', 'T', batch_size,
                state_size, 1.0, ht_1, state_size, tmp + x_size, batch_size);
    } else {
#if defined(_OPENMP)
#pragma OMP_FOR_SIMD
#endif
        for (int ii = 0; ii < h_size; ++ii)
            tmp[ii + x_size] = ht_1[ii];
    }
    array_set(tmp + x_size + h_size, 1.0, 2 * batch_size);
    cblas_gemm_compute<data_traits<data_t>::data_type>(CblasRowMajor,
            CblasPacked, CblasNoTrans, 4 * state_size, batch_size,
            input_size + state_size + 2, w, 4 * state_size, tmp, batch_size,
            0.0, gates, batch_size);
    if (tranct_1 == TRANS) {
        omatcopy<data_traits<data_t>::data_type>('R', 'T', batch_size,
                state_size, 1.0, ct_1, state_size, tmp, batch_size);
        lstm_fwd_ele_wise<data_t>(gates, tmp, ct, ht, h_size);
    } else {
        lstm_fwd_ele_wise<data_t>(gates, ct_1, ct, ht, h_size);
    }
#endif
}

template <typename data_t>
inline void lstm_bwd_prop_single(const int input_size,
        const int state_size, const int batch_size, const data_t *x,
        int tranx, const data_t *ht_1, int tranht_1, const data_t *ct_1,
        int tranct_1, const data_t *ct, const data_t *w, const data_t *gates,
        data_t *dht, data_t *dct, data_t *dw, data_t *dx, data_t *dht_1,
        data_t *dct_1, data_t *dgates, data_t *tmp)
{
#ifdef USE_MKL
    auto x_size = input_size * batch_size;
    auto h_size = state_size * batch_size;
    if (tranct_1 == TRANS) {
        omatcopy<data_traits<data_t>::data_type>('R', 'T', batch_size,
                state_size, 1.0, ct_1, state_size, tmp, batch_size);
        lstm_bwd_ele_wise<data_t>(
                gates, dgates, tmp, tmp + h_size, ct, dct, dht, h_size);
        omatcopy<data_traits<data_t>::data_type>('R', 'T', state_size,
                batch_size, 1.0, tmp + h_size, batch_size, dct_1, state_size);
    } else {
        lstm_bwd_ele_wise<data_t>(
                gates, dgates, ct_1, dct_1, ct, dct, dht, h_size);
    }
    if (tranx == TRANS) {
        omatcopy<data_traits<data_t>::data_type>('R', 'T', batch_size,
                input_size, 1.0, x, input_size, tmp, batch_size);
    } else {
#if defined(_OPENMP)
#pragma OMP_FOR_SIMD
#endif
        for (int ii = 0; ii < x_size; ++ii)
            tmp[ii] = x[ii];
    }
    if (tranht_1 == TRANS) {
        omatcopy<data_traits<data_t>::data_type>('R', 'T', batch_size,
                state_size, 1.0, ht_1, state_size, tmp + x_size, batch_size);
    } else {
#if defined(_OPENMP)
#pragma OMP_FOR_SIMD
#endif
        for (int ii = 0; ii < h_size; ++ii)
            tmp[ii + x_size] = ht_1[ii];
    }
    array_set(tmp + x_size + h_size, 1.0, 2 * batch_size);
    cblas_gemm<data_traits<data_t>::data_type>(CblasRowMajor, CblasNoTrans,
            CblasTrans, (input_size + state_size + 2), 4 * state_size,
            batch_size, 1.0, tmp, batch_size, dgates, batch_size, 1.0, dw,
            4 * state_size);
    cblas_gemm_compute<data_traits<data_t>::data_type>(CblasRowMajor,
            CblasPacked, CblasNoTrans, (input_size + state_size + 2),
            batch_size, 4 * state_size, w, 4 * state_size, dgates, batch_size,
            0.0, tmp, batch_size);
    if (tranx == TRANS) {
        omatcopy<data_traits<data_t>::data_type>('R', 'T', input_size,
                batch_size, 1.0, tmp, batch_size, dx, input_size);
    } else {
        cblas_axpy<data_traits<data_t>::data_type>(x_size, 1, tmp, 1, dx, 1);
    }
    if (tranht_1 == TRANS) {
        omatcopy<data_traits<data_t>::data_type>('R', 'T', state_size,
                batch_size, 1.0, tmp + x_size, batch_size, dht_1, state_size);
    } else {
        cblas_axpy<data_traits<data_t>::data_type>(
                h_size, 1, tmp + x_size, 1, dht_1, 1);
    }
#endif
}

template <typename data_t>
inline void lstm_fwd_prop(const int seq_length, const int num_layers,
        const int batch_size, const int input_size,
        const int state_size, const int direction, const int w1_size,
        const int wx_size, const int h_size, const int x_size,
        const int h_nlayer_size, const int gates_size,
        const int gates_nlayer_size, const int gates_space_size,
        const int h_space_size, const data_t *x, const data_t *hx,
        const data_t *cx, const data_t *w, data_t *y, data_t *hy, data_t *cy,
        data_t *ws, data_t *ts_, data_t **weights_pack)
{
#ifdef USE_MKL
    const int total_layers = num_layers * direction;
    const int gates_space_off = 0;
    const int h_space_off = gates_space_off + gates_space_size;
    const int c_space_off = h_space_off + h_space_size;
    int tmp_space_off = 0;
    int w_off = 0;
    int in_size = 0;
    int wa = w1_size + (num_layers - 1) * wx_size;
    int dl, rl, roff, rt;
    data_t *ws_ptr;
    if (ws) {
        ws_ptr = ws;
        tmp_space_off = 0;
    } else {
        ws_ptr = ts_;
        tmp_space_off = c_space_off + h_space_size;
    }
    for (int l = 0; l < total_layers; l++) {
        dl = l / num_layers;
        rl = l % num_layers;
        roff = (rl == 0) ? 0 : (w1_size + (rl - 1) * wx_size);
        w_off = wa * dl + roff;
        in_size = (rl == 0) ? input_size : state_size;
        cblas_gemm_pack<data_traits<data_t>::data_type>(CblasRowMajor,
                CblasAMatrix, CblasTrans, 4 * state_size, batch_size,
                (in_size + state_size + 2), 1.0, w + w_off, 4 * state_size,
                weights_pack[l]);
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
                    lstm_fwd_prop_single<data_t>(input_size, state_size,
                            batch_size, x, TRANS, hx, TRANS, cx, TRANS,
                            weights_pack[l], ws_ptr + h_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            ws_ptr + c_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            ws_ptr + gates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                } else if (rt == 0 && rl > 0) {
                    lstm_fwd_prop_single<data_t>(state_size, state_size,
                            batch_size,
                            ws_ptr + h_space_off + (rl - 1) * h_size, NOTRANS,
                            hx + l * h_size, TRANS, cx + l * h_size, TRANS,
                            weights_pack[l], ws_ptr + h_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            ws_ptr + c_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            ws_ptr + gates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                } else if (rt > 0 && rl == 0) {
                    lstm_fwd_prop_single<data_t>(input_size, state_size,
                            batch_size, x + t * x_size, TRANS,
                            ws_ptr + h_space_off + (rt - 1) * h_nlayer_size,
                            NOTRANS,
                            ws_ptr + c_space_off + (rt - 1) * h_nlayer_size,
                            NOTRANS, weights_pack[l], ws_ptr + h_space_off
                                    + rl * h_size + rt * h_nlayer_size,
                            ws_ptr + c_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            ws_ptr + gates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                } else if (rt > 0 && rl > 0) {
                    lstm_fwd_prop_single<data_t>(state_size, state_size,
                            batch_size, ws_ptr + h_space_off + (rl - 1) * h_size
                                    + rt * h_nlayer_size,
                            NOTRANS, ws_ptr + h_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            NOTRANS, ws_ptr + c_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            NOTRANS, weights_pack[l], ws_ptr + h_space_off
                                    + rl * h_size + rt * h_nlayer_size,
                            ws_ptr + c_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            ws_ptr + gates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                }
                // save y output
                if (rl == (num_layers - 1)) {
                    omatcopy<data_traits<data_t>::data_type>('R', 'T',
                            state_size, batch_size, 1.0, ws_ptr + h_space_off
                                    + rl * h_size + rt * h_nlayer_size,
                            batch_size, y + t * h_size * direction, state_size);
                }
                if (direction == 1 && rt == (seq_length - 1)) {
                    if (hy != nullptr)
                        omatcopy<data_traits<data_t>::data_type>('R', 'T',
                                state_size, batch_size, 1.0,
                                ws_ptr + h_space_off + rl * h_size
                                        + rt * h_nlayer_size,
                                batch_size, hy + l * h_size, state_size);
                    if (cy != nullptr)
                        omatcopy<data_traits<data_t>::data_type>('R', 'T',
                                state_size, batch_size, 1.0,
                                ws_ptr + c_space_off + rl * h_size
                                        + rt * h_nlayer_size,
                                batch_size, cy + l * h_size, state_size);
                }
            }
        } else if (dl == 1) {
            for (int t = (seq_length - 1); t >= 0; t--) {
                rt = 2 * seq_length - t - 1;
                if (rl == 0) {
                    lstm_fwd_prop_single<data_t>(input_size, state_size,
                            batch_size, x + t * x_size, TRANS,
                            ws_ptr + h_space_off + (rt - 1) * h_nlayer_size,
                            NOTRANS,
                            ws_ptr + c_space_off + (rt - 1) * h_nlayer_size,
                            NOTRANS, weights_pack[l], ws_ptr + h_space_off
                                    + rl * h_size + rt * h_nlayer_size,
                            ws_ptr + c_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            ws_ptr + gates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                } else if (rl > 0) {
                    lstm_fwd_prop_single<data_t>(state_size, state_size,
                            batch_size, ws_ptr + h_space_off + (rl - 1) * h_size
                                    + rt * h_nlayer_size,
                            NOTRANS, ws_ptr + h_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            NOTRANS, ws_ptr + c_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            NOTRANS, weights_pack[l], ws_ptr + h_space_off
                                    + rl * h_size + rt * h_nlayer_size,
                            ws_ptr + c_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            ws_ptr + gates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                }
                // save y output
                if (rl == (num_layers - 1)) {
                    omatcopy<data_traits<data_t>::data_type>('R', 'T',
                            state_size, batch_size, 1.0, ws_ptr + h_space_off
                                    + rl * h_size + rt * h_nlayer_size,
                            batch_size, y + t * h_size * direction + h_size,
                            state_size);
                }
                if (direction == 2 && t == 0) {
                    if (hy != nullptr)
                        omatcopy<data_traits<data_t>::data_type>('R', 'T',
                                state_size, batch_size, 1.0,
                                ws_ptr + h_space_off + rl * h_size
                                        + rt * h_nlayer_size,
                                batch_size, hy + rl * h_size, state_size);
                    if (cy != nullptr)
                        omatcopy<data_traits<data_t>::data_type>('R', 'T',
                                state_size, batch_size, 1.0,
                                ws_ptr + c_space_off + rl * h_size
                                        + rt * h_nlayer_size,
                                batch_size, cy + rl * h_size, state_size);
                }
            }
        }
    }
#endif
}

template <typename data_t>
inline void lstm_bwd_prop(const int seq_length, const int num_layers,
        const int batch_size, const int input_size,
        const int state_size, const int direction, const int w1_size,
        const int wx_size, const int h_size, const int x_size,
        const int h_nlayer_size, const int gates_size,
        const int gates_nlayer_size, const int gates_space_size,
        const int h_space_size, const int state_outputs, const data_t *x,
        const data_t *hx, const data_t *cx, const data_t *dy, const data_t *dhy,
        const data_t *dcy, const data_t *w, const data_t *ws, data_t *dx,
        data_t *dhx, data_t *dcx, data_t *dw, data_t *ts_,
        data_t **weights_pack)
{
#ifdef USE_MKL
    const int total_layers = num_layers * direction;
    const int gates_space_off = 0;
    const int h_space_off = gates_space_size;
    const int c_space_off = h_space_off + h_space_size;

    const int dgates_space_off = 0;
    const int dh_space_off = dgates_space_off + gates_space_size;
    const int dc_space_off = dh_space_off + h_space_size;
    const int tmp_space_off = dc_space_off + h_space_size;

#if defined(_OPENMP)
#pragma OMP_FOR_SIMD
#endif
    for (int i = 0; i < 2 * h_space_size; i++)
        ts_[dh_space_off + i] = 0;
    int w_off = 0;
    int in_size = 0;
    int wa = w1_size + (num_layers - 1) * wx_size;
    int dl, rl, roff, rt;

#pragma omp parallel for
    // put top dy into dhout space
    for (int seq = 0; seq < seq_length; seq++) {
        omatcopy<data_traits<data_t>::data_type>('R', 'T', batch_size,
                state_size, 1.0, dy + seq * h_size * direction, state_size,
                ts_ + dh_space_off + (h_nlayer_size - h_size)
                        + seq * h_nlayer_size,
                batch_size);
        if (direction == 2)
            omatcopy<data_traits<data_t>::data_type>('R', 'T', batch_size,
                    state_size, 1.0, dy + seq * h_size * direction + h_size,
                    state_size, ts_ + dh_space_off + (h_nlayer_size - h_size)
                            + (seq_length * direction - 1 - seq)
                                    * h_nlayer_size,
                    batch_size);
    }
    if (state_outputs) {
// put dhy into dhout space
#pragma omp parallel for
        for (int ly = 0; ly < num_layers; ly++) {
            omatcopy<data_traits<data_t>::data_type>('R', 'T', batch_size,
                    state_size, 1.0, dhy + ly * h_size, state_size,
                    ts_ + tmp_space_off + ly * h_size, batch_size);
            cblas_axpy<data_traits<data_t>::data_type>(h_size, 1.0,
                    ts_ + tmp_space_off + ly * h_size, 1, ts_ + dh_space_off
                            + (seq_length * direction - 1) * h_nlayer_size
                            + ly * h_size,
                    1);
        }
#pragma omp parallel for
        for (int ly = 0; ly < num_layers; ly++) {
            omatcopy<data_traits<data_t>::data_type>('R', 'T', batch_size,
                    state_size, 1.0, dcy + ly * h_size, state_size,
                    ts_ + dc_space_off
                            + (seq_length * direction - 1) * h_nlayer_size
                            + ly * h_size,
                    batch_size);
        }
    }
    for (int l = 0; l < total_layers; l++) {
        dl = l / num_layers;
        rl = l % num_layers;
        roff = (rl == 0) ? 0 : (w1_size + (rl - 1) * wx_size);
        w_off = wa * dl + roff;
        in_size = (rl == 0) ? input_size : state_size;
        cblas_gemm_pack<data_traits<data_t>::data_type>(CblasRowMajor,
                CblasAMatrix, CblasNoTrans, (in_size + state_size + 2),
                batch_size, 4 * state_size, 1.0, w + w_off, 4 * state_size,
                weights_pack[l]);
    }
    for (int l = (total_layers - 1); l >= 0; l--) {
        dl = l / num_layers;
        rl = l % num_layers;
        roff = (rl == 0) ? 0 : (w1_size + (rl - 1) * wx_size);
        w_off = wa * dl + roff;
        if (dl == 1) {
            for (int t = 0; t < seq_length; t++) {
                rt = 2 * seq_length - t - 1;
                if (rl > 0) {
                    lstm_bwd_prop_single<data_t>(state_size, state_size,
                            batch_size, ws + h_space_off + (rl - 1) * h_size
                                    + rt * h_nlayer_size,
                            NOTRANS, ws + h_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            NOTRANS, ws + c_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            NOTRANS,
                            ws + c_space_off + rl * h_size + rt * h_nlayer_size,
                            weights_pack[l], ws + gates_space_off
                                    + rl * gates_size + rt * gates_nlayer_size,
                            ts_ + dh_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            ts_ + dc_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            dw + w_off, ts_ + dh_space_off + (rl - 1) * h_size
                                    + rt * h_nlayer_size,
                            ts_ + dh_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            ts_ + dc_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            ts_ + dgates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                } else if (rl == 0) {
                    lstm_bwd_prop_single<data_t>(input_size, state_size,
                            batch_size, x + t * x_size, TRANS, ws + h_space_off
                                    + rl * h_size + (rt - 1) * h_nlayer_size,
                            NOTRANS, ws + c_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            NOTRANS,
                            ws + c_space_off + rl * h_size + rt * h_nlayer_size,
                            weights_pack[l], ws + gates_space_off
                                    + rl * gates_size + rt * gates_nlayer_size,
                            ts_ + dh_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            ts_ + dc_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            dw + w_off, dx + t * x_size, ts_ + dh_space_off
                                    + rl * h_size + (rt - 1) * h_nlayer_size,
                            ts_ + dc_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            ts_ + dgates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                }
            }
        } else if (dl == 0) {
            for (int t = (seq_length - 1); t >= 0; t--) {
                rt = t;
                if (rl > 0 && rt > 0) {
                    lstm_bwd_prop_single<data_t>(state_size, state_size,
                            batch_size, ws + h_space_off + (rl - 1) * h_size
                                    + rt * h_nlayer_size,
                            NOTRANS, ws + h_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            NOTRANS, ws + c_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            NOTRANS,
                            ws + c_space_off + rl * h_size + rt * h_nlayer_size,
                            weights_pack[l], ws + gates_space_off
                                    + rl * gates_size + rt * gates_nlayer_size,
                            ts_ + dh_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            ts_ + dc_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            dw + w_off, ts_ + dh_space_off + (rl - 1) * h_size
                                    + rt * h_nlayer_size,
                            ts_ + dh_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            ts_ + dc_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            ts_ + dgates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                } else if (rl == 0 && rt > 0) {
                    lstm_bwd_prop_single<data_t>(input_size, state_size,
                            batch_size, x + t * x_size, TRANS, ws + h_space_off
                                    + rl * h_size + (rt - 1) * h_nlayer_size,
                            NOTRANS, ws + c_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            NOTRANS,
                            ws + c_space_off + rl * h_size + rt * h_nlayer_size,
                            weights_pack[l], ws + gates_space_off
                                    + rl * gates_size + rt * gates_nlayer_size,
                            ts_ + dh_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            ts_ + dc_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            dw + w_off, dx + t * x_size, ts_ + dh_space_off
                                    + rl * h_size + (rt - 1) * h_nlayer_size,
                            ts_ + dc_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            ts_ + dgates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                } else if (rl > 0 && rt == 0) {
                    lstm_bwd_prop_single<data_t>(state_size, state_size,
                            batch_size, ws + h_space_off + (rl - 1) * h_size
                                    + rt * h_nlayer_size,
                            NOTRANS, hx + l * h_size, TRANS, cx + l * h_size,
                            TRANS,
                            ws + c_space_off + rl * h_size + rt * h_nlayer_size,
                            weights_pack[l], ws + gates_space_off
                                    + rl * gates_size + rt * gates_nlayer_size,
                            ts_ + dh_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            ts_ + dc_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            dw + w_off, ts_ + dh_space_off + (rl - 1) * h_size
                                    + rt * h_nlayer_size,
                            dhx + l * h_size, dcx + l * h_size,
                            ts_ + dgates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                } else if (rl == 0 && rt == 0) {
                    lstm_bwd_prop_single<data_t>(input_size, state_size,
                            batch_size, x + t * x_size, TRANS, hx + l * h_size,
                            TRANS, cx + l * h_size, TRANS,
                            ws + c_space_off + rl * h_size + rt * h_nlayer_size,
                            weights_pack[l], ws + gates_space_off
                                    + rl * gates_size + rt * gates_nlayer_size,
                            ts_ + dh_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            ts_ + dc_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            dw + w_off, dx + t * x_size, dhx + l * h_size,
                            dcx + l * h_size, ts_ + dgates_space_off
                                    + rl * gates_size + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                }
            }
        }
    }
#endif
}

template <typename data_t>
inline void rnn_fwd_ele_wise(const data_t *Gates, data_t *Ht,
        const int Length, const int alg_kind)
{
#ifdef USE_MKL
    if (alg_kind == rnn_relu) {
#if defined(_OPENMP)
#pragma OMP_FOR_SIMD
#endif
        for (int i = 0; i < Length; i++) {
            Ht[i] = (Gates[i] > 0.0) ? Gates[i] : 0.0;
        }
    } else if (alg_kind == rnn_tanh) {
#if defined(_OPENMP)
#pragma OMP_FOR_SIMD
#endif
        for (int i = 0; i < Length; i++) {
            Ht[i] = Tanh<data_traits<data_t>::data_type>(Gates[i]);
        }
    }
#endif
}

template <typename data_t>
inline void rnn_fwd_prop_single(const int input_size,
        const int state_size, const int batch_size, const int alg_kind,
        const data_t *x, int tranx, const data_t *ht_1, int tranht_1,
        const data_t *w, data_t *ht, data_t *gates, data_t *tmp)
{
#ifdef USE_MKL
    auto x_size = input_size * batch_size;
    auto h_size = state_size * batch_size;
    if (tranx == TRANS) {
        omatcopy<data_traits<data_t>::data_type>('R', 'T', batch_size,
                input_size, 1.0, x, input_size, tmp, batch_size);
    } else {
#if defined(_OPENMP)
#pragma OMP_FOR_SIMD
#endif
        for (int ii = 0; ii < x_size; ++ii)
            tmp[ii] = x[ii];
    }
    if (tranht_1 == TRANS) {
        omatcopy<data_traits<data_t>::data_type>('R', 'T', batch_size,
                state_size, 1.0, ht_1, state_size, tmp + x_size, batch_size);
    } else {
#if defined(_OPENMP)
#pragma OMP_FOR_SIMD
#endif
        for (int ii = 0; ii < h_size; ++ii)
            tmp[ii + x_size] = ht_1[ii];
    }
    array_set(tmp + x_size + h_size, 1.0, 2 * batch_size);
    // U*x + W*h + b1 + b2
    cblas_gemm_compute<data_traits<data_t>::data_type>(CblasRowMajor,
            CblasPacked, CblasNoTrans, state_size, batch_size,
            input_size + state_size + 2, w, 0, tmp, batch_size, 0.0, gates,
            batch_size);

    // element-wise
    rnn_fwd_ele_wise<data_t>(gates, ht, h_size, alg_kind);
#endif
}

template <typename data_t>
inline void rnn_bwd_ele_wise(const data_t *Gates, data_t *dGates,
        const data_t *dHt, int Length, const int alg_kind)
{
#ifdef USE_MKL
    if (alg_kind == rnn_relu) {
#if defined(_OPENMP)
#pragma OMP_FOR_SIMD
#endif
        for (int i = 0; i < Length; i++) {
            dGates[i] = (Gates[i] > 0.0) ? dHt[i] : 0.0;
        }
    } else if (alg_kind == rnn_tanh) {
#if defined(_OPENMP)
#pragma OMP_FOR_SIMD
#endif
        for (int i = 0; i < Length; i++) {
            dGates[i] = (1.0 - Pow<data_traits<data_t>::data_type>(Gates[i], 2))
                    * dHt[i];
        }
    }
#endif
}

template <typename data_t>
inline void rnn_bwd_prop_single(const int input_size,
        const int state_size, const int batch_size, const int alg_kind,
        const data_t *x, int tranx, const data_t *ht_1, int tranht_1,
        const data_t *w, const data_t *gates, data_t *dht, data_t *dw,
        data_t *dx, data_t *dht_1, data_t *dgates, data_t *tmp)
{
#ifdef USE_MKL
    auto x_size = input_size * batch_size;
    auto h_size = state_size * batch_size;

    // element-wise back prop
    rnn_bwd_ele_wise<data_t>(gates, dgates, dht, h_size, alg_kind);

    // bwd w.r.t. U*x + W*h + b1 + b2
    if (tranx == TRANS) {
        omatcopy<data_traits<data_t>::data_type>('R', 'T', batch_size,
                input_size, 1.0, x, input_size, tmp, batch_size);
    } else {
#if defined(_OPENMP)
#pragma OMP_FOR_SIMD
#endif
        for (int ii = 0; ii < x_size; ++ii)
            tmp[ii] = x[ii];
    }
    if (tranht_1 == TRANS) {
        omatcopy<data_traits<data_t>::data_type>('R', 'T', batch_size,
                state_size, 1.0, ht_1, state_size, tmp + x_size, batch_size);
    } else {
#if defined(_OPENMP)
#pragma OMP_FOR_SIMD
#endif
        for (int ii = 0; ii < h_size; ++ii)
            tmp[ii + x_size] = ht_1[ii];
    }
    array_set(tmp + x_size + h_size, 1.0, 2 * batch_size);
    cblas_gemm<data_traits<data_t>::data_type>(CblasRowMajor, CblasNoTrans,
            CblasTrans, input_size + state_size + 2, state_size, batch_size,
            1.0, tmp, batch_size, dgates, batch_size, 1.0, dw, state_size);
    cblas_gemm_compute<data_traits<data_t>::data_type>(CblasRowMajor,
            CblasPacked, CblasNoTrans, input_size + state_size + 2, batch_size,
            state_size, w, 0, dgates, batch_size, 0.0, tmp, batch_size);

    if (tranx == TRANS) {
        omatcopy<data_traits<data_t>::data_type>('R', 'T', input_size,
                batch_size, 1.0, tmp, batch_size, dx, input_size);
    } else {
        cblas_axpy<data_traits<data_t>::data_type>(x_size, 1, tmp, 1, dx, 1);
    }
    if (tranht_1 == TRANS) {
        omatcopy<data_traits<data_t>::data_type>('R', 'T', state_size,
                batch_size, 1.0, tmp + x_size, batch_size, dht_1, state_size);
    } else {
        cblas_axpy<data_traits<data_t>::data_type>(
                h_size, 1, tmp + x_size, 1, dht_1, 1);
    }
#endif
}

template <typename data_t>
inline void rnn_fwd_prop(const int seq_length, const int num_layers,
        const int batch_size, const int input_size,
        const int state_size, const int direction, const int alg_kind,
        const int w1_size, const int wx_size, const int h_size,
        const int x_size, const int h_nlayer_size,
        const int gates_size, const int gates_nlayer_size,
        const int gates_space_size, const int hout_space_size,
        const data_t *x, const data_t *hx, const data_t *cx, const data_t *w,
        data_t *y, data_t *hy, data_t *cy, data_t *ws, data_t *ts_,
        data_t **weights_pack)
{
#ifdef USE_MKL
    const int total_layers = num_layers * direction;
    const int gates_space_off = 0;
    const int hout_space_off = gates_space_off + gates_space_size;
    int tmp_space_off = 0;
    int w_off = 0;
    int in_size = 0;
    int wa = w1_size + (num_layers - 1) * wx_size;
    int dl, rl, roff, rt;
    data_t *ws_ptr;
    if (ws) {
        ws_ptr = ws;
        tmp_space_off = 0;
    } else {
        ws_ptr = ts_;
        tmp_space_off = hout_space_off + hout_space_size;
    }
    for (int l = 0; l < total_layers; l++) {
        dl = l / num_layers;
        rl = l % num_layers;
        roff = (rl == 0) ? 0 : (w1_size + (rl - 1) * wx_size);
        w_off = wa * dl + roff;
        in_size = (rl == 0) ? input_size : state_size;
        data_t* reordered_w = new data_t[(in_size + state_size + 2) * state_size];
        // Wx
#pragma omp parallel for
        for (size_t ii = 0; ii < state_size; ii++) {
            for (size_t jj = 0; jj < in_size; jj++) {
                reordered_w[ii*(in_size + state_size + 2) + jj] = 
                    (w + w_off)[ii*in_size + jj];
            }
        }
        // Wh
        size_t offset = in_size*state_size;
#pragma omp parallel for
        for (size_t ii = 0; ii < state_size; ii++) {
            for (size_t jj = 0; jj < state_size; jj++) {
                reordered_w[ii*(in_size + state_size + 2) + in_size + jj] = 
                    (w + w_off)[offset + ii*state_size + jj];
            }
        }
        // bx
        offset += state_size*state_size;
#pragma omp parallel for
        for (size_t ii = 0; ii < state_size; ii++) {
            for (size_t jj = 0; jj < 2; jj++) {
                reordered_w[ii*(in_size + state_size + 2) + in_size + state_size + jj] = 
                    (w + w_off)[offset + ii*2 + jj];
            }
        }
        cblas_gemm_pack<data_traits<data_t>::data_type>(CblasRowMajor,
                CblasAMatrix, CblasNoTrans, state_size, batch_size,
                in_size + state_size + 2, 1.0, reordered_w, in_size + state_size + 2,
                weights_pack[l]);
        delete[] reordered_w;
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
                    rnn_fwd_prop_single<data_t>(input_size, state_size,
                            batch_size, alg_kind, x, TRANS, hx, TRANS,
                            weights_pack[l], ws_ptr + hout_space_off
                                    + rl * h_size + rt * h_nlayer_size,
                            ws_ptr + gates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                } else if (rt == 0 && rl > 0) {
                    rnn_fwd_prop_single<data_t>(state_size, state_size,
                            batch_size, alg_kind,
                            ws_ptr + hout_space_off + (rl - 1) * h_size,
                            NOTRANS, hx + l * h_size, TRANS, weights_pack[l],
                            ws_ptr + hout_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            ws_ptr + gates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                } else if (rt > 0 && rl == 0) {
                    rnn_fwd_prop_single<data_t>(input_size, state_size,
                            batch_size, alg_kind, x + t * x_size, TRANS,
                            ws_ptr + hout_space_off + (rt - 1) * h_nlayer_size,
                            NOTRANS, weights_pack[l], ws_ptr + hout_space_off
                                    + rl * h_size + rt * h_nlayer_size,
                            ws_ptr + gates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                } else if (rt > 0 && rl > 0) {
                    rnn_fwd_prop_single<data_t>(state_size, state_size,
                            batch_size, alg_kind, ws_ptr + hout_space_off
                                    + (rl - 1) * h_size + rt * h_nlayer_size,
                            NOTRANS, ws_ptr + hout_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            NOTRANS, weights_pack[l], ws_ptr + hout_space_off
                                    + rl * h_size + rt * h_nlayer_size,
                            ws_ptr + gates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                }
                // save y output
                if (rl == (num_layers - 1)) {
                    omatcopy<data_traits<data_t>::data_type>('R', 'T',
                            state_size, batch_size, 1.0, ws_ptr + hout_space_off
                                    + rl * h_size + rt * h_nlayer_size,
                            batch_size, y + t * h_size * direction, state_size);
                }
                if (direction == 1 && rt == (seq_length - 1)) {
                    if (hy != nullptr)
                        omatcopy<data_traits<data_t>::data_type>('R', 'T',
                                state_size, batch_size, 1.0,
                                ws_ptr + hout_space_off + rl * h_size
                                        + rt * h_nlayer_size,
                                batch_size, hy + l * h_size, state_size);
                }
            }
        } else if (dl == 1) {
            for (int t = (seq_length - 1); t >= 0; t--) {
                rt = 2 * seq_length - t - 1;
                if (rl == 0) {
                    rnn_fwd_prop_single<data_t>(input_size, state_size,
                            batch_size, alg_kind, x + t * x_size, TRANS,
                            ws_ptr + hout_space_off + (rt - 1) * h_nlayer_size,
                            NOTRANS, weights_pack[l], ws_ptr + hout_space_off
                                    + rl * h_size + rt * h_nlayer_size,
                            ws_ptr + gates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                } else if (rl > 0) {
                    rnn_fwd_prop_single<data_t>(state_size, state_size,
                            batch_size, alg_kind, ws_ptr + hout_space_off
                                    + (rl - 1) * h_size + rt * h_nlayer_size,
                            NOTRANS, ws_ptr + hout_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            NOTRANS, weights_pack[l], ws_ptr + hout_space_off
                                    + rl * h_size + rt * h_nlayer_size,
                            ws_ptr + gates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                }
                // save y output
                if (rl == (num_layers - 1)) {
                    omatcopy<data_traits<data_t>::data_type>('R', 'T',
                            state_size, batch_size, 1.0, ws_ptr + hout_space_off
                                    + rl * h_size + rt * h_nlayer_size,
                            batch_size, y + t * h_size * direction + h_size,
                            state_size);
                }
                if (direction == 2 && t == 0) {
                    if (hy != nullptr)
                        omatcopy<data_traits<data_t>::data_type>('R', 'T',
                                state_size, batch_size, 1.0,
                                ws_ptr + hout_space_off + rl * h_size
                                        + rt * h_nlayer_size,
                                batch_size, hy + rl * h_size, state_size);
                }
            }
        }
    }
#endif
}

template <typename data_t>
inline void rnn_bwd_prop(const int seq_length, const int num_layers,
        const int batch_size, const int input_size,
        const int state_size, const int direction, const int alg_kind,
        const int w1_size, const int wx_size, const int h_size,
        const int x_size, const int h_nlayer_size,
        const int gates_size, const int gates_nlayer_size,
        const int gates_space_size, const int h_space_size,
        const int state_outputs, const data_t *x, const data_t *hx,
        const data_t *cx, const data_t *dy, const data_t *dhy,
        const data_t *dcy, const data_t *w, const data_t *ws, data_t *dx,
        data_t *dhx, data_t *dcx, data_t *dw, data_t *ts_,
        data_t **weights_pack)
{
#ifdef USE_MKL
    const int total_layers = num_layers * direction;
    const int gates_space_off = 0;
    const int h_space_off = gates_space_size;

    const int dgates_space_off = 0;
    const int dh_space_off = dgates_space_off + gates_space_size;
    const int tmp_space_off = dh_space_off + h_space_size;

#pragma omp parallel for
    for (int i = 0; i < h_space_size; i++)
        ts_[dh_space_off + i] = 0;
    int w_off = 0;
    int in_size = 0;
    int wa = w1_size + (num_layers - 1) * wx_size;
    int dl, rl, roff, rt;

#pragma omp parallel for
    // put top dy into dhout space
    for (int seq = 0; seq < seq_length; seq++) {
        omatcopy<data_traits<data_t>::data_type>('R', 'T', batch_size,
                state_size, 1.0, dy + seq * h_size * direction, state_size,
                ts_ + dh_space_off + (h_nlayer_size - h_size)
                        + seq * h_nlayer_size,
                batch_size);
        if (direction == 2)
            omatcopy<data_traits<data_t>::data_type>('R', 'T', batch_size,
                    state_size, 1.0, dy + seq * h_size * direction + h_size,
                    state_size, ts_ + dh_space_off + (h_nlayer_size - h_size)
                            + (seq_length * direction - 1 - seq)
                                    * h_nlayer_size,
                    batch_size);
    }
    if (state_outputs) {
// put dhy into dhout space
#pragma omp parallel for
        for (int ly = 0; ly < num_layers; ly++) {
            omatcopy<data_traits<data_t>::data_type>('R', 'T', batch_size,
                    state_size, 1.0, dhy + ly * h_size, state_size,
                    ts_ + tmp_space_off + ly * h_size, batch_size);
            cblas_axpy<data_traits<data_t>::data_type>(h_size, 1.0,
                    ts_ + tmp_space_off + ly * h_size, 1, ts_ + dh_space_off
                            + (seq_length * direction - 1) * h_nlayer_size
                            + ly * h_size,
                    1);
        }
    }
    for (int l = 0; l < total_layers; l++) {
        dl = l / num_layers;
        rl = l % num_layers;
        roff = (rl == 0) ? 0 : (w1_size + (rl - 1) * wx_size);
        w_off = wa * dl + roff;
        in_size = (rl == 0) ? input_size : state_size;
        cblas_gemm_pack<data_traits<data_t>::data_type>(CblasRowMajor,
                CblasAMatrix, CblasNoTrans, in_size + state_size + 2,
                batch_size, state_size, 1.0, w + w_off, state_size,
                weights_pack[l]);
    }

    for (int l = (total_layers - 1); l >= 0; l--) {
        dl = l / num_layers;
        rl = l % num_layers;
        roff = (rl == 0) ? 0 : (w1_size + (rl - 1) * wx_size);
        w_off = wa * dl + roff;
        if (dl == 1) {
            for (int t = 0; t < seq_length; t++) {
                rt = 2 * seq_length - t - 1;
                if (rl > 0) {
                    rnn_bwd_prop_single<data_t>(state_size, state_size,
                            batch_size, alg_kind, ws + h_space_off
                                    + (rl - 1) * h_size + rt * h_nlayer_size,
                            NOTRANS, ws + h_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            NOTRANS, weights_pack[l], ws + gates_space_off
                                    + rl * gates_size + rt * gates_nlayer_size,
                            ts_ + dh_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            dw + w_off, ts_ + dh_space_off + (rl - 1) * h_size
                                    + rt * h_nlayer_size,
                            ts_ + dh_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            ts_ + dgates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                } else if (rl == 0) {
                    rnn_bwd_prop_single<data_t>(input_size, state_size,
                            batch_size, alg_kind, x + t * x_size, TRANS,
                            ws + h_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            NOTRANS, weights_pack[l], ws + gates_space_off
                                    + rl * gates_size + rt * gates_nlayer_size,
                            ts_ + dh_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            dw + w_off, dx + t * x_size, ts_ + dh_space_off
                                    + rl * h_size + (rt - 1) * h_nlayer_size,
                            ts_ + dgates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                }
            }
        } else if (dl == 0) {
            for (int t = (seq_length - 1); t >= 0; t--) {
                rt = t;
                if (rl > 0 && rt > 0) {
                    rnn_bwd_prop_single<data_t>(state_size, state_size,
                            batch_size, alg_kind, ws + h_space_off
                                    + (rl - 1) * h_size + rt * h_nlayer_size,
                            NOTRANS, ws + h_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            NOTRANS, weights_pack[l], ws + gates_space_off
                                    + rl * gates_size + rt * gates_nlayer_size,
                            ts_ + dh_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            dw + w_off, ts_ + dh_space_off + (rl - 1) * h_size
                                    + rt * h_nlayer_size,
                            ts_ + dh_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            ts_ + dgates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                } else if (rl == 0 && rt > 0) {
                    rnn_bwd_prop_single<data_t>(input_size, state_size,
                            batch_size, alg_kind, x + t * x_size, TRANS,
                            ws + h_space_off + rl * h_size
                                    + (rt - 1) * h_nlayer_size,
                            NOTRANS, weights_pack[l], ws + gates_space_off
                                    + rl * gates_size + rt * gates_nlayer_size,
                            ts_ + dh_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            dw + w_off, dx + t * x_size, ts_ + dh_space_off
                                    + rl * h_size + (rt - 1) * h_nlayer_size,
                            ts_ + dgates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                } else if (rl > 0 && rt == 0) {
                    rnn_bwd_prop_single<data_t>(state_size, state_size,
                            batch_size, alg_kind, ws + h_space_off
                                    + (rl - 1) * h_size + rt * h_nlayer_size,
                            NOTRANS, hx + l * h_size, TRANS, weights_pack[l],
                            ws + gates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + dh_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            dw + w_off,
                            // dx + t * x_size,
                            ts_ + dh_space_off + (rl - 1) * h_size
                                    + rt * h_nlayer_size,
                            dhx + l * h_size, ts_ + dgates_space_off
                                    + rl * gates_size + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                } else if (rl == 0 && rt == 0) {
                    rnn_bwd_prop_single<data_t>(input_size, state_size,
                            batch_size, alg_kind, x + t * x_size, TRANS,
                            hx + l * h_size, TRANS, weights_pack[l],
                            ws + gates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + dh_space_off + rl * h_size
                                    + rt * h_nlayer_size,
                            dw + w_off, dx + t * x_size, dhx + l * h_size,
                            ts_ + dgates_space_off + rl * gates_size
                                    + rt * gates_nlayer_size,
                            ts_ + tmp_space_off);
                }
            }
        }
    }
#endif
}

template <impl::data_type_t data_type>
void pgemm_rnn_fwd_t<data_type>::execute_forward()
{
#ifdef USE_MKL
    auto x = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto hx = reinterpret_cast<const data_t *>(this->input_memory(1));
    const int state_outputs = conf_.state_outputs();
    data_t *y = nullptr;
    data_t *hy = nullptr;
    data_t *cy = nullptr;
    data_t *ws = nullptr;
    if (state_outputs) {
        y = reinterpret_cast<data_t *>(this->memory(0));
        hy = reinterpret_cast<data_t *>(this->memory(1));
        if (conf_.desc()->alg_kind == rnn_lstm) {
            cy = reinterpret_cast<data_t *>(this->memory(2));
            ws = reinterpret_cast<data_t *>(this->memory(3));
        } else {
            ws = reinterpret_cast<data_t *>(this->memory(2));
        }
    } else {
        y = reinterpret_cast<data_t *>(this->memory(0));
        ws = reinterpret_cast<data_t *>(this->memory(1));
    }
    const int seq_length = conf_.tau();
    const int num_layers = conf_.layers();
    const int batch_size = conf_.batch();
    const int input_size = conf_.input_size();
    const int state_size = conf_.hidden_size();
    const int direction = conf_.direction();
    const int w1_size = conf_.w1_size();
    const int wx_size = conf_.wx_size();
    const int h_size = conf_.h_size();
    const int x_size = conf_.x_size();
    const int h_nlayer_size = conf_.h_nlayer_size();
    const int gates_size = conf_.gates_size();
    const int gates_nlayer_size = conf_.gates_nlayer_size();
    const int gates_space_size = conf_.gates_space_size();
    const int h_space_size = conf_.h_space_size();
    const int alg_kind = conf_.desc()->alg_kind;

    if (alg_kind == rnn_relu || alg_kind == rnn_tanh) {
        data_t *cx = nullptr;
        auto w = reinterpret_cast<const data_t *>(this->input_memory(2));
        rnn_fwd_prop(seq_length, num_layers, batch_size, input_size, state_size,
                direction, alg_kind, w1_size, wx_size, h_size, x_size,
                h_nlayer_size, gates_size, gates_nlayer_size, gates_space_size,
                h_space_size, x, hx, cx, w, y, hy, cy, ws, ts_, weights_pack_);
    } else if (conf_.desc()->alg_kind == rnn_lstm) {
        auto cx = reinterpret_cast<const data_t *>(this->input_memory(2));
        auto w = reinterpret_cast<const data_t *>(this->input_memory(3));
        lstm_fwd_prop(seq_length, num_layers, batch_size, input_size,
                state_size, direction, w1_size, wx_size, h_size, x_size,
                h_nlayer_size, gates_size, gates_nlayer_size, gates_space_size,
                h_space_size, x, hx, cx, w, y, hy, cy, ws, ts_, weights_pack_);
    }
#endif // USE_MKL
}

template <impl::data_type_t data_type>
void pgemm_rnn_bwd_t<data_type>::execute_backward()
{
#ifdef USE_MKL
    auto x = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto hx = reinterpret_cast<const data_t *>(this->input_memory(1));
    const data_t *cx;
    const data_t *dy;
    const data_t *dhy;
    const data_t *dcy;
    const data_t *w;
    const data_t *ws;
    int state_outputs = conf_.state_outputs();
    if (state_outputs) {
        if (conf_.desc()->alg_kind == rnn_lstm) {
            cx = reinterpret_cast<const data_t *>(this->input_memory(2));
            dy = reinterpret_cast<const data_t *>(this->input_memory(3));
            dhy = reinterpret_cast<const data_t *>(this->input_memory(4));
            dcy = reinterpret_cast<const data_t *>(this->input_memory(5));
            w = reinterpret_cast<const data_t *>(this->input_memory(6));
            ws = reinterpret_cast<const data_t *>(this->input_memory(7));
        } else {
            cx = nullptr;
            dy = reinterpret_cast<const data_t *>(this->input_memory(2));
            dhy = reinterpret_cast<const data_t *>(this->input_memory(3));
            dcy = nullptr;
            w = reinterpret_cast<const data_t *>(this->input_memory(4));
            ws = reinterpret_cast<const data_t *>(this->input_memory(5));
        }
    } else {
        if (conf_.desc()->alg_kind == rnn_lstm) {
            cx = reinterpret_cast<const data_t *>(this->input_memory(2));
            dy = reinterpret_cast<const data_t *>(this->input_memory(3));
            dhy = nullptr;
            dcy = nullptr;
            w = reinterpret_cast<const data_t *>(this->input_memory(4));
            ws = reinterpret_cast<const data_t *>(this->input_memory(5));
        } else {
            cx = nullptr;
            dy = reinterpret_cast<const data_t *>(this->input_memory(2));
            dhy = nullptr;
            dcy = nullptr;
            w = reinterpret_cast<const data_t *>(this->input_memory(3));
            ws = reinterpret_cast<const data_t *>(this->input_memory(4));
        }
    }

    auto dx = reinterpret_cast<data_t *>(this->memory(0));
    auto dhx = reinterpret_cast<data_t *>(this->memory(1));
    data_t *dcx;
    data_t *dw;

    if (conf_.desc()->alg_kind == rnn_lstm) {
        dcx = reinterpret_cast<data_t *>(this->memory(2));
        dw = reinterpret_cast<data_t *>(this->memory(3));
    } else {
        dcx = nullptr;
        dw = reinterpret_cast<data_t *>(this->memory(2));
    }

    const int seq_length = conf_.tau();
    const int num_layers = conf_.layers();
    const int batch_size = conf_.batch();
    const int input_size = conf_.input_size();
    const int state_size = conf_.hidden_size();
    const int direction = conf_.direction();
    const int w1_size = conf_.w1_size();
    const int wx_size = conf_.wx_size();
    const int h_size = conf_.h_size();
    const int x_size = conf_.x_size();
    const int h_nlayer_size = conf_.h_nlayer_size();
    const int gates_size = conf_.gates_size();
    const int gates_nlayer_size = conf_.gates_nlayer_size();
    const int gates_space_size = conf_.gates_space_size();
    const int h_space_size = conf_.h_space_size();
    const int alg_kind = conf_.desc()->alg_kind;

    if (alg_kind == rnn_relu || alg_kind == rnn_tanh) {
        rnn_bwd_prop(seq_length, num_layers, batch_size, input_size, state_size,
                direction, alg_kind, w1_size, wx_size, h_size, x_size,
                h_nlayer_size, gates_size, gates_nlayer_size, gates_space_size,
                h_space_size, state_outputs, x, hx, cx, dy, dhy, dcy, w, ws, dx,
                dhx, dcx, dw, ts_, weights_pack_);
    } else if (conf_.desc()->alg_kind == rnn_lstm) {
        lstm_bwd_prop(seq_length, num_layers, batch_size, input_size,
                state_size, direction, w1_size, wx_size, h_size, x_size,
                h_nlayer_size, gates_size, gates_nlayer_size, gates_space_size,
                h_space_size, state_outputs, x, hx, cx, dy, dhy, dcy, w, ws, dx,
                dhx, dcx, dw, ts_, weights_pack_);
    }
#endif // USE_MKL
}

template struct pgemm_rnn_fwd_t<data_type::f32>;
template struct pgemm_rnn_bwd_t<data_type::f32>;

} // namespace cpu
} // namespace impl
} // namespace mkldnn
