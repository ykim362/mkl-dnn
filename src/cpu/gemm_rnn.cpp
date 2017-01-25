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

#include "gemm_rnn.hpp"
#include "cpu_math_util.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

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
inline void vsigmoid(data_t *X, data_t *tmp, size_t Len) {
#ifdef USE_MKL
  cblas_scal<data_trait<data_t>::data_type>(Len, -1.0, X, 1);
  vExp<data_trait<data_t>::data_type>(Len, X, X);
  array_set(tmp, 1.0, Len);
  vAdd<data_trait<data_t>::data_type>(Len, tmp, X, X);
  vDiv<data_trait<data_t>::data_type>(Len, tmp, X, X);
#endif
}

template <typename data_t>
inline void lstm_fwd_ele_wise(data_t *Gates,
                              const data_t *Ct_1, data_t *Ct, data_t *Ht,
                              data_t *tmp, size_t Length) {
#ifdef USE_MKL
  // sigmoid
  vsigmoid<data_t>(Gates, tmp, 3 * Length);
  // ft * c_t_1
  vMul<data_trait<data_t>::data_type>(Length, Gates + Length, Ct_1, Ct);
  // tanh(gt) * it
  vTanh<data_trait<data_t>::data_type>(Length, Gates + 3 * Length, Gates + 3 * Length);
  vMul<data_trait<data_t>::data_type>(Length, Gates, Gates + 3 * Length, tmp);
  // Ct=ft*Ct-1 + Gt*It
  vAdd<data_trait<data_t>::data_type>(Length, Ct, tmp, Ct);
  // h_t = ot * tan(ct)
  vTanh<data_trait<data_t>::data_type>(Length, Ct, tmp);
  vMul<data_trait<data_t>::data_type>(Length, Gates + 2 * Length, tmp, Ht);
#endif
}

template <typename data_t>
inline void
lstm_fwd_prop_single(const size_t input_size, const size_t state_size,
                     const size_t batch_size, const data_t *x, int tranx,
                     const data_t *ht_1, int tranht_1, const data_t *ct_1,
                     int tranct_1, const data_t *w, data_t *ht, data_t *ct,
                     data_t *gates, data_t *tmp) {
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

  cblas_gemm<data_trait<data_t>::data_type>(
      CblasRowMajor, CblasTrans, CblasNoTrans, 4 * state_size, batch_size,
      input_size + state_size + 2, 1.0, w, 4 * state_size, tmp, batch_size, 0.0,
      gates, batch_size);
  if (tranct_1 == TRANS) {
    omatcopy<data_trait<data_t>::data_type>('R', 'T', batch_size, state_size,
                                            1.0, ct_1, state_size, tmp,
                                            batch_size);
    lstm_fwd_ele_wise<data_t>(gates, tmp, ct, ht, tmp + h_size,
                              h_size);
  } else {
    lstm_fwd_ele_wise<data_t>(gates, ct_1, ct, ht, tmp, h_size);
  }
#endif
}

template <impl::data_type_t data_type>
void gemm_rnn_fwd_t<data_type>::execute_forward() {
#ifdef USE_MKL
  auto x = reinterpret_cast<const data_t *>(this->input_memory(0));
  auto hx = reinterpret_cast<const data_t *>(this->input_memory(1));
  auto cx = reinterpret_cast<const data_t *>(this->input_memory(2));
  auto w = reinterpret_cast<const data_t *>(this->input_memory(3));

  auto y = reinterpret_cast<data_t *>(this->memory(0));
  auto hy = reinterpret_cast<data_t *>(this->memory(1));
  auto cy = reinterpret_cast<data_t *>(this->memory(2));
  auto ws = reinterpret_cast<data_t *>(this->memory(3));

  const size_t seq_length = conf_.tau();
  const size_t num_layers = conf_.layers();
  const size_t batch_size = conf_.batch();
  const size_t input_size = conf_.input_size();
  const size_t state_size = conf_.hidden_size();
  const size_t w1_size = conf_.w1_size();
  const size_t wx_size = conf_.wx_size();
  const size_t h_size = conf_.h_size();
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

  data_t *ws_ptr;
  if (ws) {
    ws_ptr = ws;
    tmp_space_off = 0;
  } else {
    ws_ptr = ts_;
    tmp_space_off = c_space_off + c_space_size;
  }

  for (size_t t = 0; t < seq_length; t++) {
    for (size_t l = 0; l < num_layers; l++) {
      w_off = 0;
      if (l > 0) {
        w_off = w1_size + (l - 1) * wx_size;
      }
      if (t == 0 && l == 0) {
        lstm_fwd_prop_single<data_t>(
            input_size, state_size, batch_size, x, TRANS, hx, TRANS, cx, TRANS,
            w + w_off, ws_ptr + hout_space_off + l * h_size + t * h_nlayer_size,
            ws_ptr + c_space_off + l * h_size + t * h_nlayer_size,
            ws_ptr + gates_space_off + l * gates_size + t * gates_nlayer_size,
            ts_ + tmp_space_off);
      } else if (t == 0 && l > 0) {
        lstm_fwd_prop_single<data_t>(
            state_size, state_size, batch_size,
            ws_ptr + hout_space_off + (l - 1) * h_size, NOTRANS,
            hx + l * h_size, TRANS, cx + l * h_size, TRANS, w + w_off,
            ws_ptr + hout_space_off + l * h_size + t * h_nlayer_size,
            ws_ptr + c_space_off + l * h_size + t * h_nlayer_size,
            ws_ptr + gates_space_off + l * gates_size + t * gates_nlayer_size,
            ts_ + tmp_space_off);
      } else if (t > 0 && l == 0) {
        lstm_fwd_prop_single<data_t>(
            state_size, state_size, batch_size, x + t * h_size, TRANS,
            ws_ptr + hout_space_off + (t - 1) * h_nlayer_size, NOTRANS,
            ws_ptr + c_space_off + (t - 1) * h_nlayer_size, NOTRANS, w + w_off,
            ws_ptr + hout_space_off + l * h_size + t * h_nlayer_size,
            ws_ptr + c_space_off + l * h_size + t * h_nlayer_size,
            ws_ptr + gates_space_off + l * gates_size + t * gates_nlayer_size,
            ts_ + tmp_space_off);
      } else if (t > 0 && l > 0) {
        lstm_fwd_prop_single<data_t>(
            state_size, state_size, batch_size,
            ws_ptr + hout_space_off + (l - 1) * h_size + t * h_nlayer_size,
            NOTRANS,
            ws_ptr + hout_space_off + l * h_size + (t - 1) * h_nlayer_size,
            NOTRANS,
            ws_ptr + c_space_off + l * h_size + (t - 1) * h_nlayer_size,
            NOTRANS, w + w_off,
            ws_ptr + hout_space_off + l * h_size + t * h_nlayer_size,
            ws_ptr + c_space_off + l * h_size + t * h_nlayer_size,
            ws_ptr + gates_space_off + l * gates_size + t * gates_nlayer_size,
            ts_ + tmp_space_off);
      }
      // save output
      if (l == (num_layers - 1)) {
        omatcopy<data_type>('R', 'T', state_size, batch_size, 1.0,
                            ws_ptr + hout_space_off + l * h_size +
                                t * h_nlayer_size,
                            batch_size, y + t * h_size, state_size);
      }

      if (t == (seq_length - 1)) {
        if (hy != nullptr)
          omatcopy<data_type>('R', 'T', state_size, batch_size, 1.0,
                              ws_ptr + hout_space_off + l * h_size +
                                  t * h_nlayer_size,
                              batch_size, hy + l * h_size, state_size);
        if (cy != nullptr)
          omatcopy<data_type>('R', 'T', state_size, batch_size, 1.0,
                              ws_ptr + c_space_off + l * h_size +
                                  t * h_nlayer_size,
                              batch_size, cy + l * h_size, state_size);
      }
    }
  }
#endif // USE_MKL
}

template <impl::data_type_t data_type>
void gemm_rnn_bwd_t<data_type>::execute_backward() {
#ifdef USE_MKL
  auto x = reinterpret_cast<const data_t *>(this->input_memory(0));
  auto hx = reinterpret_cast<const data_t *>(this->input_memory(1));
  auto cx = reinterpret_cast<const data_t *>(this->input_memory(2));
  auto dy = reinterpret_cast<const data_t *>(this->input_memory(3));
  auto dhy = reinterpret_cast<const data_t *>(this->input_memory(4));
  auto dcy = reinterpret_cast<const data_t *>(this->input_memory(5));
  auto w = reinterpret_cast<const data_t *>(this->input_memory(6));
  auto ws = reinterpret_cast<const data_t *>(this->input_memory(7));

  auto dx = reinterpret_cast<data_t *>(this->memory(0));
  auto dhx = reinterpret_cast<data_t *>(this->memory(1));
  auto dcx = reinterpret_cast<data_t *>(this->memory(2));
  auto dw = reinterpret_cast<data_t *>(this->memory(3));

  const size_t seq_length = conf_.tau();
  const size_t num_layers = conf_.layers();
  const size_t batch_size = conf_.batch();
  const size_t input_size = conf_.input_size();
  const size_t state_size = conf_.hidden_size();
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
  const size_t temp_space_off = dc_space_off + c_space_size;

#pragma omp parallel for
  for (size_t seq = 0; seq < seq_length; seq++) {
    omatcopy<data_type>(
        'R', 'T', batch_size, state_size, 1.0, dy + seq * h_size, state_size,
        ts_ + dhout_space_off + (h_nlayer_size - h_size) + seq * h_nlayer_size,
        batch_size);
  }

  if (num_layers > 1) {
#pragma omp parallel for
    for (size_t ly = 0; ly < (num_layers - 1); ly++) {
      omatcopy<data_type>('R', 'T', batch_size, state_size, 1.0,
                          dhy + ly * h_size, state_size,
                          ts_ + dhout_space_off +
                              (seq_length - 1) * h_nlayer_size + ly * h_size,
                          batch_size);
    }
  }
#pragma omp parallel for
  for (size_t ly = 0; ly < num_layers; ly++) {
    omatcopy<data_type>(
        'R', 'T', batch_size, state_size, 1.0, dcy + ly * h_size, state_size,
        ts_ + dc_space_off + (seq_length - 1) * h_nlayer_size + ly * h_size,
        batch_size);
  }

  for (int t = (seq_length - 1); t >= 0; t--) {
    for (int l = (num_layers - 1); l >= 0; l--) {
      // formula: do[t] = dh[t] * tanh(c[t])
      vTanh<data_type>(h_size,
                       ws + c_space_off + l * h_size + t * h_nlayer_size,
                       ts_ + temp_space_off);

      vMul<data_type>(h_size, ts_ + temp_space_off,
                      ts_ + dhout_space_off + l * h_size + t * h_nlayer_size,
                      ts_ + dgates_space_off + 2 * h_size + (l * gates_size) +
                          (t * gates_nlayer_size));

      // formula: dc[t] += (1-tanh(c[t] **2) * dh[t] * o[t]
      array_set(ts_ + temp_space_off + h_size, 1.0, h_size);
      vSqr<data_type>(h_size, ts_ + temp_space_off, ts_ + temp_space_off);
      vSub<data_type>(h_size, ts_ + temp_space_off + h_size,
                      ts_ + temp_space_off, ts_ + temp_space_off);
      vMul<data_type>(h_size, ts_ + temp_space_off,
                      ws + gates_space_off + 2 * h_size + l * gates_size +
                          t * gates_nlayer_size,
                      ts_ + temp_space_off);
      vMul<data_type>(h_size, ts_ + temp_space_off,
                      ts_ + dhout_space_off + l * h_size + t * h_nlayer_size,
                      ts_ + temp_space_off);
      vAdd<data_type>(h_size, ts_ + temp_space_off,
                      ts_ + dc_space_off + l * h_size + t * h_nlayer_size,
                      ts_ + dc_space_off + l * h_size + t * h_nlayer_size);

      // formula: df[t] = dc[t] * c[t-1]
      // formula: dc[t-1] = dc[t] * f[t]
      if (t > 0) {
        vMul<data_type>(h_size,
                        ts_ + dc_space_off + l * h_size + t * h_nlayer_size,
                        ws + c_space_off + l * h_size + (t - 1) * h_nlayer_size,
                        ts_ + dgates_space_off + h_size + l * gates_size +
                            t * gates_nlayer_size);
        vMul<data_type>(h_size, ws + gates_space_off + h_size + l * gates_size +
                                    t * gates_nlayer_size,
                        ts_ + dc_space_off + l * h_size + t * h_nlayer_size,
                        ts_ + dc_space_off + l * h_size +
                            (t - 1) * h_nlayer_size);
      } else {
        omatcopy<data_type>('R', 'T', batch_size, state_size, 1.0,
                            cx + l * h_size, state_size, ts_ + temp_space_off,
                            batch_size);
        vMul<data_type>(h_size, ts_ + dc_space_off + l * h_size,
                        ts_ + temp_space_off,
                        ts_ + dgates_space_off + h_size + l * gates_size);
        vMul<data_type>(h_size, ws + gates_space_off + h_size + l * gates_size,
                        ts_ + dc_space_off + l * h_size, ts_ + temp_space_off);
        omatcopy<data_type>('R', 'T', state_size, batch_size, 1.0,
                            ts_ + temp_space_off, batch_size, dcx + l * h_size,
                            state_size);
      }

      // di[t] = dc[t] * g[t]
      vMul<data_type>(h_size, ws + gates_space_off + 3 * h_size +
                                  l * gates_size + t * gates_nlayer_size,
                      ts_ + dc_space_off + l * h_size + t * h_nlayer_size,
                      ts_ + dgates_space_off + l * gates_size +
                          t * gates_nlayer_size);

      // dg[t] = dc[t] * i[t]
      vMul<data_type>(h_size, ws + gates_space_off + l * gates_size +
                                  t * gates_nlayer_size,
                      ts_ + dc_space_off + l * h_size + t * h_nlayer_size,
                      ts_ + dgates_space_off + 3 * h_size + l * gates_size +
                          t * gates_nlayer_size);

      // dg'[t] = dg[t] * (1 - g[t]**2)
      vSqr<data_type>(h_size, ws + gates_space_off + 3 * h_size +
                                  l * gates_size + t * gates_nlayer_size,
                      ts_ + temp_space_off);
      vMul<data_type>(h_size, ts_ + temp_space_off,
                      ts_ + dgates_space_off + 3 * h_size + l * gates_size +
                          t * gates_nlayer_size,
                      ts_ + temp_space_off);
      vSub<data_type>(h_size, ts_ + dgates_space_off + 3 * h_size +
                                  l * gates_size + t * gates_nlayer_size,
                      ts_ + temp_space_off,
                      ts_ + dgates_space_off + 3 * h_size + l * gates_size +
                          t * gates_nlayer_size);

      // di'[t] = di[t] * i[t] * (1-i[t])
      // df'[t] = df[t] * f[t] * (1-f[t])
      // do'[t] = do[t] * o[t] * (1-o[t])
      array_set(ts_ + temp_space_off, 1.0, h_size * 3);
      vSub<data_type>(h_size * 3, ts_ + temp_space_off,
                      ws + gates_space_off + l * gates_size +
                          t * gates_nlayer_size,
                      ts_ + temp_space_off);
      vMul<data_type>(h_size * 3, ws + gates_space_off + l * gates_size +
                                      t * gates_nlayer_size,
                      ts_ + temp_space_off, ts_ + temp_space_off);
      vMul<data_type>(
          h_size * 3,
          ts_ + dgates_space_off + l * gates_size + t * gates_nlayer_size,
          ts_ + temp_space_off,
          ts_ + dgates_space_off + l * gates_size + t * gates_nlayer_size);

      if ((t > 0) && (l > 0)) {
        omatcopy<data_type>('R', 'N', state_size, batch_size, 1.0,
                            ws + hout_space_off + (l - 1) * h_size +
                                t * h_nlayer_size,
                            batch_size, ts_ + temp_space_off, batch_size);
        omatcopy<data_type>(
            'R', 'N', state_size, batch_size, 1.0,
            ws + hout_space_off + l * h_size + (t - 1) * h_nlayer_size,
            batch_size, ts_ + temp_space_off + h_size, batch_size);
        array_set(ts_ + temp_space_off + 2 * h_size, 1.0, 2 * batch_size);
      } else if ((t == 0) && (l > 0)) {
        omatcopy<data_type>('R', 'N', state_size, batch_size, 1.0,
                            ws + hout_space_off + (l - 1) * h_size, batch_size,
                            ts_ + temp_space_off, batch_size);
        omatcopy<data_type>('R', 'T', batch_size, state_size, 1.0,
                            hx + l * h_size, state_size,
                            ts_ + temp_space_off + h_size, batch_size);
        array_set(ts_ + temp_space_off + 2 * h_size, 1.0, 2 * batch_size);
      } else if ((l == 0) && (t > 0)) {
        omatcopy<data_type>('R', 'T', batch_size, input_size, 1.0,
                            x + t * x_size, input_size, ts_ + temp_space_off,
                            batch_size);
        omatcopy<data_type>('R', 'N', state_size, batch_size, 1.0,
                            ws + hout_space_off + (t - 1) * h_nlayer_size,
                            batch_size, ts_ + temp_space_off + x_size,
                            batch_size);
        array_set(ts_ + temp_space_off + (input_size + state_size) * batch_size,
                  1.0, 2 * batch_size);
      } else {
        omatcopy<data_type>('R', 'T', batch_size, input_size, 1.0, x,
                            input_size, ts_ + temp_space_off, batch_size);
        omatcopy<data_type>('R', 'T', batch_size, state_size, 1.0, hx,
                            state_size, ts_ + temp_space_off + x_size,
                            batch_size);
        array_set(ts_ + temp_space_off + (input_size + state_size) * batch_size,
                  1.0, 2 * batch_size);
      }

      size_t woffset = 0;
      if (l > 0) {
        woffset = w1_size + (l - 1) * wx_size;
      }
      size_t in_size = (l == 0) ? input_size : state_size;
      cblas_gemm<data_type>(
          CblasRowMajor, CblasNoTrans, CblasTrans, (in_size + state_size + 2),
          4 * state_size, batch_size, 1.0, ts_ + temp_space_off, batch_size,
          ts_ + dgates_space_off + l * gates_size + t * gates_nlayer_size,
          batch_size, 1.0, dw + woffset, 4 * state_size);
      cblas_gemm<data_type>(
          CblasRowMajor, CblasNoTrans, CblasNoTrans, (in_size + state_size + 2),
          batch_size, 4 * state_size, 1.0, w + woffset, 4 * state_size,
          ts_ + dgates_space_off + l * gates_size + t * gates_nlayer_size,
          batch_size, 0.0, ts_ + temp_space_off, batch_size);
      if ((t > 0) && (l > 0)) {
        omatcopy<data_type>('R', 'N', state_size, batch_size, 1.0,
                            ts_ + temp_space_off + h_size, batch_size,
                            ts_ + dhout_space_off + l * h_size +
                                (t - 1) * h_nlayer_size,
                            batch_size);
      } else if ((l == 0) && (t > 0)) {
        omatcopy<data_type>('R', 'T', input_size, batch_size, 1.0,
                            ts_ + temp_space_off, batch_size, dx + t * x_size,
                            input_size);
        omatcopy<data_type>('R', 'N', state_size, batch_size, 1.0,
                            ts_ + temp_space_off + x_size, batch_size,
                            ts_ + dhout_space_off + (t - 1) * h_nlayer_size,
                            batch_size);
      } else if ((t == 0) && (l > 0)) {
        omatcopy<data_type>('R', 'T', state_size, batch_size, 1.0,
                            ts_ + temp_space_off + h_size, batch_size,
                            dhx + l * h_size, state_size);
      } else {
        omatcopy<data_type>('R', 'T', input_size, batch_size, 1.0,
                            ts_ + temp_space_off, batch_size, dx, input_size);
        omatcopy<data_type>('R', 'T', state_size, batch_size, 1.0,
                            ts_ + temp_space_off + x_size, batch_size, dhx,
                            state_size);
      }
    }
  }

#endif // USE_MKL
}

template struct gemm_rnn_fwd_t<data_type::f32>;
template struct gemm_rnn_bwd_t<data_type::f32>;
}
}
}
