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

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::alg_kind;
using namespace mkldnn::impl::data_type;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::primitive_kind;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::cpu::cpu_blas;
using namespace mkldnn::impl::cpu::cpu_trans;
using namespace mkldnn::impl::cpu::cpu_vml;


template <impl::data_type_t data_type>
void gemm_rnn_fwd_t<data_type>::execute_forward() {
#if defined(USE_CBLAS) && defined(USE_TRANS) && defined(USE_VML)
    auto x = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto hx = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto cx = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto w = reinterpret_cast<const data_t *>(this->input_memory(3));

    auto y = reinterpret_cast<data_t*>(this->memory(0));
    auto hy = reinterpret_cast<data_t*>(this->memory(1));
    auto cy = reinterpret_cast<data_t*>(this->memory(2));
    auto ws = reinterpret_cast<data_t*>(this->memory(3));

    const size_t seq_length = conf_.Tau();
    const size_t num_layers = conf_.Layers();
    const size_t batch_size = conf_.Batch();
    const size_t input_size = conf_.Input_size();
    const size_t state_size = conf_.Hidden_size();
    const size_t w1_size = conf_.W1_size();
    const size_t wx_size = conf_.Wx_size();
    const size_t h_size = conf_.H_size();
    const size_t x_size = conf_.X_size();
    const size_t h_nlayer_size = conf_.H_nlayer_size();
    const size_t gates_size = conf_.Gates_size();
    const size_t gates_nlayer_size = conf_.Gates_nlayer_size();
    const size_t gates_space_size = conf_.Gates_space_size();
    const size_t hout_space_size = conf_.Hout_space_size();
    const size_t c_space_size = conf_.C_space_size();

    const memory_desc_wrapper ws_d(conf_.workspace_pd());
    const size_t gates_space_off = ws_d.blk_off(0);
    const size_t hout_space_off = ws_d.blk_off(gates_space_size);
    const size_t c_space_off = ws_d.blk_off(hout_space_off + hout_space_size);
    const size_t tanc_space_off = ws_d.blk_off(c_space_off + c_space_size);

    for (size_t t = 0; t < seq_length; t++) {
      for (size_t l = 0; l < num_layers; l++) {
        // Hin
        if (t == 0 && l == 0) {
            omatcopy<data_type>('R', 'T', batch_size, input_size, 1.0,
                x,
                input_size,
                ts_,
                batch_size);
            omatcopy<data_type>('R', 'T', batch_size, state_size, 1.0,
                hx,
                state_size,
                ts_ + x_size,
                batch_size);
            array_set(ts_ + x_size + h_size,
                1.0, 2 * batch_size);
        } else if (t == 0 && l > 0) {
            omatcopy<data_type>('R', 'N', state_size, batch_size, 1.0,
                ws + hout_space_off + (l - 1) * h_size,
                batch_size,
                ts_,
                batch_size);
            omatcopy<data_type>('R', 'T', batch_size, state_size, 1.0,
                hx + l * h_size,
                state_size,
                ts_ + x_size,
                batch_size);
            array_set(ts_ + x_size + h_size,
                1.0, 2 * batch_size);
        } else if (t > 0 && l == 0) {
            omatcopy<data_type>('R', 'T', batch_size, input_size, 1.0,
                x + t * x_size,
                input_size,
                ts_,
                batch_size);
            omatcopy<data_type>('R', 'N', state_size, batch_size, 1.0,
                ws + hout_space_off + (t - 1) * h_nlayer_size,
                batch_size,
                ts_ + x_size,
                batch_size);
            array_set(ts_ + x_size + h_size,
                1.0, 2 * batch_size);
        } else if (t > 0 && l > 0) {
            omatcopy<data_type>('R', 'N', state_size, batch_size, 1.0,
                ws + hout_space_off + (l - 1) * h_size + t * h_nlayer_size,
                batch_size,
                ts_,
                batch_size);
            omatcopy<data_type>('R', 'N', state_size, batch_size, 1.0,
                ws + hout_space_off + l * h_size + (t - 1) * h_nlayer_size,
                batch_size,
                ts_ + x_size,
                batch_size);
            array_set(ts_ + x_size + h_size,
                1.0, 2 * batch_size);
        }

        size_t w_base_offset = 0;
        if (l > 0) {
          w_base_offset = w1_size + (l - 1) * wx_size;
        }
        size_t in_size = (l == 0) ? input_size : state_size;
        
        cblas_gemm<data_type>(CblasRowMajor, CblasTrans, CblasNoTrans,
                static_cast<int>(4 * state_size), static_cast<int>(batch_size),
                static_cast<int>(in_size + state_size + 2),
                1.0,
                w + w_base_offset,
                static_cast<int>(4 * state_size),
                ts_,
                static_cast<int>(batch_size),
                0.0,
                ws + gates_space_off + l * gates_size + t * gates_nlayer_size,
                static_cast<int>(batch_size));

        // sigmoid
        cblas_scal<data_type>(3 * h_size, -1.0,
                    ws + gates_space_off + l * gates_size + t * gates_nlayer_size, 1);
        vExp<data_type>(3 * h_size,
            ws + gates_space_off + l * gates_size + t * gates_nlayer_size,
            ws + gates_space_off + l * gates_size + t * gates_nlayer_size);
        array_set(ts_,
                    1.0, 3 * h_size);
        vAdd<data_type>(3 * h_size,
            ws + gates_space_off + l * gates_size + t * gates_nlayer_size,
            ts_,
            ws + gates_space_off + l * gates_size + t * gates_nlayer_size);
        vDiv<data_type>(3 * h_size,
          ts_,
          ws + gates_space_off + l * gates_size + t * gates_nlayer_size,
          ws + gates_space_off + l * gates_size + t * gates_nlayer_size);

        // tanh
        vTanh<data_type>(h_size,
              ws + gates_space_off + l * gates_size + t * gates_nlayer_size + 3 * h_size,
              ws + gates_space_off + l * gates_size + t * gates_nlayer_size + 3 * h_size);

        // ft * c_t_1
        if (t == 0) {
        vMul<data_type>(h_size,
              ws + gates_space_off + l * gates_size + t * gates_nlayer_size + h_size,
              cx + l * h_size,
              ws + c_space_off + l * h_size);
        } else {
        vMul<data_type>(h_size,
              ws + gates_space_off + l * gates_size + t * gates_nlayer_size + h_size,
              ws + c_space_off + l * h_size + (t - 1) * h_nlayer_size,
              ws + c_space_off + l * h_size + t * h_nlayer_size);
        }

        // tanh(gt) * it
        vMul<data_type>(h_size,
              ws + gates_space_off + l * gates_size + t * gates_nlayer_size,
              ws + gates_space_off + l * gates_size + t * gates_nlayer_size + 3 * h_size,
              ts_);

        // Ct=ft*Ct-1 + Tanh(gt)*It
        vAdd<data_type>(h_size,
            ws + c_space_off + l * h_size + t * h_nlayer_size,
            ts_,
            ws + c_space_off + l * h_size + t * h_nlayer_size);

        // h_t = ot * tan(ct)
        vTanh<data_type>(h_size,
            ws + c_space_off + l * h_size + t * h_nlayer_size,
            ts_);

        vMul<data_type>(h_size,
            ws + gates_space_off + l * gates_size + t * gates_nlayer_size + 2 * h_size,
            ts_,
            ws + hout_space_off + l * h_size + t * h_nlayer_size);

        // save output
        if (l == num_layers - 1) {
            omatcopy<data_type>('R', 'T', state_size, batch_size, 1.0,
                        ws + hout_space_off +
                        (h_nlayer_size - h_size) + t * h_nlayer_size,
                        batch_size,
                        y + t * h_size,
                        state_size);
        }

        if (t == (seq_length - 1)) {
            if (y != nullptr)
            omatcopy<data_type>('R', 'T', state_size, batch_size, 1.0,
                          ws + hout_space_off +
                          (seq_length - 1) * h_nlayer_size + l * h_size,
                          batch_size,
                          hy + l * h_size,
                          state_size);
            if (cy != nullptr)
            omatcopy<data_type>('R', 'T', state_size, batch_size, 1.0,
                          ws + c_space_off +
                          (seq_length - 1) * h_nlayer_size + l * h_size,
                          batch_size,
                          cy + l * h_size,
                          state_size);
          }
        }
      }
#endif
}

template <impl::data_type_t data_type>
void gemm_rnn_bwd_t<data_type>::execute_backward() {
#if defined(USE_CBLAS) && defined(USE_TRANS) && defined(USE_VML)
    auto x = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto hx = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto cx = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dy = reinterpret_cast<const data_t *>(this->input_memory(3));
    auto dhy = reinterpret_cast<const data_t *>(this->input_memory(4));
    auto dcy = reinterpret_cast<const data_t *>(this->input_memory(5));
    auto w = reinterpret_cast<const data_t *>(this->input_memory(6));
    auto ws = reinterpret_cast<const data_t *>(this->input_memory(7));

    auto dx = reinterpret_cast<data_t*>(this->memory(0));
    auto dhx = reinterpret_cast<data_t*>(this->memory(1));
    auto dcx = reinterpret_cast<data_t*>(this->memory(2));
    auto dw = reinterpret_cast<data_t*>(this->memory(3));

    const size_t seq_length = conf_.Tau();
    const size_t num_layers = conf_.Layers();
    const size_t batch_size = conf_.Batch();
    const size_t input_size = conf_.Input_size();
    const size_t state_size = conf_.Hidden_size();
    const size_t w1_size = conf_.W1_size();
    const size_t wx_size = conf_.Wx_size();
    const size_t h_size = conf_.H_size();
    const size_t x_size = conf_.X_size();
    const size_t h_nlayer_size = conf_.H_nlayer_size();
    const size_t gates_size = conf_.Gates_size();
    const size_t gates_nlayer_size = conf_.Gates_nlayer_size();
    const size_t gates_space_size = conf_.Gates_space_size();
    const size_t hout_space_size = conf_.Hout_space_size();
    const size_t c_space_size = conf_.C_space_size();

    const memory_desc_wrapper ws_d(conf_.workspace_pd());
    const size_t gates_space_off = ws_d.blk_off(0);
    const size_t hout_space_off = ws_d.blk_off(gates_space_size);
    const size_t c_space_off = ws_d.blk_off(hout_space_off + hout_space_size);
    const size_t tanc_space_off = ws_d.blk_off(c_space_off + c_space_size);

    const size_t dgates_space_off = 0;
    const size_t dhout_space_off = dgates_space_off + gates_space_size;
    const size_t dc_space_off = dhout_space_off + hout_space_size;
    const size_t temp_space_off = dc_space_off + c_space_size;

#pragma omp parallel for
    for (size_t seq = 0; seq < seq_length; seq++) {
        omatcopy<data_type>('R', 'T', batch_size, state_size, 1.0,
                      dy + seq * h_size,
                      state_size,
                      ts_ + dhout_space_off
                      + (h_nlayer_size - h_size) + seq * h_nlayer_size,
                      batch_size);
      }

    if (num_layers > 1) {
#pragma omp parallel for
        for (size_t ly = 0; ly < (num_layers - 1); ly++) {
            omatcopy<data_type>('R', 'T', batch_size, state_size, 1.0,
                          dhy + ly * h_size,
                          state_size,
                          ts_ + dhout_space_off + (seq_length - 1) * h_nlayer_size + ly * h_size,
                          batch_size);
          }
    }
#pragma omp parallel for
    for (size_t ly = 0; ly < num_layers; ly++) {
      omatcopy<data_type>('R', 'T', batch_size, state_size, 1.0,
                    dcy + ly * h_size,
                    state_size,
                    ts_ + dc_space_off +
                    (seq_length - 1) * h_nlayer_size + ly * h_size,
                    batch_size);
    }

    for (int t = (seq_length - 1); t >= 0; t--) {
        for (int l = (num_layers - 1); l >= 0; l--) {
        // formula: do[t] = dh[t] * tanh(c[t])
        // Impl: dIFOGf_space[t,l,s3,b] = tanC_space[t,l,s,b] * dHout_space[t,l,s,b]
        // vsTanh(h_size,
        // (float*)C_space_ptr + l * h_size + t * h_nlayer_size,
        //  (float*)C_space_ptr + l * h_size + t * h_nlayer_size);

        vMul<data_type>(h_size,
              ws + tanc_space_off
                + l * h_size + t * h_nlayer_size,
              ts_ + dhout_space_off
                + l * h_size + t * h_nlayer_size,
              ts_ + dgates_space_off
                + 2 * h_size + (l * gates_size)
                + (t * gates_nlayer_size));

        // formula: dc[t] += (1-tanh(c[t] **2) * dh[t] * o[t]
        // Impl: dC_space[t,l,s,b] += (1-tanC_space[t][l][s][b] ** 2)
        //           * (IFOGf_space[t,l,s3,b] * dHout_space[t,l,s,b])
        array_set(ts_ + temp_space_off, 1.0, h_size);
        vSqr<data_type>(h_size,
              ws + tanc_space_off
                + l * h_size + t * h_nlayer_size,
              ts_ + temp_space_off
                + h_size);
        vSub<data_type>(h_size,
              ts_ + temp_space_off,
              ts_ + temp_space_off
                + h_size,
              ts_ + temp_space_off);
        vMul<data_type>(h_size,
              ts_ + temp_space_off,
              ws + gates_space_off
                + 2 * h_size + l * gates_size
                + t * gates_nlayer_size,
              ts_ + temp_space_off);
        vMul<data_type>(h_size,
              ts_ + temp_space_off,
              ts_ + dhout_space_off
                + l * h_size + t * h_nlayer_size,
              ts_ + dc_space_off
                + l * h_size + t * h_nlayer_size);
        // ws + temp_space_off);
        // vsAdd(h_size,
        //       ws + temp_space_off,
        //       ws + dc_space_off
        //         + l * h_size + t * h_nlayer_size,
        //       ws + dc_space_off
        //         + l * h_size + t * h_nlayer_size);

        // formula: df[t] = dc[t] * c[t-1]
        // formula: dc[t-1] = dc[t] * f[t]
        // Impl: if t>0
        //      dIFOGf_space[t,l,s2,b] = dC_space[t,l,s,b] * C_space[t-1,l,s,b]
        //      dC_space[t-1,l,s,b] = dC_space[t,l,s,b] * IFOGf_space[t,l,s2,b]
        //    else if t=0
        //      dIFOGf_space[0,l,s2,b] = dC_space[0,l,s,b] * trans(cx[l,b,s])
        //      trans(dcx[l,b,s]) = dC_space[0,l,s,b] * IFOGf_space[0,l,s2,b]
        if (t > 0) {
          vMul<data_type>(h_size,
                ts_ + dc_space_off
                  + l * h_size + t * h_nlayer_size,
                ws + c_space_off
                  + l * h_size + (t - 1) * h_nlayer_size,
                ts_ + dgates_space_off
                  + h_size + l * gates_size + t * gates_nlayer_size);
          vMul<data_type>(h_size,
                ws + gates_space_off
                  + h_size + l * gates_size + t * gates_nlayer_size,
                ts_ + dc_space_off
                  + l * h_size + t * h_nlayer_size,
                ts_ + dc_space_off
                  + l * h_size + (t - 1) * h_nlayer_size);
        } else {
          omatcopy<data_type>('R', 'T',
                        batch_size, state_size,
                        1.0,
                        cx + l * h_size,
                        state_size,
                        ts_ + temp_space_off,
                        batch_size);
          vMul<data_type>(h_size,
                ts_ + dc_space_off
                  + l * h_size,
                ts_ + temp_space_off,
                ts_ + dgates_space_off
                  + h_size + l * gates_size);
          vMul<data_type>(h_size,
                ws + gates_space_off
                  + h_size + l * gates_size,
                ts_ + dc_space_off
                  + l * h_size,
                ts_ + temp_space_off);
          omatcopy<data_type>('R', 'T',
                        state_size, batch_size,
                        1.0,
                        ts_ + temp_space_off,
                        batch_size,
                        dcx + l * h_size,
                        state_size);
        }

        // formula: di[t] = dc[t] * g[t]
        // Impl: dIFOGf[t,l,s1,b] = dC_space[t,l,s,b] * IFOGf_space[t,l,s4,b]
        vMul<data_type>(h_size,
              ws + gates_space_off
                + 3 * h_size + l * gates_size
                + t * gates_nlayer_size,
              ts_ + dc_space_off
                + l * h_size + t * h_nlayer_size,
              ts_ + dgates_space_off
                + l * gates_size
                + t * gates_nlayer_size);

        // formula: dg[t] = dc[t] * i[t]
        // Impl: dIFOGf[t,l,s4,b] = dC_space[t,l,s,b] * IFOGf_space[t,l,s1,b]
        vMul<data_type>(h_size,
              ws + gates_space_off
                + l * gates_size + t * gates_nlayer_size,
              ts_ + dc_space_off
                + l * h_size + t * h_nlayer_size,
              ts_ + dgates_space_off
                + 3 * h_size + l * gates_size
                + t * gates_nlayer_size);

        // formula: dg'[t] = dg[t] * (1 - g[t]**2)
        // Impl: dIFOGf[t,l,s4,b]) = (1 - IFOGf[t,l,s4,b]**2) * dIFOGf[t,l,s4,b]
        vSqr<data_type>(h_size,
              ws + gates_space_off
                + 3 * h_size + l * gates_size
                + t * gates_nlayer_size,
              ts_ + temp_space_off);
        vMul<data_type>(h_size,
              ts_ + temp_space_off,
              ts_ + dgates_space_off
                + 3 * h_size + l * gates_size
                + t * gates_nlayer_size,
              ts_ + temp_space_off);
        vSub<data_type>(h_size,
              ts_ + dgates_space_off
                + 3 * h_size + l * gates_size
                + t * gates_nlayer_size,
              ts_ + temp_space_off,
              ts_ + dgates_space_off
                + 3 * h_size + l * gates_size
                + t * gates_nlayer_size);

        // formula: di'[t] = di[t] * i[t] * (1-i[t])
        //      df'[t] = df[t] * f[t] * (1-f[t])
        //      do'[t] = do[t] * o[t] * (1-o[t])
        // Impl: dIFOGf[t,l,s1-3,b] = dIFOGf[t,l,s1-3,b] * IFOGf[t,l,s1-3,b] * (1-IFOGf[t,l,s1-3,b])
        array_set(ts_ + temp_space_off, 1.0, h_size * 3);
        vSub<data_type>(h_size * 3,
              ts_ + temp_space_off,
              ws + gates_space_off
                + l * gates_size + t * gates_nlayer_size,
              ts_ + temp_space_off);
        vMul<data_type>(h_size * 3,
              ws + gates_space_off
                + l * gates_size + t * gates_nlayer_size,
              ts_ + temp_space_off,
              ts_ + temp_space_off);
        vMul<data_type>(h_size * 3,
              ts_ + dgates_space_off
                + l * gates_size + t * gates_nlayer_size,
              ts_ + temp_space_off,
              ts_ + dgates_space_off
                + l * gates_size + t * gates_nlayer_size);

        if ((t > 0) && (l > 0)) {
          // HinX
          omatcopy<data_type>('R', 'N',
                        state_size, batch_size,
                        1.0,
                        ws + hout_space_off
                          + (l - 1) * h_size + t * h_nlayer_size,
                        batch_size,
                        ts_ + temp_space_off,
                        batch_size);
          omatcopy<data_type>('R', 'N',
                        state_size, batch_size,
                        1.0,
                        ws + hout_space_off
                          + l * h_size + (t - 1) * h_nlayer_size,
                        batch_size,
                        ts_ + temp_space_off + h_size,
                        batch_size);
          array_set(ts_ + temp_space_off
                      + 2 * h_size, 1.0, 2 * batch_size);
        } else if ((t == 0) && (l > 0)) {
          // HinX
          omatcopy<data_type>('R', 'N',
                        state_size, batch_size,
                        1.0,
                        ws + hout_space_off
                          + (l - 1) * h_size,
                        batch_size,
                        ts_ + temp_space_off,
                        batch_size);
          omatcopy<data_type>('R', 'T',
                        batch_size, state_size,
                        1.0,
                        hx + l * h_size,
                        state_size,
                        ts_ + temp_space_off + h_size,
                        batch_size);
          array_set(ts_ + temp_space_off
            + 2 * h_size, 1.0, 2 * batch_size);
        } else if ((l == 0) && (t > 0)) {
          // Hin1
          omatcopy<data_type>('R', 'T',
                        batch_size, input_size,
                        1.0,
                        x + t * x_size,
                        input_size,
                        ts_ + temp_space_off,
                        batch_size);
          omatcopy<data_type>('R', 'N',
                        state_size, batch_size,
                        1.0,
                        ws + hout_space_off
                          + (t - 1) * h_nlayer_size,
                        batch_size,
                        ts_ + temp_space_off + x_size,
                        batch_size);
          array_set(ts_ + temp_space_off
                      + (input_size + state_size) * batch_size,
                      1.0, 2 * batch_size);
        } else {
          // Hin1
          omatcopy<data_type>('R', 'T',
                        batch_size, input_size,
                        1.0,
                        x,
                        input_size,
                        ts_ + temp_space_off,
                        batch_size);
          omatcopy<data_type>('R', 'T',
                        batch_size, state_size,
                        1.0,
                        hx,
                        state_size,
                        ts_ + temp_space_off + x_size,
                        batch_size);
          array_set(ts_ + temp_space_off
                      + (input_size + state_size) * batch_size,
                      1.0, 2 * batch_size);
        }

        size_t mp = 0, woffset = 0;
        if (l == 0) {
          mp = input_size + state_size + 2;
          woffset = 0;
        } else {
          mp = state_size * 2 + 2;
          woffset = w1_size + (l - 1) * wx_size;
        }
        cblas_gemm<data_type>(CblasRowMajor, CblasNoTrans, CblasTrans,
                    mp, 4 * state_size, batch_size,
                    1.0,
                    ts_ + temp_space_off,
                    batch_size,
                    ts_ + dgates_space_off
                      + l * gates_size + t * gates_nlayer_size,
                    batch_size,
                    1.0,
                    dw + woffset,
                    4 * state_size);
        if ((t > 0) && (l > 0)) {
          // dHout[t,l-1,b,s]
          cblas_gemm<data_type>(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                      2 * state_size + 2, batch_size, 4 * state_size,
                      1.0,
                      w + woffset,
                      4 * state_size,
                      ts_ + dgates_space_off
                        + l * gates_size + t * gates_nlayer_size,
                      batch_size,
                      0.0,
                      ts_ + temp_space_off,
                      batch_size);
          omatcopy<data_type>('R', 'N',
            state_size, batch_size,
            1.0,
            ts_ + temp_space_off,
            batch_size,
            ts_ + dhout_space_off
              + (l - 1)*h_size + t*h_nlayer_size,
            batch_size);
          omatcopy<data_type>('R', 'N',
            state_size, batch_size,
            1.0,
            ts_ + temp_space_off + h_size,
            batch_size,
            ts_ + dhout_space_off
              + l * h_size + (t - 1) * h_nlayer_size,
            batch_size);
        } else if ((l == 0) && (t > 0)) {
          cblas_gemm<data_type>(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            input_size + state_size + 2, batch_size, 4 * state_size,
            1.0,
            w + woffset,
            4 * state_size,
            ts_ + dgates_space_off
              + l * gates_size + t * gates_nlayer_size,
            batch_size,
            0.0,
            ts_ + temp_space_off,
            batch_size);
          omatcopy<data_type>('R', 'T',
            input_size, batch_size,
            1.0,
            ts_ + temp_space_off,
            batch_size,
            dx + t * x_size,
            input_size);
          omatcopy<data_type>('R', 'N',
            state_size, batch_size,
            1.0,
            ts_ + temp_space_off + input_size * batch_size,
            batch_size,
            ts_ + dhout_space_off
              + (t - 1) * h_nlayer_size,
            batch_size);
         } else if ((t == 0) && (l > 0)) {
          cblas_gemm<data_type>(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                      2 * state_size + 2, batch_size, 4 * state_size,
                      1.0,
                      w + woffset,
                      4 * state_size,
                      ts_ + dgates_space_off
                        + l * gates_size + t * gates_nlayer_size,
                      batch_size,
                      0.0,
                      ts_ + temp_space_off,
                      batch_size);
          omatcopy<data_type>('R', 'N',
            state_size, batch_size,
            1.0,
            ts_ + temp_space_off,
            batch_size,
            ts_ + dhout_space_off
              + (l - 1)*h_size + t*h_nlayer_size,
            batch_size);
          omatcopy<data_type>('R', 'T',
            state_size, batch_size,
            1.0,
            ts_ + temp_space_off + h_size,
            batch_size,
            dhx + l * h_size,
            state_size);
        } else {
          cblas_gemm<data_type>(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            input_size + state_size + 2, batch_size, 4 * state_size,
            1.0,
            w + woffset,
            4 * state_size,
            ts_ + dgates_space_off
              + l * gates_size + t * gates_nlayer_size,
            batch_size,
            0.0,
            ts_ + temp_space_off,
            batch_size);
          omatcopy<data_type>('R', 'T',
            input_size, batch_size,
            1.0,
            ts_ + temp_space_off,
            batch_size,
            dx + t * x_size,
            input_size);
          omatcopy<data_type>('R', 'T',
            state_size, batch_size,
            1.0,
            ts_ + temp_space_off + input_size * batch_size,
            batch_size,
            dhx + l * h_size,
            state_size);
        }

        }
    }


#endif
}

template struct gemm_rnn_fwd_t<data_type::f32>;
template struct gemm_rnn_bwd_t<data_type::f32>;

}
}
}