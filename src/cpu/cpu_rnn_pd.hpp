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

#ifndef CPU_RNN_FWD_PD_HPP
#define CPU_RNN_FWD_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "rnn_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_memory.hpp"
#include "cpu_primitive.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct cpu_rnn_fwd_pd_t : public rnn_fwd_pd_t {
  using cpu_memory_pd_t = cpu_memory_t::pd_t;

  cpu_rnn_fwd_pd_t(engine_t *engine, const rnn_desc_t *adesc,
                   const rnn_fwd_pd_t *hint_fwd_pd)
      : rnn_fwd_pd_t(engine, adesc, hint_fwd_pd),
        x_pd_(this->engine_, &desc_.x_desc),
        hx_pd_(this->engine_, &desc_.hx_desc),
        cx_pd_(this->engine_, &desc_.hx_desc),
        weights_pd_(this->engine_, &desc_.weights_desc),
        y_pd_(this->engine_, &desc_.y_desc),
        hy_pd_(this->engine_),
        cy_pd_(this->engine_), ws_pd_(this->engine_) {}
  virtual ~cpu_rnn_fwd_pd_t() {}

  virtual const cpu_memory_pd_t *src_pd(int index = 0) const override {
    switch (index) {
    case 0:
      return &x_pd_;
    case 1:
      return &hx_pd_;
    case 2:
      return &cx_pd_;
    default:
      return nullptr;
    }
  }
  virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override {
    return index == 0 ? &weights_pd_ : nullptr;
  }
  virtual const cpu_memory_pd_t *dst_pd(int index = 0) const override {
    switch (index) {
    case 0:
      return &y_pd_;
    case 1:
      return (index == 1 && !hy_pd_.is_zero()) ? &hy_pd_ : nullptr;
    case 2:
      return (index == 2 && !cy_pd_.is_zero()) ? &cy_pd_ : nullptr;
    default:
      return nullptr;
    }
  }
  virtual const cpu_memory_pd_t *workspace_pd(int index = 0) const override {
    return (index == 0 && !ws_pd_.is_zero()) ? &ws_pd_ : nullptr;
  }

protected:
  cpu_memory_pd_t x_pd_;
  cpu_memory_pd_t hx_pd_;
  cpu_memory_pd_t cx_pd_;
  cpu_memory_pd_t weights_pd_;
  cpu_memory_pd_t y_pd_;
  cpu_memory_pd_t hy_pd_;
  cpu_memory_pd_t cy_pd_;
  cpu_memory_pd_t ws_pd_;

  virtual status_t init() = 0;

  virtual status_t set_default_params() {
    using namespace memory_format;
    if (x_pd_.desc()->format == any)
      CHECK(x_pd_.set_format(rnx));
    if (hx_pd_.desc()->format == any)
      CHECK(hx_pd_.set_format(rnx));
    if (cx_pd_.desc()->format == any)
      CHECK(cx_pd_.set_format(rnx));
    if (y_pd_.desc()->format == any)
      CHECK(y_pd_.set_format(rnx));
    if (!hy_pd_.is_zero())
      if (hy_pd_.desc()->format == any)
        CHECK(hy_pd_.set_format(rnx));
    if (!cy_pd_.is_zero())
      if (cy_pd_.desc()->format == any)
        CHECK(cy_pd_.set_format(rnx));

    return status::success;
  }
};

struct cpu_rnn_bwd_pd_t : public rnn_bwd_pd_t {
  using cpu_memory_pd_t = cpu_memory_t::pd_t;

  cpu_rnn_bwd_pd_t(engine_t *engine, const rnn_desc_t *adesc,
                   const rnn_fwd_pd_t *hint_fwd_pd)
      : rnn_bwd_pd_t(engine, adesc, hint_fwd_pd),
        x_pd_(this->engine_, &desc_.x_desc),
        hx_pd_(this->engine_, &desc_.hx_desc),
        cx_pd_(this->engine_, &desc_.hx_desc),
        dx_pd_(this->engine_, &desc_.x_desc),
        dhx_pd_(this->engine_, &desc_.hx_desc),
        dcx_pd_(this->engine_, &desc_.hx_desc),
        dy_pd_(this->engine_, &desc_.y_desc),
        dhy_pd_(this->engine_),
        dcy_pd_(this->engine_),
        weights_pd_(this->engine_, &desc_.weights_desc),
        diff_weights_pd_(this->engine_, &desc_.weights_desc),
        ws_pd_(this->engine_) {}
  virtual ~cpu_rnn_bwd_pd_t() {}

  virtual const cpu_memory_pd_t *src_pd(int index = 0) const override {
    switch (index) {
    case 0:
      return &x_pd_;
    case 1:
      return &hx_pd_;
    case 2:
      return &cx_pd_;
    default:
      return nullptr;
    }
  }
  virtual const cpu_memory_pd_t *diff_dst_pd(int index = 0) const override {
    switch (index) {
    case 0:
      return &dy_pd_;
    case 1:
      return (index == 1 && !dhy_pd_.is_zero()) ? &dhy_pd_ : nullptr;
    case 2:
      return (index == 2 && !dcy_pd_.is_zero()) ? &dcy_pd_ : nullptr;;
    default:
      return nullptr;
    }
  }
  virtual const cpu_memory_pd_t *diff_src_pd(int index = 0) const override {
    switch (index) {
    case 0:
      return &dx_pd_;
    case 1:
      return &dhx_pd_;
    case 2:
      return &dcx_pd_;
    default:
      return nullptr;
    }
  }
  virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override {
    return index == 0 ? &weights_pd_ : nullptr;
  }
  virtual const cpu_memory_pd_t *diff_weights_pd(int index = 0) const override {
    return index == 0 ? &diff_weights_pd_ : nullptr;
  }
  virtual const cpu_memory_pd_t *workspace_pd(int index = 0) const override {
    return index == 0 ? &ws_pd_ : nullptr;
  }

protected:
  cpu_memory_pd_t x_pd_;
  cpu_memory_pd_t hx_pd_;
  cpu_memory_pd_t cx_pd_;
  cpu_memory_pd_t dx_pd_;
  cpu_memory_pd_t dhx_pd_;
  cpu_memory_pd_t dcx_pd_;
  cpu_memory_pd_t dy_pd_;
  cpu_memory_pd_t dhy_pd_;
  cpu_memory_pd_t dcy_pd_;
  cpu_memory_pd_t weights_pd_;
  cpu_memory_pd_t diff_weights_pd_;
  cpu_memory_pd_t ws_pd_;

  virtual status_t init() = 0;
  virtual status_t set_default_params() {
    using namespace memory_format;
    if (x_pd_.desc()->format == any)
      CHECK(x_pd_.set_format(rnx));
    if (hx_pd_.desc()->format == any)
      CHECK(hx_pd_.set_format(rnx));
    if (cx_pd_.desc()->format == any)
      CHECK(cx_pd_.set_format(rnx));
    if (dx_pd_.desc()->format == any)
      CHECK(dx_pd_.set_format(rnx));
    if (dhx_pd_.desc()->format == any)
      CHECK(dhx_pd_.set_format(rnx));
    if (dcx_pd_.desc()->format == any)
      CHECK(dcx_pd_.set_format(rnx));
    if (dy_pd_.desc()->format == any)
      CHECK(dy_pd_.set_format(rnx));
    if (!dhy_pd_.is_zero())
      if (dhy_pd_.desc()->format == any)
        CHECK(dhy_pd_.set_format(rnx));
    if (!dcy_pd_.is_zero())
      if (dcy_pd_.desc()->format == any)
        CHECK(dcy_pd_.set_format(rnx));

    return status::success;
  }
};
}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
