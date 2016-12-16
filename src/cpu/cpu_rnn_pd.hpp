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

struct cpu_rnn_fwd_pd_t: public rnn_fwd_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_rnn_fwd_pd_t(engine_t *engine, const rnn_desc_t *adesc,
            const rnn_fwd_pd_t *hint_fwd_pd)
        : rnn_fwd_pd_t(engine, adesc, hint_fwd_pd)
        , x_pd_(this->engine_, &desc_.x_desc)
        , hx_pd_(this->engine_, &desc_.hx_desc)
        , cx_pd_(this->engine_, &desc_.hx_desc)
        , weights_pd_(this->engine_, &desc_.weights_desc)
        , y_pd_(this->engine_, &desc_.y_desc)
        , hy_pd_(this->engine_, &desc_.hx_desc)
        , cy_pd_(this->engine_, &desc_.hx_desc)
        , ws_pd_(this->engine_) {}
    virtual ~cpu_rnn_fwd_pd_t() {}

    virtual const cpu_memory_pd_t *x_pd(int index = 0) const override
    { return index == 0 ? &x_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *hx_pd(int index = 0) const override
    { return index == 0 ? &hx_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *cx_pd(int index = 0) const override
    { return index == 0 ? &cx_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override
    { return index == 0 ? &weights_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *y_pd(int index = 0) const override
    { return index == 0 ? &y_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *hy_pd(int index = 0) const override
    { return index == 0 ? &hy_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *cy_pd(int index = 0) const override
    { return index == 0 ? &cy_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *workspace_pd(int index = 0) const override
    { return index == 0 ? &ws_pd_ : nullptr; }

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
};

struct cpu_rnn_bwd_pd_t: public rnn_bwd_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_rnn_bwd_pd_t(engine_t *engine, const rnn_desc_t *adesc,
            const rnn_fwd_pd_t *hint_fwd_pd)
        : rnn_bwd_pd_t(engine, adesc, hint_fwd_pd)
        , x_pd_(this->engine_, &desc_.x_desc)
        , hx_pd_(this->engine_, &desc_.hx_desc)
        , cx_pd_(this->engine_, &desc_.hx_desc)
        , dx_pd_(this->engine_, &desc_.x_desc)
        , dhx_pd_(this->engine_, &desc_.hx_desc)
        , dcx_pd_(this->engine_, &desc_.hx_desc)
        , dy_pd_(this->engine_, &desc_.y_desc)
        , dhy_pd_(this->engine_, &desc_.hx_desc)
        , dcy_pd_(this->engine_, &desc_.hx_desc)
        , weights_pd_(this->engine_, &desc_.weights_desc)
        , diff_weights_pd_(this->engine_, &desc_.weights_desc)
        , ws_pd_(this->engine_) {}
    virtual ~cpu_rnn_bwd_pd_t() {}

    virtual const cpu_memory_pd_t *x_pd(int index = 0) const override
    { return index == 0 ? &x_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *hx_pd(int index = 0) const override
    { return index == 0 ? &hx_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *cx_pd(int index = 0) const override
    { return index == 0 ? &cx_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *dx_pd(int index = 0) const override
    { return index == 0 ? &dx_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *dhx_pd(int index = 0) const override
    { return index == 0 ? &dhx_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *dcx_pd(int index = 0) const override
    { return index == 0 ? &dcx_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *dy_pd(int index = 0) const override
    { return index == 0 ? &dy_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *dhy_pd(int index = 0) const override
    { return index == 0 ? &dhy_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *dcy_pd(int index = 0) const override
    { return index == 0 ? &dcy_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override
    { return index == 0 ? &weights_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *diff_weights_pd(int index = 0) const override
    { return index == 0 ? &diff_weights_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *workspace_pd(int index = 0) const override
    { return index == 0 ? &ws_pd_ : nullptr; }

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
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
