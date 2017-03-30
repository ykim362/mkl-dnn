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

#ifndef CPU_CONVOLUTION_FWD_PD_HPP
#define CPU_CONVOLUTION_FWD_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_memory.hpp"
#include "cpu_primitive.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <bool with_relu>
struct _cpu_convolution_fwd_pd_t: public _convolution_fwd_pd_t<with_relu> {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    _cpu_convolution_fwd_pd_t(engine_t *engine,
            const typename _cpu_convolution_fwd_pd_t::base_desc_t *adesc,
            const typename _cpu_convolution_fwd_pd_t::base_class *hint_fwd_pd)
        : _convolution_fwd_pd_t<with_relu>(engine, adesc, hint_fwd_pd)
        , src_pd_(this->engine_, &this->cdesc_().src_desc)
        , dst_pd_(this->engine_, &this->cdesc_().dst_desc)
        , weights_pd_(this->engine_, &this->cdesc_().weights_desc)
        , bias_pd_(this->engine_, &this->cdesc_().bias_desc) {}
    virtual ~_cpu_convolution_fwd_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override
    { return index == 0 ? &src_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *dst_pd(int index = 0) const override
    { return index == 0 ? &dst_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override {
        if (index == 0) return &weights_pd_;
        if (index == 1 && this->with_bias()) return &bias_pd_;
        return nullptr;
    }

protected:
    cpu_memory_pd_t src_pd_, dst_pd_;
    cpu_memory_pd_t weights_pd_, bias_pd_;

    virtual status_t set_default_params() {
        using namespace memory_format;
        if (src_pd_.desc()->format == any)
            CHECK(src_pd_.set_format(nchw));
        if (dst_pd_.desc()->format == any)
            CHECK(dst_pd_.set_format(src_pd_.desc()->format));
        if (weights_pd_.desc()->format == any)
            CHECK(weights_pd_.set_format(this->with_groups() ? goihw : oihw));
        if (bias_pd_.desc()->format == any)
            CHECK(bias_pd_.set_format(x));
        return status::success;
    }
};

using cpu_convolution_fwd_pd_t = _cpu_convolution_fwd_pd_t<false>;
using cpu_convolution_relu_fwd_pd_t = _cpu_convolution_fwd_pd_t<true>;

struct cpu_convolution_bwd_data_pd_t: public convolution_bwd_data_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_convolution_bwd_data_pd_t(engine_t *engine,
            const convolution_desc_t *adesc,
            const convolution_fwd_pd_t *hint_fwd_pd)
        : convolution_bwd_data_pd_t(engine, adesc, hint_fwd_pd)
        , diff_src_pd_(this->engine_, &this->desc_.diff_src_desc)
        , diff_dst_pd_(this->engine_, &this->desc_.diff_dst_desc)
        , weights_pd_(this->engine_, &this->desc_.weights_desc) {}
    virtual ~cpu_convolution_bwd_data_pd_t() {}

    virtual const cpu_memory_pd_t *diff_src_pd(int index = 0) const override
    { return index == 0 ? &diff_src_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *diff_dst_pd(int index = 0) const override
    { return index == 0 ? &diff_dst_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override
    { return index == 0 ? &weights_pd_ : nullptr; }

protected:
    cpu_memory_pd_t diff_src_pd_, diff_dst_pd_;
    cpu_memory_pd_t weights_pd_;

    virtual status_t set_default_params() {
        using namespace memory_format;
        if (diff_src_pd_.desc()->format == any)
            CHECK(diff_src_pd_.set_format(nchw));
        if (diff_dst_pd_.desc()->format == any)
            CHECK(diff_dst_pd_.set_format(diff_src_pd_.desc()->format));
        if (weights_pd_.desc()->format == any)
            CHECK(weights_pd_.set_format(this->with_groups() ? goihw : oihw));
        return status::success;
    }
};

struct cpu_convolution_bwd_weights_pd_t: public convolution_bwd_weights_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_convolution_bwd_weights_pd_t(engine_t *engine,
            const convolution_desc_t *adesc,
            const convolution_fwd_pd_t *hint_fwd_pd)
        : convolution_bwd_weights_pd_t(engine, adesc, hint_fwd_pd)
        , src_pd_(this->engine_, &this->desc_.src_desc)
        , diff_dst_pd_(this->engine_, &this->desc_.diff_dst_desc)
        , diff_weights_pd_(this->engine_, &this->desc_.diff_weights_desc)
        , diff_bias_pd_(this->engine_, &this->desc_.diff_bias_desc) {}
    virtual ~cpu_convolution_bwd_weights_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override
    { return index == 0 ? &src_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *diff_dst_pd(int index = 0) const override
    { return index == 0 ? &diff_dst_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *diff_weights_pd(int index = 0) const
        override {
            if (index == 0) return &diff_weights_pd_;
            if (index == 1 && this->with_bias()) return &diff_bias_pd_;
            return  nullptr;
        }

protected:
    cpu_memory_pd_t src_pd_;
    cpu_memory_pd_t diff_dst_pd_;
    cpu_memory_pd_t diff_weights_pd_, diff_bias_pd_;

    virtual status_t set_default_params() {
        using namespace memory_format;
        if (src_pd_.desc()->format == any)
            CHECK(src_pd_.set_format(nchw));
        if (diff_dst_pd_.desc()->format == any)
            CHECK(diff_dst_pd_.set_format(nchw));
        if (diff_weights_pd_.desc()->format == any)
            CHECK(diff_weights_pd_.set_format(
                        this->with_groups() ? goihw : oihw));
        if (diff_bias_pd_.desc()->format == any)
            CHECK(diff_bias_pd_.set_format(x));
        return status::success;
    }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
