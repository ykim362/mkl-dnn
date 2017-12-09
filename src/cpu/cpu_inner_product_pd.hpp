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

#ifndef CPU_INNER_PRODUCT_FWD_PD_HPP
#define CPU_INNER_PRODUCT_FWD_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "inner_product_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_memory.hpp"
#include "cpu_primitive.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct cpu_inner_product_fwd_pd_t: public inner_product_fwd_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_inner_product_fwd_pd_t(engine_t *engine,
            const inner_product_desc_t *adesc,
            const primitive_attr_t *attr,
            const inner_product_fwd_pd_t *hint_fwd_pd)
        : inner_product_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        , src_pd_(engine_, &desc_.src_desc), dst_pd_(engine_, &desc_.dst_desc)
        , weights_pd_(engine_, &desc_.weights_desc)
        , bias_pd_(engine_, &desc_.bias_desc) {}
    virtual ~cpu_inner_product_fwd_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override
    { return index == 0 ? &src_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *dst_pd(int index = 0) const override
    { return index == 0 ? &dst_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override {
        if (index == 0) return &weights_pd_;
        if (index == 1 && with_bias()) return &bias_pd_;
        return nullptr;
    }

protected:
    cpu_memory_pd_t src_pd_, dst_pd_;
    cpu_memory_pd_t weights_pd_, bias_pd_;

    virtual status_t set_default_params() {
        using namespace memory_format;
        if (src_pd_.desc()->format == any)
            CHECK(src_pd_.set_format(ndims() == 4 ? nchw : nc));
        if (dst_pd_.desc()->format == any)
            CHECK(dst_pd_.set_format(nc));
        if (weights_pd_.desc()->format == any)
            CHECK(weights_pd_.set_format(ndims() == 4 ? oihw : oi));
        if (bias_pd_.desc()->format == any)
            CHECK(bias_pd_.set_format(x));
        return status::success;
    }
};

struct cpu_inner_product_bwd_data_pd_t: public inner_product_bwd_data_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_inner_product_bwd_data_pd_t(engine_t *engine,
            const inner_product_desc_t *adesc,
            const primitive_attr_t *attr,
            const inner_product_fwd_pd_t *hint_fwd_pd)
        : inner_product_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
        , diff_src_pd_(engine_, &desc_.diff_src_desc)
        , diff_dst_pd_(engine_, &desc_.diff_dst_desc)
        , weights_pd_(engine_, &desc_.weights_desc) {}
    virtual ~cpu_inner_product_bwd_data_pd_t() {}

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
            CHECK(diff_src_pd_.set_format(ndims() == 4 ? nchw : nc));
        if (diff_dst_pd_.desc()->format == any)
            CHECK(diff_dst_pd_.set_format(nc));
        if (weights_pd_.desc()->format == any)
            CHECK(weights_pd_.set_format(ndims() == 4 ? oihw : oi));
        return status::success;
    }
};

struct cpu_inner_product_bwd_weights_pd_t: public inner_product_bwd_weights_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_inner_product_bwd_weights_pd_t(engine_t *engine,
            const inner_product_desc_t *adesc,
            const primitive_attr_t *attr,
            const inner_product_fwd_pd_t *hint_fwd_pd)
        : inner_product_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
        , src_pd_(engine_, &desc_.src_desc)
        , diff_dst_pd_(engine_, &desc_.diff_dst_desc)
        , diff_weights_pd_(engine_, &desc_.diff_weights_desc)
        , diff_bias_pd_(engine_, &desc_.diff_bias_desc) {}
    virtual ~cpu_inner_product_bwd_weights_pd_t() {}

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
            CHECK(src_pd_.set_format(ndims() == 4 ? nchw : nc));
        if (diff_dst_pd_.desc()->format == any)
            CHECK(diff_dst_pd_.set_format(nc));
        if (diff_weights_pd_.desc()->format == any)
            CHECK(diff_weights_pd_.set_format(ndims() == 4 ? oihw : oi));
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
