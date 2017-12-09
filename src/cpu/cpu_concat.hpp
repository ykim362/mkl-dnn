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

#ifndef CPU_CONCAT_HPP
#define CPU_CONCAT_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "memory_pd.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "cpu_memory.hpp"
#include "cpu_primitive.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

#define DECLARE_CPU_CONCAT_PD_T(...) \
    static status_t create(concat_pd_t **concat_pd, \
            const memory_desc_t *output_d, int n, int concat_dim, \
            const memory_pd_t **input_pds, const primitive_attr_t *attr) { \
        auto _pd = new pd_t(output_d, n, concat_dim, \
                (const cpu_memory_pd_t **)input_pds, attr); \
        if (_pd == nullptr) return out_of_memory; \
        if (_pd->init() != success) { delete _pd; return unimplemented; } \
        return safe_ptr_assign<concat_pd_t>(*concat_pd, _pd); \
    } \
    virtual status_t create_primitive(primitive_t **primitive, \
            const primitive_at_t *inputs, \
            const primitive_t **outputs) const override { \
        primitive_t::input_vector ins(inputs, inputs + n_); \
        primitive_t::output_vector outs(outputs, outputs + 1); \
        return safe_ptr_assign<primitive_t>(*primitive, \
                new (__VA_ARGS__)(this, ins, outs)); \
    } \
    virtual pd_t *clone() const override { return nullptr; }

struct cpu_concat_pd_t: public concat_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_concat_pd_t(const memory_desc_t *output_d, int n,
            int concat_dim, const cpu_memory_pd_t **input_pds,
            const primitive_attr_t *attr)
        : concat_pd_t(input_pds[0]->engine(), n, concat_dim, attr),
        dst_pd_(input_pds[0]->engine()) {
            for (int i = 0; i < n_; ++i)
                src_pds_.push_back(*input_pds[i]); /* make a copy */
            dst_pd_ = cpu_memory_pd_t(input_pds[0]->engine(), output_d);
        }
    cpu_concat_pd_t(const cpu_concat_pd_t &rhs)
        : concat_pd_t(rhs), src_pds_(rhs.src_pds_)
        , src_image_pds_(rhs.src_image_pds_)
        , dst_pd_(rhs.dst_pd_) {}

    virtual ~cpu_concat_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override
    { return index < this->n_ ? &src_pds_[index] : nullptr; }
    virtual const cpu_memory_pd_t *src_image_pd(int index = 0) const
    { return index < this->n_ ? &src_image_pds_[index] : nullptr; }
    virtual const cpu_memory_pd_t *dst_pd(int index = 0) const override
    { return index == 0 ? &dst_pd_ : nullptr; }

protected:
    nstl::vector<cpu_memory_pd_t> src_pds_;
    nstl::vector<cpu_memory_pd_t> src_image_pds_;
    cpu_memory_pd_t dst_pd_;

    virtual status_t init() {
        bool ok = true
            && set_default_params() == success
            && attr()->has_default_values();
        if (!ok) return unimplemented;

        const int ndims = dst_pd_.desc()->ndims;
        int current_concat_dim_offset = 0;
        for (int i = 0; i < n_; ++i) {
            const int dim = src_pds_[i].desc()->dims[concat_dim_];
            dims_t dims, offsets = {};
            utils::array_copy(dims, dst_pd_.desc()->dims, ndims);
            dims[concat_dim_] = dim;
            offsets[concat_dim_] = current_concat_dim_offset;

            cpu_view_t::pd_t v_pd(src_pds_[i].engine(), &dst_pd_, dims,
                    offsets);
            src_image_pds_.push_back(*v_pd.dst_pd());
            current_concat_dim_offset += dim;
        }

        return success;
    }

    virtual status_t set_default_params() {
        if (dst_pd_.desc()->format == memory_format::any) {
            /* the stupidest ever heuristics */
            memory_format_t out_fmt = dst_pd_.desc()->format;
            for (int i = 0; i < n_; ++i) {
                out_fmt = nstl::max(out_fmt, src_pds_[i].desc()->format);
            }
            CHECK(dst_pd_.set_format(out_fmt));
        }

        return success;
    }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
