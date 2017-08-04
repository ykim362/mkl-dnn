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

#ifndef CONVOLUTION_PD_HPP
#define CONVOLUTION_PD_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "memory_pd.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {

template <bool with_relu>
struct _convolution_fwd_pd_t: public primitive_desc_t {
    typedef _convolution_fwd_pd_t base_class;
    typedef _convolution_fwd_pd_t hint_class;
    typedef typename utils::conditional<with_relu,
            convolution_relu_desc_t, convolution_desc_t>::type base_desc_t;
    static constexpr auto base_pkind =
        utils::conditional_v<with_relu, primitive_kind_t,
        primitive_kind::convolution_relu, primitive_kind::convolution>::value;

    _convolution_fwd_pd_t(mkldnn::impl::engine_t *engine,
            const base_desc_t *adesc, const _convolution_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, base_pkind), desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd) {}
    virtual ~_convolution_fwd_pd_t() {}

    const base_desc_t *desc() const { return &desc_; }
    inline const convolution_desc_t *cdesc() const { return &cdesc_(); }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }

    virtual const memory_pd_t *input_pd(int index = 0) const override {
        switch (index) {
        case 0: return src_pd();
        case 1: case 2: return weights_pd(index - 1);
        default: return nullptr;
        }
    }
    virtual const memory_pd_t *output_pd(int index = 0) const override
    { return index == 0 ? dst_pd() : nullptr; }

    virtual int n_inputs() const override { return 2 + with_bias(); }
    virtual int n_outputs() const override { return 1; }

    virtual status_t query(query_t what, int idx, void *result) const override
    {
        switch (what) {
        case pkind_traits<base_pkind>::query_d:
            *(const base_desc_t**)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common conv aux functions */

    inline int MB() const { return cdesc_().src_desc.dims[0]; }

    inline int IC() const { return cdesc_().src_desc.dims[1]; }
    inline int OC() const { return cdesc_().dst_desc.dims[1]; }
    inline int G() const
    { return with_groups() ? cdesc_().weights_desc.dims[0] : 1; }

    inline int IH() const { return cdesc_().src_desc.dims[2]; }
    inline int IW() const { return cdesc_().src_desc.dims[3]; }
    inline int OH() const { return cdesc_().dst_desc.dims[2]; }
    inline int OW() const { return cdesc_().dst_desc.dims[3]; }
    inline int KH() const
    { return cdesc_().weights_desc.dims[2 + with_groups()]; }
    inline int KW() const
    { return cdesc_().weights_desc.dims[3 + with_groups()]; }

    inline int KSH() const { return cdesc_().strides[0]; }
    inline int KSW() const { return cdesc_().strides[1]; }

    inline int KDH() const { return cdesc_().dilates[0]; }
    inline int KDW() const { return cdesc_().dilates[1]; }

    inline int padT() const { return cdesc_().padding[0][0]; }
    inline int padB() const { return cdesc_().padding[1][0]; }
    inline int padL() const { return cdesc_().padding[0][1]; }
    inline int padR() const { return cdesc_().padding[1][1]; }

    inline double negative_slope() const;

    inline bool with_bias() const
    { return !memory_desc_wrapper(cdesc_().bias_desc).is_zero(); }
    inline bool with_groups() const
    { return cdesc_().weights_desc.ndims == cdesc_().src_desc.ndims + 1; }

protected:
    base_desc_t desc_;
    const _convolution_fwd_pd_t *hint_fwd_pd_;

    inline const convolution_desc_t &cdesc_() const;

    virtual status_t init() = 0;
};

using convolution_fwd_pd_t = mkldnn::impl::_convolution_fwd_pd_t<false>;
using convolution_relu_fwd_pd_t = mkldnn::impl::_convolution_fwd_pd_t<true>;

template<> inline double convolution_fwd_pd_t::negative_slope() const
{ return 0.; }
template<> inline double convolution_relu_fwd_pd_t::negative_slope() const
{ return desc()->negative_slope; }

template<bool with_relu> inline const
convolution_desc_t &_convolution_fwd_pd_t<with_relu>::cdesc_() const
{ return desc_; }
template<>
inline const convolution_desc_t &convolution_relu_fwd_pd_t::cdesc_() const
{ return desc_.convolution_desc; }

struct convolution_bwd_data_pd_t: public primitive_desc_t {
    typedef convolution_bwd_data_pd_t base_class;
    typedef convolution_fwd_pd_t hint_class;
    static constexpr auto base_pkind = primitive_kind::convolution;

    convolution_bwd_data_pd_t(mkldnn::impl::engine_t *engine,
            const convolution_desc_t *adesc,
            const convolution_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, base_pkind), desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd) {}
    virtual ~convolution_bwd_data_pd_t() {}

    const convolution_desc_t *desc() const { return &desc_; }
    const convolution_desc_t *cdesc() const { return desc(); }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }

    virtual const memory_pd_t *input_pd(int index = 0) const override {
        switch (index) {
        case 0: return diff_dst_pd();
        case 1: return weights_pd(0);
        default: return nullptr;
        }
    }
    virtual const memory_pd_t *output_pd(int index = 0) const override
    { return index == 0 ? diff_src_pd() : nullptr; }

    virtual int n_inputs() const override { return 2; }
    virtual int n_outputs() const override { return 1; }

    virtual status_t query(query_t what, int idx, void *result) const override
    {
        switch (what) {
        case query::convolution_d:
            *(const convolution_desc_t**)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common conv aux functions */

    inline int MB() const { return desc_.diff_src_desc.dims[0]; }

    inline int IC() const { return desc_.diff_src_desc.dims[1]; }
    inline int OC() const { return desc_.diff_dst_desc.dims[1]; }
    inline int G() const
    { return with_groups() ? desc_.weights_desc.dims[0] : 1; }

    inline int IH() const { return desc_.diff_src_desc.dims[2]; }
    inline int IW() const { return desc_.diff_src_desc.dims[3]; }
    inline int OH() const { return desc_.diff_dst_desc.dims[2]; }
    inline int OW() const { return desc_.diff_dst_desc.dims[3]; }
    inline int KH() const
    { return desc_.weights_desc.dims[2 + with_groups()]; }
    inline int KW() const
    { return desc_.weights_desc.dims[3 + with_groups()]; }

    inline int KSH() const { return desc_.strides[0]; }
    inline int KSW() const { return desc_.strides[1]; }

    inline int padT() const { return desc_.padding[0][0]; }
    inline int padB() const { return desc_.padding[1][0]; }
    inline int padL() const { return desc_.padding[0][1]; }
    inline int padR() const { return desc_.padding[1][1]; }

    inline bool with_bias() const
    { return !memory_desc_wrapper(desc_.bias_desc).is_zero(); }
    inline bool with_groups() const
    { return desc_.weights_desc.ndims == desc_.diff_src_desc.ndims + 1; }

protected:
    convolution_desc_t desc_;
    const convolution_fwd_pd_t *hint_fwd_pd_;

    virtual status_t init() = 0;
};

struct convolution_bwd_weights_pd_t: public primitive_desc_t {
    typedef convolution_bwd_weights_pd_t base_class;
    typedef convolution_fwd_pd_t hint_class;
    static constexpr auto base_pkind = primitive_kind::convolution;

    convolution_bwd_weights_pd_t(mkldnn::impl::engine_t *engine,
            const convolution_desc_t *adesc,
            const convolution_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, base_pkind), desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd) {}
    virtual ~convolution_bwd_weights_pd_t() {}

    const convolution_desc_t *desc() const { return &desc_; }
    const convolution_desc_t *cdesc() const { return desc(); }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }

    virtual const memory_pd_t *input_pd(int index = 0) const override {
        switch (index) {
        case 0: return src_pd();
        case 1: return diff_dst_pd();
        default: return nullptr;
        }
    }
    virtual const memory_pd_t *output_pd(int index = 0) const override
    { switch (index) {
        case 0: return diff_weights_pd(0);
        case 1: with_bias() ? diff_weights_pd(1) : nullptr;
        default: return nullptr;
        }
    }

    virtual int n_inputs() const override { return 2; }
    virtual int n_outputs() const override { return 1 + with_bias(); }

    virtual status_t query(query_t what, int idx, void *result) const override
    {
        switch (what) {
        case query::convolution_d:
            *(const convolution_desc_t**)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common conv aux functions */

    inline int MB() const { return desc_.src_desc.dims[0]; }

    inline int IC() const { return desc_.src_desc.dims[1]; }
    inline int OC() const { return desc_.diff_dst_desc.dims[1]; }
    inline int G() const
    { return with_groups() ? desc_.diff_weights_desc.dims[0] : 1; }

    inline int IH() const { return desc_.src_desc.dims[2]; }
    inline int IW() const { return desc_.src_desc.dims[3]; }
    inline int OH() const { return desc_.diff_dst_desc.dims[2]; }
    inline int OW() const { return desc_.diff_dst_desc.dims[3]; }
    inline int KH() const
    { return desc_.diff_weights_desc.dims[2 + with_groups()]; }
    inline int KW() const
    { return desc_.diff_weights_desc.dims[3 + with_groups()]; }

    inline int KSH() const { return desc_.strides[0]; }
    inline int KSW() const { return desc_.strides[1]; }

    inline int padT() const { return desc_.padding[0][0]; }
    inline int padB() const { return desc_.padding[1][0]; }
    inline int padL() const { return desc_.padding[0][1]; }
    inline int padR() const { return desc_.padding[1][1]; }

    inline bool with_bias() const
    { return !memory_desc_wrapper(desc_.diff_bias_desc).is_zero(); }
    inline bool with_groups() const
    { return desc_.diff_weights_desc.ndims == desc_.diff_dst_desc.ndims + 1; }

protected:
    convolution_desc_t desc_;
    const convolution_fwd_pd_t *hint_fwd_pd_;

    virtual status_t init() = 0;
};

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
