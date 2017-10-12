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

#ifndef CPU_JIT_UNI_POOLING_HPP
#define CPU_JIT_UNI_POOLING_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_pooling_pd.hpp"
#include "cpu_engine.hpp"
#include "jit_uni_pool_kernel_f32.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_pooling_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_pooling_fwd_pd_t {
        pd_t(engine_t *engine, const pooling_desc_t *adesc,
                const pooling_fwd_pd_t *hint_fwd_pd)
            : cpu_pooling_fwd_pd_t(engine, adesc, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(jit_uni_pooling_fwd_t<isa>);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace alg_kind;
            using namespace utils;
            auto desired_fmt = isa == avx512_common
                ? memory_format::nChw16c
                : memory_format::nChw8c;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && mayiuse(isa)
                && set_default_params() == status::success
                && one_of(desc()->prop_kind, forward_training,
                        forward_inference)
                && one_of(desc()->alg_kind, pooling_max,
                        pooling_avg_include_padding,
                        pooling_avg_exclude_padding)
                && everyone_is(data_type::f32, src_pd()->desc()->data_type,
                        dst_pd()->desc()->data_type)
                && everyone_is(desired_fmt, src_pd()->desc()->format,
                        dst_pd()->desc()->format);
            if (!ok) return status::unimplemented;

            bool is_training = desc_.prop_kind == forward_training;
            if (desc()->alg_kind == pooling_max && is_training) {
                auto indices_desc = *dst_pd()->desc();
                indices_desc.data_type = pooling_index_data_type(desc());
                ws_pd_ = cpu_memory_t::pd_t(engine_, &indices_desc);
            }

            return jit_uni_pool_kernel_f32<isa>::init_conf(jpp_, desc_,
                    src_pd_.desc(), dst_pd_.desc());
        }

        jit_pool_conf_t jpp_;

    protected:
        virtual status_t set_default_params() override {
            auto desired_fmt = isa == avx512_common
                ? memory_format::nChw16c
                : memory_format::nChw8c;
            if (dst_pd_.desc()->format == memory_format::any)
               CHECK(dst_pd_.set_format(desired_fmt));
            return status::success;
        }
    };

    jit_uni_pooling_fwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
    { kernel_ = new jit_uni_pool_kernel_f32<isa>(conf_.jpp_); }

    ~jit_uni_pooling_fwd_t() { delete kernel_; }

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;
    jit_uni_pool_kernel_f32<isa> *kernel_;
};

template <cpu_isa_t isa>
struct jit_uni_pooling_bwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_pooling_bwd_pd_t {
        pd_t(engine_t *engine, const pooling_desc_t *adesc,
                const pooling_fwd_pd_t *hint_fwd_pd)
            : cpu_pooling_bwd_pd_t(engine, adesc, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(jit_uni_pooling_bwd_t<isa>);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace alg_kind;
            using namespace utils;

            auto desired_fmt = isa == avx512_common
                ? memory_format::nChw16c
                : memory_format::nChw8c;

            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && mayiuse(isa)
                && set_default_params() == status::success
                && one_of(desc()->prop_kind, backward, backward_data)
                && one_of(desc()->alg_kind, pooling_max,
                        pooling_avg_include_padding,
                        pooling_avg_exclude_padding)
                && everyone_is(desired_fmt, diff_src_pd()->desc()->format,
                        diff_dst_pd()->desc()->format)
                && everyone_is(data_type::f32, diff_src_pd()->desc()->data_type,
                        diff_dst_pd()->desc()->data_type)
                && utils::implication(desc()->alg_kind == pooling_max,
                        hint_fwd_pd_ && hint_fwd_pd_->workspace_pd()
                        && hint_fwd_pd_->workspace_pd()->desc()->format
                                == desired_fmt);
            if (!ok) return status::unimplemented;

            if (desc()->alg_kind == pooling_max)
                ws_pd_ = *(cpu_memory_t::pd_t*)hint_fwd_pd_->workspace_pd();

            return jit_uni_pool_kernel_f32<isa>::init_conf(jpp_, desc_,
                    diff_src_pd_.desc(), diff_dst_pd_.desc());
        }

        jit_pool_conf_t jpp_;

    protected:
        virtual status_t set_default_params() override {
            auto desired_fmt = isa == avx512_common
                ? memory_format::nChw16c
                : memory_format::nChw8c;
            if (diff_src_pd_.desc()->format == memory_format::any)
               CHECK(diff_src_pd_.set_format(desired_fmt));
           return status::success;
        }
    };

    jit_uni_pooling_bwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
    { kernel_ = new jit_uni_pool_kernel_f32<isa>(conf_.jpp_); }

    ~jit_uni_pooling_bwd_t() { delete kernel_; }

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) {
        execute_backward();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward();
    pd_t conf_;
    jit_uni_pool_kernel_f32<isa> *kernel_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
