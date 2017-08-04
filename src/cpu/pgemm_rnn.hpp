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

#ifndef CPU_PGEMM_RNN_FWD_HPP
#define CPU_PGEMM_RNN_FWD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_engine.hpp"
#include "cpu_rnn_pd.hpp"
#include "type_helpers.hpp"
#include "cpu_math_util.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct pgemm_rnn_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_rnn_fwd_pd_t {
        pd_t(engine_t *engine, const rnn_desc_t *adesc,
                const rnn_fwd_pd_t *hint_fwd_pd)
            : cpu_rnn_fwd_pd_t(engine, adesc, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(pgemm_rnn_fwd_t);

        virtual status_t init() override {
#ifdef USE_MKL
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(desc()->prop_kind, prop_kind::forward_training,
                        prop_kind::forward_inference)
                && utils::one_of(desc()->alg_kind, alg_kind::rnn_relu, alg_kind::rnn_tanh,
                        alg_kind::rnn_lstm)
                && utils::everyone_is(data_type, desc()->x_desc.data_type,
                        desc()->hx_desc.data_type, desc()->y_desc.data_type,
                        desc()->weights_desc.data_type);
            if (!ok) return status::unimplemented;

            if (desc_.prop_kind == prop_kind::forward_training) {
                auto ws_size = static_cast<int>(workspace_size());
                memory_desc_t ws_d;
                mkldnn_memory_desc_init(&ws_d, 1, &ws_size, data_type,
                        memory_format::x);
                ws_pd_ = cpu_memory_pd_t(this->engine(), &ws_d);
            }

            return status::success;
#else
            return status::unimplemented;
#endif // USE_MKL
        }
    };

    pgemm_rnn_fwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd), ts_(nullptr)
    {
        auto insize = conf_.input_size() > conf_.hidden_size()
            ? conf_.input_size() : conf_.hidden_size();
        auto tmp = insize + conf_.hidden_size() + 2;
        auto ts_size_ = tmp * conf_.batch();
        if (conf_.desc()->prop_kind != prop_kind::forward_training)
            ts_size_ += conf_.workspace_size();
        ts_ = (data_t *)malloc(ts_size_ * sizeof(data_t), 64);
        size_t total_layers = conf_.layers() * conf_.direction();
        weights_pack_ = new data_t *[total_layers];
        size_t rl, in_size, m;
        for (size_t ii = 0; ii < total_layers; ++ii)
        {
            rl = ii % conf_.layers();
            in_size = (rl == 0) ? conf_.input_size() : conf_.hidden_size();
            m = conf_.hidden_size();
            if (conf_.desc()->alg_kind == alg_kind::rnn_lstm) m *= 4;
            weights_pack_[ii]
                    = cpu_blas::cblas_gemm_alloc<data_type>(CblasAMatrix,
                            m, conf_.batch(), (in_size + conf_.hidden_size() + 2));
        }
    }

    ~pgemm_rnn_fwd_t()
    {
        if (weights_pack_)
        {
            size_t total_layers = conf_.layers() * conf_.direction();
            for (size_t ii = 0; ii < total_layers; ++ii)
                cpu_blas::cblas_gemm_free<data_type>(weights_pack_[ii]);

            delete[] weights_pack_;
        }
        if (ts_) free(ts_);
    }

    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e)
    {
        switch (conf_.desc()->prop_kind) {
        case prop_kind::forward_training:
        case prop_kind::forward_inference: execute_forward(); break;
        default: assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;
    data_t *ts_;
    data_t **weights_pack_;
};

template <impl::data_type_t data_type>
struct pgemm_rnn_bwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_rnn_bwd_pd_t {
        pd_t(engine_t *engine, const rnn_desc_t *adesc,
                const rnn_fwd_pd_t *hint_fwd_pd)
            : cpu_rnn_bwd_pd_t(engine, adesc, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(pgemm_rnn_bwd_t);

        virtual status_t init() override {
#ifdef USE_MKL
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(desc()->prop_kind, prop_kind::backward)
                && utils::one_of(desc()->alg_kind, alg_kind::rnn_relu, alg_kind::rnn_tanh,
                        alg_kind::rnn_lstm)
                && utils::everyone_is(data_type, desc()->x_desc.data_type,
                        desc()->hx_desc.data_type, desc()->y_desc.data_type,
                        desc()->weights_desc.data_type);
            if (!ok) return status::unimplemented;

            bool stats_ok = true
                && hint_fwd_pd_ != nullptr
                && hint_fwd_pd_->layers() == desc()->num_layers
                && hint_fwd_pd_->tau() == desc()->num_seqs
                && hint_fwd_pd_->direction() == desc()->direction
                && hint_fwd_pd_->hidden_size() == desc()->num_states
                && hint_fwd_pd_->input_size() == desc()->x_desc.dims[2]
                && hint_fwd_pd_->batch() == desc()->x_desc.dims[1]
                && hint_fwd_pd_->input_pd(1)->desc()->format
                    == memory_format::rnx
                && hint_fwd_pd_->input_pd(1)->desc()->ndims == 3
                && hint_fwd_pd_->input_pd(1)->desc()->data_type
                    == data_type;
            if (!stats_ok) return status::unimplemented;

            auto ws_size = static_cast<int>(workspace_size());
            memory_desc_t ws_d;
            mkldnn_memory_desc_init(
                    &ws_d, 1, &ws_size, data_type, memory_format::x);
            ws_pd_ = cpu_memory_pd_t(this->engine(), &ws_d);

            return status::success;
#else
            return status::unimplemented;
#endif // USE_MKL
        }
    };

    pgemm_rnn_bwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd), ts_(nullptr)
    {
        auto bsize = conf_.input_size() > conf_.hidden_size()
            ? conf_.input_size() : conf_.hidden_size();
        auto tmp1 = bsize + conf_.hidden_size() + 2;
        auto tmp2 = conf_.hidden_size() * conf_.layers();
        auto tmp = tmp1 > tmp2 ? tmp1 : tmp2;
        auto ts_size_ = tmp * conf_.batch() + conf_.gates_space_size()
                + 2 * conf_.h_space_size();
        ts_ = (data_t *)malloc(ts_size_ * sizeof(data_t), 64);
        size_t total_layers = conf_.layers() * conf_.direction();
        weights_pack_ = new data_t *[total_layers];
        size_t rl, in_size, m;
        for (size_t ii = 0; ii < total_layers; ++ii)
        {
            rl = ii % conf_.layers();
            in_size = (rl == 0) ? conf_.input_size() : conf_.hidden_size();
            m = conf_.hidden_size();
            if (conf_.desc()->alg_kind == alg_kind::rnn_lstm) m *= 4;
            weights_pack_[ii]
                    = cpu_blas::cblas_gemm_alloc<data_type>(CblasAMatrix,
                            m, conf_.batch(), (in_size + conf_.hidden_size() + 2));
        }
    }

    ~pgemm_rnn_bwd_t()
    {
        if (weights_pack_)
        {
            size_t total_layers = conf_.layers() * conf_.direction();
            for (size_t ii = 0; ii < total_layers; ++ii)
                cpu_blas::cblas_gemm_free<data_type>(weights_pack_[ii]);

            delete[] weights_pack_;
        }
        if (ts_) free(ts_);
    }

    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e)
    {
        switch (conf_.desc()->prop_kind) {
        case prop_kind::backward: execute_backward(); break;
        default: assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_backward();
    pd_t conf_;
    data_t *ts_;
    data_t **weights_pack_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
