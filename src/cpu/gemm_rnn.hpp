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

#ifndef CPU_GEMM_RNN_FWD_HPP
#define CPU_GEMM_RNN_FWD_HPP

#include <assert.h>
#include "c_types_map.hpp"
#include "cpu_rnn_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct gemm_rnn_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_rnn_fwd_pd_t {
        pd_t(engine_t *engine, const rnn_desc_t *adesc,
                const rnn_fwd_pd_t *hint_fwd_pd)
            : cpu_rnn_fwd_pd_t(engine, adesc, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(gemm_rnn_fwd_t);

        virtual status_t init() override {
#if defined(USE_CBLAS) && defined(USE_TRANS) && defined(USE_VML)
            using namespace prop_kind;
            using namespace alg_kind;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(desc()->prop_kind, forward_training,
                        forward_inference)
                && utils::one_of(desc()->alg_kind, rnn_relu,
                        rnn_tanh, rnn_lstm, rnn_gru)
                && utils::everyone_is(data_type, desc()->x_desc.data_type,
                        desc()->hx_desc.data_type,
                        desc()->y_desc.data_type,
                        desc()->weights_desc.data_type);
            if (!ok) return status::unimplemented;
            auto ws_size = static_cast<int>(Workspace_size());
            memory_desc_t ws_d;
            mkldnn_memory_desc_init(&ws_d, 1, { &ws_size }, data_type, memory_format::x);
            ws_pd_ = cpu_memory_pd_t(this->engine(), &ws_d);
            return status::success;
#else
        return status::unimplemented;
#endif // defined(USE_CBLAS) && defined(USE_TRANS) && defined(USE_VML)
        }
    };

    gemm_rnn_fwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd), ts_(nullptr) {
            using namespace mkldnn::impl::utils;
            auto bsize = (conf_.Input_size() > conf_.Hidden_size()) ? conf_.Input_size() : conf_.Hidden_size();
            auto tmp1 = bsize + conf_.Hidden_size() + 2;
            auto tmp2 = conf_.Hidden_size() * 4;
            auto btmp = (tmp1 > tmp2) ? tmp1 : tmp2;
            auto ts_size_ = btmp * conf_.Batch();
            ts_ = new data_t[ts_size_];
            array_set(ts_, 0.0, ts_size_);
        }
    ~gemm_rnn_fwd_t() { if (ts_) delete [] ts_; }

    typedef typename prec_trait<data_type>::type data_t;

    virtual void execute(event_t *e) {
        switch (conf_.desc()->prop_kind) {
        case prop_kind::forward_training:
        case prop_kind::forward_inference:
            execute_forward();
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;
    data_t *ts_;
};

template <impl::data_type_t data_type>
struct gemm_rnn_bwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_rnn_bwd_pd_t {
        pd_t(engine_t *engine, const rnn_desc_t *adesc,
                const rnn_fwd_pd_t *hint_fwd_pd)
            : cpu_rnn_bwd_pd_t(engine, adesc, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(gemm_rnn_bwd_t);

        virtual status_t init() override {
#if defined(USE_CBLAS) && defined(USE_TRANS) && defined(USE_VML)
            using namespace prop_kind;
            using namespace alg_kind;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(desc()->prop_kind, backward)
                && utils::one_of(desc()->alg_kind, rnn_relu,
                        rnn_tanh, rnn_lstm, rnn_gru)
                && utils::everyone_is(data_type, desc()->x_desc.data_type,
                        desc()->hx_desc.data_type,
                        desc()->y_desc.data_type,
                        desc()->weights_desc.data_type);
            if (!ok) return status::unimplemented;
            auto ws_size = static_cast<int>(Workspace_size());
            memory_desc_t ws_d;
            mkldnn_memory_desc_init(&ws_d, 1, { &ws_size }, data_type, memory_format::x);
            ws_pd_ = cpu_memory_pd_t(this->engine(), &ws_d);
            return status::success;
#else
        return status::unimplemented;
#endif // USE_CBLAS
        }
    };

    gemm_rnn_bwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd), ts_(nullptr) {
            auto bsize = (conf_.Input_size() > conf_.Hidden_size()) ? conf_.Input_size() : conf_.Hidden_size();
            auto tmp1 = bsize + conf_.Hidden_size() + 2;
            auto tmp2 = conf_.Hidden_size() * 4;
            auto btmp = (tmp1 > tmp2) ? tmp1 : tmp2;
             auto ts_size_ = btmp * conf_.Batch()
                + conf_.Gates_space_size()
                + conf_.Hout_space_size()
                +conf_.C_space_size();
            ts_ = new data_t[ts_size_];
        }
    ~gemm_rnn_bwd_t() { if (ts_) delete [] ts_; }

    typedef typename prec_trait<data_type>::type data_t;

    virtual void execute(event_t *e) {
        switch (conf_.desc()->prop_kind) {
        case prop_kind::backward:
        case prop_kind::backward_data:
        case prop_kind::backward_weights:
            execute_backward();
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_backward();
    pd_t conf_;
    data_t *ts_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
