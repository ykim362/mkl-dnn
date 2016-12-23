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

#ifndef RNN_PD_HPP
#define RNN_PD_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "memory_pd.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {

using namespace mkldnn::impl::alg_kind;

struct rnn_fwd_pd_t: public primitive_desc_t {
    typedef rnn_fwd_pd_t base_class;
    typedef rnn_fwd_pd_t hint_class;
    static constexpr auto base_pkind = primitive_kind::rnn;

    rnn_fwd_pd_t(mkldnn::impl::engine_t *engine, const rnn_desc_t *adesc,
            const rnn_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, primitive_kind::rnn)
        , desc_(*adesc), hint_fwd_pd_(hint_fwd_pd) {}
    virtual ~rnn_fwd_pd_t() {}

    const rnn_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }

    virtual const memory_pd_t *input_pd(int index = 0) const override {
        switch(index) {
        case 0: return x_pd();
        case 1: return hx_pd();
        case 2: return cx_pd();
        case 3: return weights_pd();
        default: return nullptr;
        }
     }
    virtual const memory_pd_t *output_pd(int index = 0) const override {
        switch(index) {
        case 0: return y_pd();
        case 1: return hy_pd();
        case 2: return cy_pd();
        case 3: return workspace_pd();
        default: return nullptr;
        }
    }

    virtual int n_inputs() const override { return 4; }
    virtual int n_outputs() const override
    { return 3 + (workspace_pd() != nullptr); }

    virtual status_t query(query_t what, int idx, void *result) const override
    {
        switch (what) {
        case query::rnn_d:
            *(const rnn_desc_t**)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common rnn aux functions */

    inline size_t Tau() const { return desc_.num_seqs; }
    inline size_t Layers() const { return desc_.num_layers; }
    inline size_t Hidden_size() const { return desc_.num_states; }
    inline int Direction() const { return desc_.direction; }
    inline size_t Batch() const { return static_cast<size_t>(desc_.x_desc.dims[1]); }
    inline size_t Input_size() const { return static_cast<size_t>(desc_.x_desc.dims[2]); }
    inline int Gates_num() const {
        switch (desc_.alg_kind) {
            case rnn_relu: return 1;
            case rnn_tanh: return 1;
            case rnn_lstm: return 4;
            case rnn_gru: return 3;
            default: return 0;
        }
    }
    inline size_t W1_size() const { 
        size_t size = Hidden_size() * (Hidden_size() + Input_size() + 2);
        switch (desc_.alg_kind) {
        case rnn_relu: size *= 1;
          break;
        case rnn_tanh: size *= 1;
          break;
        case rnn_lstm:size *= 4;
          break;
        case rnn_gru: size *= 3;
          break;
        default: return 0;
        }
        return size;
    }
    inline size_t Wx_size() const {
        if (Layers() > 1) {
            size_t size = Hidden_size() * (Hidden_size() + Hidden_size() + 2);
            switch (desc_.alg_kind) {
            case rnn_relu: size *= 1;
              break;
            case rnn_tanh: size *= 1;
              break;
            case rnn_lstm:size *= 4;
              break;
            case rnn_gru: size *= 3;
              break;
            default: return 0;
            }
            return size;
        }
        else 
            return 0;
    }
    inline size_t Total_param_size() const {
        if (Layers() > 1) {
            return W1_size() + (Layers() - 1) * Wx_size();
        } else
            return W1_size();
    }
    inline size_t H_size() const { return Batch() * Hidden_size(); }
    inline size_t X_size() const { return Batch() * Input_size(); }
    inline size_t H_nlayer_size() const { return H_size() * Layers(); }
    inline size_t Gates_size() const { return H_size() * Gates_num(); }
    inline size_t Gates_nlayer_size() const { return Gates_size() * Layers(); }
    inline size_t Gates_space_size() const { return Gates_nlayer_size() * Tau(); }
    inline size_t Hout_space_size() const { return H_nlayer_size() * Tau(); }
    inline size_t C_space_size() const { return Hout_space_size(); }
    inline size_t Workspace_size() const {
        return Gates_space_size()
            + Hout_space_size()
            + C_space_size();
    }

protected:
    rnn_desc_t desc_;
    const rnn_fwd_pd_t *hint_fwd_pd_;
};

struct rnn_bwd_pd_t: public primitive_desc_t {
    typedef rnn_bwd_pd_t base_class;
    typedef rnn_fwd_pd_t hint_class;
    static constexpr auto base_pkind = primitive_kind::rnn;

    rnn_bwd_pd_t(mkldnn::impl::engine_t *engine, const rnn_desc_t *adesc,
            const rnn_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, primitive_kind::rnn)
        , desc_(*adesc), hint_fwd_pd_(hint_fwd_pd) {}
    virtual ~rnn_bwd_pd_t() {}

    const rnn_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }

    virtual const memory_pd_t *input_pd(int index = 0) const override {
        switch(index) {
        case 0: return x_pd();
        case 1: return hx_pd();
        case 2: return cx_pd();
        case 3: return dy_pd();
        case 4: return dhy_pd();
        case 5: return dcy_pd();
        case 6: return weights_pd();
        case 7: return workspace_pd();
        default: return nullptr;
        }
    }
    virtual const memory_pd_t *output_pd(int index = 0) const override
    {
        switch(index) {
        case 0: return dx_pd();
        case 1: return dhx_pd();
        case 2: return dcx_pd();
        case 3: return diff_weights_pd();
        default: return nullptr;
        }
    }

    virtual int n_inputs() const override
    { return 7 + (workspace_pd() != nullptr); }
    virtual int n_outputs() const override
    { return 4; }

    virtual status_t query(query_t what, int idx, void *result) const override
    {
        switch (what) {
        case query::rnn_d:
            *(const rnn_desc_t**)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common rnn aux functions */

    inline size_t Tau() const { return desc_.num_seqs; }
    inline size_t Layers() const { return desc_.num_layers; }
    inline size_t Hidden_size() const { return desc_.num_states; }
    inline int Direction() const { return desc_.direction; }
    inline size_t Batch() const { return static_cast<size_t>(desc_.x_desc.dims[1]); }
    inline size_t Input_size() const { return static_cast<size_t>(desc_.x_desc.dims[2]); }
    inline int Gates_num() const {
        switch (desc_.alg_kind) {
            case rnn_relu: return 1;
            case rnn_tanh: return 1;
            case rnn_lstm: return 4;
            case rnn_gru: return 3;
            default: return 0;
        }
    }
    inline size_t W1_size() const { 
        size_t size = Hidden_size() * (Hidden_size() + Input_size() + 2);
        switch (desc_.alg_kind) {
        case rnn_relu: size *= 1;
          break;
        case rnn_tanh: size *= 1;
          break;
        case rnn_lstm:size *= 4;
          break;
        case rnn_gru: size *= 3;
          break;
        default: return 0;
        }
        return size;
    }
    inline size_t Wx_size() const {
        if (Layers() > 1) {
            size_t size = Hidden_size() * (Hidden_size() + Hidden_size() + 2);
            switch (desc_.alg_kind) {
            case rnn_relu: size *= 1;
              break;
            case rnn_tanh: size *= 1;
              break;
            case rnn_lstm:size *= 4;
              break;
            case rnn_gru: size *= 3;
              break;
            default: return 0;
            }
            return size;
        }
        else 
            return 0;
    }
    inline size_t Total_param_size() const {
        if (Layers() > 1) {
            return W1_size() + (Layers() - 1) * Wx_size();
        } else
            return W1_size();
    }
    inline size_t H_size() const { return Batch() * Hidden_size(); }
    inline size_t X_size() const { return Batch() * Input_size(); }
    inline size_t H_nlayer_size() const { return H_size() * Layers(); }
    inline size_t Gates_size() const { return H_size() * Gates_num(); }
    inline size_t Gates_nlayer_size() const { return Gates_size() * Layers(); }
    inline size_t Gates_space_size() const { return Gates_nlayer_size() * Tau(); }
    inline size_t Hout_space_size() const { return H_nlayer_size() * Tau(); }
    inline size_t C_space_size() const { return Hout_space_size(); }
    inline size_t Workspace_size() const { 
        return Gates_space_size()
            + Hout_space_size()
            + C_space_size();
    }
protected:
    rnn_desc_t desc_;
    const rnn_fwd_pd_t *hint_fwd_pd_;
};

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
