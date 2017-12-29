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

#ifndef RNN_PD_HPP
#define RNN_PD_HPP

#include "c_types_map.hpp"
#include "memory_pd.hpp"
#include "mkldnn.h"
#include "primitive_desc.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {

struct rnn_fwd_pd_t : public primitive_desc_t {
    typedef rnn_fwd_pd_t base_class;
    typedef rnn_fwd_pd_t hint_class;
    static constexpr auto base_pkind = primitive_kind::rnn;

    rnn_fwd_pd_t(engine_t *engine, const rnn_desc_t *adesc,
            const rnn_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, primitive_kind::rnn)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd) {}
    virtual ~rnn_fwd_pd_t() {}

    const rnn_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    virtual const memory_pd_t *input_pd(int index = 0) const override {
        if (desc_.alg_kind == alg_kind::rnn_lstm) {
            switch (index) {
            case 0: case 1: case 2: return src_pd(index);
            case 3: return weights_pd();
            default: return nullptr;
            }
        } else {
            switch (index) {
            case 0: case 1: return src_pd(index);
            case 2: return weights_pd();
            default: return nullptr;
            }
        }
    }
    virtual const memory_pd_t *output_pd(int index = 0) const override {
        if (desc_.state_outputs) {
            if (desc_.alg_kind == alg_kind::rnn_lstm) {
                switch (index) {
                case 0: case 1: case 2: return dst_pd(index);
                case 3: return workspace_pd();
                default: return nullptr;
                }
            } else {
                switch (index) {
                case 0: case 1: return dst_pd(index);
                case 2: return workspace_pd();
                default: return nullptr;
                }
            }
        } else {
            switch (index) {
            case 0: return dst_pd(index);
            case 1: return workspace_pd();
            default: return nullptr;
            }
        }
    }

    virtual int n_inputs() const override
    { return desc_.alg_kind == alg_kind::rnn_lstm ? 4 : 3; }
    virtual int n_outputs() const override {
        if (desc_.state_outputs) {
            if (desc_.alg_kind == alg_kind::rnn_lstm) {
                return 3 + (workspace_pd() != nullptr);
            } else {
                return 2 + (workspace_pd() != nullptr);
            }
        } else {
            return 1 + (workspace_pd() != nullptr);
        }
    }

    virtual status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
        case query::rnn_d: *(const rnn_desc_t **)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common rnn aux functions */

    int tau() const { return desc_.num_seqs; }
    int layers() const { return desc_.num_layers; }
    int hidden_size() const { return desc_.num_states; }
    int direction() const {
        return (desc_.direction == rnn_direction::unidirectional) ? 1 : 2;
    }
    int batch() const {
        return desc_.x_desc.dims[1];
    }
    int input_size() const {
        return desc_.x_desc.dims[2];
    }
    int state_outputs() const { return desc_.state_outputs; }
    int gates_num() const {
        switch (desc_.alg_kind) {
        case alg_kind::rnn_relu: return 1;
        case alg_kind::rnn_tanh: return 1;
        case alg_kind::rnn_lstm: return 4;
        case alg_kind::rnn_gru: return 3;
        default: return 0;
        }
    }
    int w1_size() const {
        int size = hidden_size() * (hidden_size() + input_size() + 2);
        switch (desc_.alg_kind) {
        case alg_kind::rnn_relu: size *= 1; break;
        case alg_kind::rnn_tanh: size *= 1; break;
        case alg_kind::rnn_lstm: size *= 4; break;
        case alg_kind::rnn_gru: size *= 3; break;
        default: return 0;
        }
        return size;
    }
    int wx_size() const {
        if (layers() > 1) {
            int size = hidden_size() * (hidden_size() + hidden_size() + 2);
            switch (desc_.alg_kind) {
            case alg_kind::rnn_relu: size *= 1; break;
            case alg_kind::rnn_tanh: size *= 1; break;
            case alg_kind::rnn_lstm: size *= 4; break;
            case alg_kind::rnn_gru: size *= 3; break;
            default: return 0;
            }
            return size;
        } else
            return 0;
    }
    int total_param_size() const {
        if (layers() > 1) {
            return direction() * (w1_size() + (layers() - 1) * wx_size());
        } else
            return direction() * w1_size();
    }
    int h_size() const { return batch() * hidden_size(); }
    int x_size() const { return batch() * input_size(); }
    int h_nlayer_size() const { return h_size() * layers(); }
    int gates_size() const { return h_size() * gates_num(); }
    int gates_nlayer_size() const { return gates_size() * layers(); }
    int gates_space_size() const
    { return gates_nlayer_size() * tau() * direction(); }
    int h_space_size() const
    { return h_nlayer_size() * tau() * direction(); }
    int workspace_size() const {
        return gates_space_size()
            + (desc_.alg_kind == alg_kind::rnn_lstm ? 2 : 1) * h_space_size();
    }

protected:
    rnn_desc_t desc_;
    const rnn_fwd_pd_t *hint_fwd_pd_;
};

struct rnn_bwd_pd_t : public primitive_desc_t {
    typedef rnn_bwd_pd_t base_class;
    typedef rnn_fwd_pd_t hint_class;
    static constexpr auto base_pkind = primitive_kind::rnn;

    rnn_bwd_pd_t(engine_t *engine, const rnn_desc_t *adesc,
            const rnn_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, primitive_kind::rnn)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd) {}
    virtual ~rnn_bwd_pd_t() {}

    const rnn_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    virtual const memory_pd_t *input_pd(int index = 0) const override {
        if (desc_.alg_kind == alg_kind::rnn_lstm) {
            if (desc_.state_outputs) {
                switch (index) {
                case 0: case 1: case 2: return src_pd(index);
                case 3: case 4: case 5: return diff_dst_pd(index - 3);
                case 6: return weights_pd();
                case 7: return workspace_pd();
                default: return nullptr;
                }
            } else {
                switch (index) {
                case 0: case 1: case 2: return src_pd(index);
                case 3: return diff_dst_pd(index - 3);
                case 4: return weights_pd();
                case 5: return workspace_pd();
                default: return nullptr;
                }
            }
        } else {
            if (desc_.state_outputs) {
                switch (index) {
                case 0: case 1: return src_pd(index);
                case 2: case 3: return diff_dst_pd(index - 2);
                case 4: return weights_pd();
                case 5: return workspace_pd();
                default: return nullptr;
                }
            } else {
                switch (index) {
                case 0: case 1: return src_pd(index);
                case 2: return diff_dst_pd(index - 2);
                case 3: return weights_pd();
                case 4: return workspace_pd();
                default: return nullptr;
                }
            }
        }
    }
    virtual const memory_pd_t *output_pd(int index = 0) const override {
        if (desc_.alg_kind == alg_kind::rnn_lstm) {
            switch (index) {
            case 0: case 1: case 2: return diff_src_pd(index);
            case 3: return diff_weights_pd();
            default: return nullptr;
            }
        } else {
            switch (index) {
            case 0: case 1: return diff_src_pd(index);
            case 2: return diff_weights_pd();
            default: return nullptr;
            }
        }
    }

    virtual int n_inputs() const override {
        if (desc_.state_outputs)
            if (desc_.alg_kind == alg_kind::rnn_lstm) {
                return 7 + (workspace_pd() != nullptr);
            } else {
                return 5 + (workspace_pd() != nullptr);
            }
        else {
            if (desc_.alg_kind == alg_kind::rnn_lstm) {
                return 5 + (workspace_pd() != nullptr);
            } else {
                return 4 + (workspace_pd() != nullptr);
            }
        }
    }
    virtual int n_outputs() const override
    { return desc_.alg_kind == alg_kind::rnn_lstm ? 4 : 3; }

    virtual status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
        case query::rnn_d: *(const rnn_desc_t **)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common rnn aux functions */

    int tau() const { return desc_.num_seqs; }
    int layers() const { return desc_.num_layers; }
    int hidden_size() const { return desc_.num_states; }
    int direction() const {
        return (desc_.direction == rnn_direction::unidirectional) ? 1 : 2;
    }
    int batch() const
    { return desc_.x_desc.dims[1]; }
    int input_size() const
    { return desc_.x_desc.dims[2]; }
    int state_outputs() const { return desc_.state_outputs; }
    int gates_num() const {
        switch (desc_.alg_kind) {
        case alg_kind::rnn_relu: return 1;
        case alg_kind::rnn_tanh: return 1;
        case alg_kind::rnn_lstm: return 4;
        case alg_kind::rnn_gru: return 3;
        default: return 0;
        }
    }
    int w1_size() const {
        int size = hidden_size() * (hidden_size() + input_size() + 2);
        switch (desc_.alg_kind) {
        case alg_kind::rnn_relu: size *= 1; break;
        case alg_kind::rnn_tanh: size *= 1; break;
        case alg_kind::rnn_lstm: size *= 4; break;
        case alg_kind::rnn_gru: size *= 3; break;
        default: return 0;
        }
        return size;
    }
    int wx_size() const {
        if (layers() > 1) {
            int size = hidden_size() * (hidden_size() + hidden_size() + 2);
            switch (desc_.alg_kind) {
            case alg_kind::rnn_relu: size *= 1; break;
            case alg_kind::rnn_tanh: size *= 1; break;
            case alg_kind::rnn_lstm: size *= 4; break;
            case alg_kind::rnn_gru: size *= 3; break;
            default: return 0;
            }
            return size;
        }
        return 0;
    }
    int total_param_size() const {
        if (layers() > 1) {
            return direction() * (w1_size() + (layers() - 1) * wx_size());
        } else
            return direction() * w1_size();
    }
    int h_size() const { return batch() * hidden_size(); }
    int x_size() const { return batch() * input_size(); }
    int h_nlayer_size() const { return h_size() * layers(); }
    int gates_size() const { return h_size() * gates_num(); }
    int gates_nlayer_size() const { return gates_size() * layers(); }
    int gates_space_size() const
    { return gates_nlayer_size() * tau() * direction(); }
    int h_space_size() const
    { return h_nlayer_size() * tau() * direction(); }
    int workspace_size() const {
        return gates_space_size()
            + (desc_.alg_kind == alg_kind::rnn_lstm ? 2 : 1) * h_space_size();
    }

protected:
    rnn_desc_t desc_;
    const rnn_fwd_pd_t *hint_fwd_pd_;
};
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s