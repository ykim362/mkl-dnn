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

#include <assert.h>
#include "mkldnn.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::alg_kind;
using namespace mkldnn::impl::types;

namespace {
status_t rnn_desc_init(rnn_desc_t *rnn_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc,
        const dims_t strides, const dims_t padding_l, const dims_t padding_r,
        padding_kind_t padding_kind) {
    bool args_ok = true
        && !any_null(conv_desc, src_desc, weights_desc, dst_desc, strides,
                padding_l)
        && one_of(alg_kind, convolution_direct)
        && one_of(padding_kind, padding_kind::padding_zero);
    if (!args_ok) return invalid_arguments;

    if (padding_r == nullptr) padding_r = padding_l;

    convolution_desc_t cd = {};
    cd.primitive_kind = primitive_kind::convolution;
    cd.prop_kind = prop_kind;
    cd.alg_kind = alg_kind;

    cd.diff_src_desc = cd.src_desc = zero_md();
    cd.diff_dst_desc = cd.dst_desc = zero_md();
    cd.diff_weights_desc = cd.weights_desc = zero_md();
    cd.diff_bias_desc = cd.bias_desc = zero_md();

    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);
    const bool with_bias = bias_desc && bias_desc->format != memory_format::undef;
    const bool with_groups = weights_desc->ndims == src_desc->ndims + 1;

    (prop_kind == backward_data ? cd.diff_src_desc : cd.src_desc) = *src_desc;
    (is_fwd ? cd.dst_desc : cd.diff_dst_desc)  = *dst_desc;
    (prop_kind == backward_weights ? cd.diff_weights_desc : cd.weights_desc) =
        *weights_desc;
    if (with_bias)
        (prop_kind == backward_weights ? cd.diff_bias_desc : cd.bias_desc) =
            *bias_desc;

    int sp_dims = src_desc->ndims - 2;
    utils::array_copy(cd.strides, strides, sp_dims);
    utils::array_copy(cd.padding[0], padding_l, sp_dims);
    utils::array_copy(cd.padding[1], padding_r, sp_dims);

    cd.padding_kind = padding_kind;

    const int g = with_groups ? weights_desc->dims[0] : 1;

    bool consistency = true
        && src_desc->ndims == 4
        && dst_desc->ndims == 4
        && utils::one_of(weights_desc->ndims, 4, 5)
        && (with_bias ? bias_desc->ndims == 1 : true)
        && (with_bias ? bias_desc->dims[0] == dst_desc->dims[1] : true)
        && src_desc->dims[0] == dst_desc->dims[0]
        && src_desc->dims[1] == g * weights_desc->dims[with_groups + 1]
        && dst_desc->dims[1] == g * weights_desc->dims[with_groups + 0];
    for (int i = 2; i <= 3; ++i)
        consistency = consistency && (
                (src_desc->dims[i] - weights_desc->dims[with_groups + i]
                 + padding_l[i - 2] + padding_r[i - 2]) / strides[i - 2] + 1
                == dst_desc->dims[i]);
    if (!consistency) return invalid_arguments;

    *conv_desc = cd;
    return success;
}
}

status_t mkldnn_rnn_forward_desc_init(mkldnn_rnn_desc_t *rnn_desc,
        mkldnn_prop_kind_t prop_kind,
        mkldnn_alg_kind_t alg_kind, mkldnn_rnn_direction_t direction,
        mkldnn_rnn_input_mode_t input_mode, size_t num_states, 
        size_t num_layers, size_t num_seqs, double dropout_prob,
        const mkldnn_memory_desc_t *x_desc,
        const mkldnn_memory_desc_t *hx_desc,
        const mkldnn_memory_desc_t *cx_desc,
        const mkldnn_memory_desc_t *y_desc,
        const mkldnn_memory_desc_t *hy_desc,
        const mkldnn_memory_desc_t *cy_desc,
        const mkldnn_memory_desc_t *weights_desc) {
        // prop_kind_t prop_kind, alg_kind_t alg_kind,
        // const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        // const memory_desc_t *bias_desc, const memory_desc_t *dst_desc,
        // const dims_t strides, const dims_t padding_l, const dims_t padding_r,
        // padding_kind_t padding_kind) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;
    return rnn_desc_init(rnn_desc, prop_kind, alg_kind, direction,
            input_mode, num_state, num_layers, num_seqs, dropout_prob,
            src_desc,
            weights_desc, bias_desc, dst_desc, strides, padding_l, padding_r,
            padding_kind);
}

status_t mkldnn_rnn_backward_data_desc_init(
        convolution_desc_t *conv_desc, alg_kind_t alg_kind,
        const memory_desc_t *diff_src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *diff_dst_desc, const dims_t strides,
        const dims_t padding_l, const dims_t padding_r,
        padding_kind_t padding_kind) {
    return rnn_desc_init(conv_desc, backward_data, alg_kind, diff_src_desc,
            weights_desc, nullptr, diff_dst_desc, strides, padding_l,
            padding_r, padding_kind);
}

status_t mkldnn_rnn_backward_weights_desc_init(
        convolution_desc_t *conv_desc, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *diff_weights_desc,
        const memory_desc_t *diff_bias_desc,
        const memory_desc_t *diff_dst_desc, const dims_t strides,
        const dims_t padding_l, const dims_t padding_r,
        padding_kind_t padding_kind) {
    return rnn_desc_init(conv_desc, backward_weights, alg_kind, src_desc,
            diff_weights_desc, diff_bias_desc, diff_dst_desc, strides,
            padding_l, padding_r, padding_kind);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
