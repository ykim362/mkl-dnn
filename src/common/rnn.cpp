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
using namespace mkldnn::impl::rnn_direction;
using namespace mkldnn::impl::rnn_input_mode;
using namespace mkldnn::impl::types;

namespace {
status_t
rnn_desc_init(rnn_desc_t *rnn_desc, prop_kind_t prop_kind, alg_kind_t alg_kind,
              rnn_direction_t direction, rnn_input_mode_t input_mode,
              size_t num_states, size_t num_layers, size_t num_seqs,
              const memory_desc_t *x_desc, const memory_desc_t *hx_desc,
              const memory_desc_t *y_desc, const memory_desc_t *weights_desc) {
  bool args_ok = true && one_of(prop_kind, forward_training, forward_inference,
                                backward) &&
                 one_of(alg_kind, rnn_lstm) &&
                 one_of(direction, rnn_unidirectional, rnn_bidirectional) &&
                 one_of(input_mode, rnn_linear_input) &&
                 !any_null(x_desc, hx_desc, y_desc, weights_desc) &&
                 num_states != 0 && num_layers != 0 && num_seqs != 0;
  if (!args_ok)
    return invalid_arguments;
  int dir = (direction == rnn_unidirectional) ? 1 : 2;
  bool consistency = true && x_desc->ndims == 3 && hx_desc->ndims == 3 &&
                     y_desc->ndims == 3 && x_desc->dims[0] == y_desc->dims[0] &&
                     x_desc->dims[1] == y_desc->dims[1] &&
                     hx_desc->dims[1] == y_desc->dims[1] &&
                     y_desc->dims[2] == dir * static_cast<int>(num_states) &&
                     x_desc->dims[0] == static_cast<int>(num_seqs) &&
                     hx_desc->dims[0] == static_cast<int>(num_layers) &&
                     hx_desc->dims[2] == static_cast<int>(num_states);
  if (!consistency)
    return invalid_arguments;

  rnn_desc_t rd = {};
  rd.primitive_kind = primitive_kind::rnn;
  rd.prop_kind = prop_kind;
  rd.alg_kind = alg_kind;
  rd.direction = direction;
  rd.input_mode = input_mode;

  rd.num_states = num_states;
  rd.num_layers = num_layers;
  rd.num_seqs = num_seqs;

  rd.x_desc = *x_desc;
  rd.hx_desc = *hx_desc;
  rd.y_desc = *y_desc;
  rd.weights_desc = *weights_desc;

  *rnn_desc = rd;
  return success;
}
}

status_t mkldnn_rnn_forward_desc_init(
    rnn_desc_t *rnn_desc, prop_kind_t prop_kind, alg_kind_t alg_kind,
    rnn_direction_t direction, rnn_input_mode_t input_mode, size_t num_states,
    size_t num_layers, size_t num_seqs, const memory_desc_t *x_desc,
    const memory_desc_t *hx_desc, const memory_desc_t *y_desc,
    const memory_desc_t *weights_desc) {
  return rnn_desc_init(rnn_desc, prop_kind, alg_kind, direction, input_mode,
                       num_states, num_layers, num_seqs, x_desc, hx_desc,
                       y_desc, weights_desc);
}

status_t mkldnn_rnn_backward_desc_init(
    rnn_desc_t *rnn_desc, prop_kind_t prop_kind, alg_kind_t alg_kind,
    rnn_direction_t direction, rnn_input_mode_t input_mode, size_t num_states,
    size_t num_layers, size_t num_seqs, const memory_desc_t *x_desc,
    const memory_desc_t *hx_desc, const memory_desc_t *y_desc,
    const memory_desc_t *weights_desc) {
  return rnn_desc_init(rnn_desc, prop_kind, alg_kind, direction, input_mode,
                       num_states, num_layers, num_seqs, x_desc, hx_desc,
                       y_desc, weights_desc);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
