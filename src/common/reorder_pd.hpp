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

#ifndef REORDER_PD_HPP
#define REORDER_PD_HPP

#include <assert.h>
#include "mkldnn.h"

#include "c_types_map.hpp"
#include "memory_pd.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {

struct reorder_pd_t: public primitive_desc_t {
    reorder_pd_t(engine_t *engine, const float alpha, const float beta)
        : primitive_desc_t(engine, primitive_kind::reorder)
        , alpha_(alpha), beta_(beta)
    {  }
    virtual ~reorder_pd_t() {}

    virtual const op_desc_t *op_desc() const override { return nullptr; }

    float alpha() const { return alpha_; }
    float beta() const { return beta_; }

protected:
    float alpha_, beta_;
};

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
