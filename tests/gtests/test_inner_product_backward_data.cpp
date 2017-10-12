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

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"

namespace mkldnn {

struct test_inner_product_descr_t {
    int mb;
    int ic;
    int oc;
    int kh, kw;
};

template <typename data_t>
void compute_ref_inner_product_bwd_data(const test_inner_product_descr_t &ipd,
        const memory &diff_dst, const memory &weights, const memory &diff_src)
{
    data_t *diff_dst_data = (data_t *)diff_dst.get_data_handle();
    data_t *weights_data = (data_t *)weights.get_data_handle();
    data_t *diff_src_data = (data_t *)diff_src.get_data_handle();

    const memory::desc diff_dst_d = diff_dst.get_primitive_desc().desc();
    const memory::desc weights_d = weights.get_primitive_desc().desc();
    const memory::desc diff_src_d = diff_src.get_primitive_desc().desc();

    bool has_spatial = ipd.kh > 1 && ipd.kw > 1;

#pragma omp parallel for collapse(2) schedule(static)
    for (int n = 0; n < ipd.mb; n++) {
        for (int ic = 0; ic < ipd.ic; ic++) {
            if (has_spatial) {
                for (int kh = 0; kh < ipd.kh; ++kh) {
                    for (int kw = 0; kw < ipd.kw; ++kw) {
                        int dsidx = n * ipd.ic * ipd.kh * ipd.kw
                                + ic * ipd.kh * ipd.kw + kh * ipd.kw + kw;
                        data_t *ds = &diff_src_data[map_index(diff_src_d, dsidx)];
                        *ds = data_t(0);
                        for (int oc = 0; oc < ipd.oc; ++oc) {
                            int ddidx = n * ipd.oc + oc;
                            int widx = oc * ipd.ic * ipd.kh * ipd.kw
                                    + ic * ipd.kh * ipd.kw + kh * ipd.kw + kw;
                            *ds += diff_dst_data[map_index(diff_dst_d, ddidx)]
                                * weights_data[map_index(weights_d, widx)];
                        }
                    }
                }
            } else {
                int dsidx = n * ipd.ic + ic;
                data_t *ds = &diff_src_data[map_index(diff_src_d, dsidx)];
                *ds = data_t(0);
                for (int oc = 0; oc < ipd.oc; ++oc) {
                    int ddidx = n * ipd.oc + oc;
                    int widx = oc * ipd.ic + ic;
                    *ds += diff_dst_data[map_index(diff_dst_d, ddidx)]
                        * weights_data[map_index(weights_d, widx)];
                }
            }
        }
    }
}

struct inprod_test_params {
    const engine::kind engine_kind;
    memory::format diff_src_format;
    memory::format weights_format;
    memory::format diff_dst_format;
    test_inner_product_descr_t test_ipd;
};

template <typename data_t>
class inner_product_test_bwd_data : public ::testing::TestWithParam<inprod_test_params> {
protected:
    virtual void SetUp()
    {
        inprod_test_params p
                = ::testing::TestWithParam<inprod_test_params>::GetParam();
        test_inner_product_descr_t ipd = p.test_ipd;
        bool has_spatial = ipd.kh > 1 && ipd.kw > 1;

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        auto eng = engine(p.engine_kind, 0);
        memory::data_type data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

        auto ip_diff_src_desc = has_spatial ?
                create_md({ ipd.mb, ipd.ic, ipd.kh, ipd.kw }, data_type,
                        p.diff_src_format) :
                create_md({ ipd.mb, ipd.ic }, data_type, p.diff_src_format);
        auto ip_weights_desc = has_spatial ?
                create_md({ ipd.oc, ipd.ic, ipd.kh, ipd.kw }, data_type,
                        p.weights_format) :
                create_md({ ipd.oc, ipd.ic }, data_type, p.weights_format);
        auto ip_diff_dst_desc =
                create_md({ ipd.mb, ipd.oc }, data_type,p.diff_dst_format);

        // Create inner product forward (hint for backward)
        auto ip_fwd_desc = inner_product_forward::desc(prop_kind::forward,
            ip_diff_src_desc, ip_weights_desc, ip_diff_dst_desc);
        auto ip_fwd_pdesc = inner_product_forward::primitive_desc(ip_fwd_desc,
                eng);

        // Create inner product backward
        auto ip_desc = inner_product_backward_data::desc(ip_diff_src_desc,
                ip_weights_desc, ip_diff_dst_desc);

        auto ip_primitive_desc = inner_product_backward_data::primitive_desc(
                ip_desc, eng, ip_fwd_pdesc);

        auto ip_diff_src = memory(ip_primitive_desc.diff_src_primitive_desc());
        auto ip_weights = memory(ip_primitive_desc.weights_primitive_desc());
        auto ip_diff_dst = memory(ip_primitive_desc.diff_dst_primitive_desc());
        auto diff_src_ref = memory(ip_primitive_desc.diff_src_primitive_desc());

        fill_data<data_t>(
                ip_diff_dst.get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)ip_diff_dst.get_data_handle());
        fill_data<data_t>(
                ip_weights.get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)ip_weights.get_data_handle());

        auto ip = inner_product_backward_data(ip_primitive_desc,
                    ip_diff_dst, ip_weights, ip_diff_src);

        std::vector<primitive> pipeline;
        pipeline.push_back(ip);

        stream(stream::kind::lazy).submit(pipeline).wait();

        compute_ref_inner_product_bwd_data<data_t>(ipd, ip_diff_dst, ip_weights,
                diff_src_ref);
        compare_data<data_t>(diff_src_ref, ip_diff_src);
    }
};

using inner_product_test_float = inner_product_test_bwd_data<float>;
using inprod_test_params_float = inprod_test_params;

TEST_P(inner_product_test_float, TestsInnerProduct)
{
}
INSTANTIATE_TEST_CASE_P(
        TestInnerProductBackwardData, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any,
                        { 2, 32, 48, 6, 6 } },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any,
                        { 2, 1024, 48, 2, 2 } },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nchw, memory::format::oihw,
                        memory::format::nc,
                        { 2, 32, 48, 6, 6 } },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nChw8c, memory::format::oIhw8i,
                        memory::format::nc,
                        { 2, 32, 48, 6, 6 } },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nChw16c, memory::format::oIhw16i,
                        memory::format::nc,
                        { 2, 32, 48, 6, 6 } },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nc, memory::format::oi,
                        memory::format::nc,
                        { 2, 32, 1152, 1, 1 } },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nc, memory::format::oi,
                        memory::format::nc,
                        { 2, 2, 4, 1, 1 } }));
}
