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
#ifndef TEST_CONVOLUTION_FORWARD_COMMON_H
#define TEST_CONVOLUTION_FORWARD_COMMON_H

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"
#include <stdint.h>

namespace mkldnn {

template <typename data_t_src, typename data_t_wei,
          typename data_t_acc, typename data_t_dst>
void compute_ref_conv_fwd(const test_convolution_sizes_t &c,
        const memory::desc &src_d,
        const memory::desc &weights_d,
        const memory::desc &bias_d,
        const memory::desc &dst_d,
        const memory &src,
        const memory &weights,
        const memory &bias,
        const memory &dst)
{
    const bool w_bias = bias_d.data.format != memory::format::format_undef;
    data_t_src *src_data = (data_t_src *)src.get_data_handle();
    data_t_wei *weights_data = (data_t_wei *)weights.get_data_handle();

    data_t_dst *bias_data = w_bias ? (data_t_dst *)bias.get_data_handle() : nullptr;
    data_t_dst *dst_data = (data_t_dst *)dst.get_data_handle();

#pragma omp parallel for collapse(5) schedule(static)
    for (int n = 0; n < c.mb; n++) {
        for (int g = 0; g < c.ng; g++) {
            for (int oc = 0; oc < c.oc / c.ng; oc++) {
                for (int oh = 0; oh < c.oh; oh++) {
                    for (int ow = 0; ow < c.ow; ow++) {
                        data_t_acc a = (data_t_acc)(bias_data ?
                                bias_data[map_index(bias_d,
                                        g * c.oc / c.ng + oc)] :
                                0);
                        for (int ic = 0; ic < c.ic / c.ng; ic++) {
                            for (int kh = 0; kh < c.kh; kh++) {
                                for (int kw = 0; kw < c.kw; kw++) {
                                    int iw = ow * c.strw
                                          - c.padw + kw * (1 + c.dilw);
                                    int ih = oh * c.strh
                                          - c.padh + kh * (1 + c.dilh);
                                    if (iw < 0 || iw >= c.iw) continue;
                                    if (ih < 0 || ih >= c.ih) continue;
                                    int iidx = n * c.ic * c.ih * c.iw
                                            + g * c.ic / c.ng * c.ih * c.iw
                                            + ic * c.ih * c.iw + ih * c.iw + iw;
                                    int widx = g * c.oc / c.ng * c.ic
                                                    / c.ng * c.kh * c.kw
                                            + oc * c.ic / c.ng * c.kh * c.kw
                                            + ic * c.kh * c.kw + kh * c.kw + kw;
                                    a += (data_t_acc)(
                                               src_data[map_index(src_d, iidx)]
                                            *  weights_data[map_index(
                                                      weights_d, widx)]);
                                }
                            }
                        }
                        int oidx = n * c.oc * c.oh * c.ow
                                 + g * c.oc / c.ng * c.oh * c.ow
                                 + oc * c.oh * c.ow + oh * c.ow + ow;
                        dst_data[map_index(dst_d, oidx)] = (data_t_dst)a;
                    }
                }
            }
        }
    }
}

template <typename data_t_src, typename data_t_wei,
          typename data_t_acc, typename data_t_dst>
class convolution_forward_test
        : public ::testing::TestWithParam<test_convolution_params_t> {
protected:
    virtual void SetUp()
    {
        test_convolution_params_t p
            = ::testing::TestWithParam<test_convolution_params_t>::GetParam();
        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_EQ(p.aalgorithm, algorithm::convolution_direct);
        auto eng = engine(p.engine_kind, 0);

        memory::data_type data_type_src = data_traits<data_t_src>::data_type;
        memory::data_type data_type_dst = data_traits<data_t_dst>::data_type;
        memory::data_type data_type_wei = data_traits<data_t_wei>::data_type;

        test_convolution_sizes_t cd = p.sizes;

        auto aprop_kind = prop_kind::forward;
        bool with_bias = p.formats.bias_format != memory::format::format_undef;

        auto c_src_desc = create_md({ cd.mb, cd.ic, cd.ih, cd.iw },
            data_type_src, p.formats.src_format);
        auto c_weights_desc = cd.ng > 1 ?
                create_md({ cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw },
                        data_type_wei, p.formats.weights_format) :
                create_md({ cd.oc, cd.ic, cd.kh, cd.kw },
                        data_type_wei,p.formats.weights_format);
        auto c_dst_desc = create_md({ cd.mb, cd.oc, cd.oh, cd.ow },
                data_type_dst, p.formats.dst_format);
        auto c_bias_desc = with_bias ?
                create_md({ cd.oc }, data_type_dst, p.formats.bias_format) :
                create_md({}, data_type_dst, p.formats.bias_format);

        auto src_primitive_desc = memory::primitive_desc(c_src_desc, eng);
        auto weights_primitive_desc = memory::primitive_desc(
                c_weights_desc, eng);
        auto bias_primitive_desc = memory::primitive_desc(c_bias_desc, eng);
        auto dst_primitive_desc = memory::primitive_desc(c_dst_desc, eng);

        auto dst_size = dst_primitive_desc.get_size();

        // TODO: free
        auto ref_dst_data = new data_t_dst[dst_size];

        auto c_src = memory(src_primitive_desc);
        auto c_weights = memory(weights_primitive_desc);
        auto c_bias = memory(bias_primitive_desc);
        auto c_dst = memory(dst_primitive_desc);

        // Only true for dense format
        fill_data<data_t_src>(
                c_src.get_primitive_desc().get_size() / sizeof(data_t_src),
                (data_t_src *)c_src.get_data_handle());
        fill_data<data_t_wei>(
                c_weights.get_primitive_desc().get_size() / sizeof(data_t_wei),
                (data_t_wei *)c_weights.get_data_handle());
        if (with_bias) {
            fill_data<data_t_dst>(
                    c_bias.get_primitive_desc().get_size() / sizeof(data_t_dst),
                    (data_t_dst *)c_bias.get_data_handle());
        }

        std::vector<int> padR = { cd.padh, cd.padw };
        for (int i = 0; i < 2; ++i) {
            if ((cd.ih - ((cd.kh - 1) * (cd.dilh + 1) + 1) + cd.padh + padR[0])
                / cd.strh + 1 != cd.oh)
                ++padR[0];
            if ((cd.iw - ((cd.kw - 1) * (cd.dilw + 1) + 1) + cd.padw + padR[1])
                / cd.strw + 1 != cd.ow)
                ++padR[1];
        }

        auto conv_desc = with_bias ?
            convolution_forward::desc(aprop_kind, p.aalgorithm,
                c_src_desc, c_weights_desc, c_bias_desc, c_dst_desc,
                { cd.strh, cd.strw }, { cd.dilh, cd.dilw },
                { cd.padh, cd.padw }, padR, padding_kind::zero) :
            convolution_forward::desc(aprop_kind, p.aalgorithm,
                c_src_desc, c_weights_desc, c_dst_desc,
                { cd.strh, cd.strw }, { cd.dilh, cd.dilw },
                { cd.padh, cd.padw }, padR, padding_kind::zero);

        auto conv_primitive_desc = convolution_forward::primitive_desc(
                conv_desc, eng);

        auto conv = with_bias ?
            convolution_forward(conv_primitive_desc,
                    c_src, c_weights, c_bias, c_dst) :
            convolution_forward(conv_primitive_desc,
                    c_src, c_weights, c_dst);

        std::vector<primitive> pipeline;
        pipeline.push_back(conv);
        auto s = stream(stream::kind::lazy);
        s.submit(pipeline).wait();

        auto ref_memory = memory(memory::primitive_desc(c_dst_desc, eng),
                ref_dst_data);
        compute_ref_conv_fwd<data_t_src,data_t_wei,data_t_acc,data_t_dst>(
            cd, c_src_desc, c_weights_desc, c_bias_desc, c_dst_desc,
            c_src, c_weights, c_bias, ref_memory);
        compare_data<data_t_dst>(ref_memory, c_dst);
    }
};

}
#endif
