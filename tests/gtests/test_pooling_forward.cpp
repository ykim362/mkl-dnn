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

struct test_pool_desc_t {
    int mb, c;
    int ih, iw;
    int oh, ow;
    int kh, kw;
    int padt, padl;
    int strh, strw;
};

struct pool_test_params {
    prop_kind aprop_kind;
    const engine::kind engine_kind;
    algorithm aalgorithm;
    memory::format src_format;
    memory::format dst_format;
    test_pool_desc_t test_pd;
};

template <typename data_t>
void check_pool_fwd(const pool_test_params &p, const memory &src,
        const memory &dst, const memory &ws)
{
    data_t *src_data = (data_t *)src.get_data_handle();
    data_t *dst_data = (data_t *)dst.get_data_handle();

    auto ws_data = [=](size_t idx) -> int {
        auto w = (unsigned char *)ws.get_data_handle();
        if (w == nullptr) return -1;
        if (ws.get_primitive_desc().desc().data.data_type == mkldnn_u8)
            return (int)w[idx];
        else
            return ((int *)w)[idx];
    };

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();
    const memory::desc ws_d  = ws.get_primitive_desc().desc();

    auto pd = p.test_pd;

#pragma omp parallel for collapse(4) schedule(static)
    for (int n = 0; n < pd.mb; n++) {
        for (int c = 0; c < pd.c; c++) {
            for (int oh = 0; oh < pd.oh; oh++) {
                for (int ow = 0; ow < pd.ow; ow++) {
                    int oidx = n * pd.c * pd.oh * pd.ow + c * pd.oh * pd.ow
                            + oh * pd.ow + ow;
                    data_t out = dst_data[map_index(dst_d, oidx)];
                    int out_index = -1;
                    if(p.aalgorithm == pooling_max
                        && p.aprop_kind == prop_kind::forward_training) {
                        out_index = ws_data(map_index(ws_d, oidx));
                    }
                    data_t out_ref = data_t(0);
                    int out_ref_index = 0;
                    bool is_initialized = false;
                    int num_summands = 0;

                    for (int kh = 0; kh < pd.kh; ++kh) {
                        for (int kw = 0; kw < pd.kw; ++kw) {
                            const int ih = oh * pd.strh - pd.padt + kh;
                            const int iw = ow * pd.strw - pd.padl + kw;

                            if (ih < 0 || ih >= pd.ih) continue;
                            if (iw < 0 || iw >= pd.iw) continue;

                            int iidx = n * pd.c * pd.ih * pd.iw
                                    + c * pd.ih * pd.iw + ih * pd.iw + iw;

                            data_t d = src_data[map_index(src_d, iidx)];
                            if (p.aalgorithm == pooling_max) {
                                if (!is_initialized) {
                                    out_ref = d;
                                    out_ref_index = kh* pd.kh + kw;
                                    is_initialized = true;
                                } else {
                                    if (out_ref < d) {
                                        out_ref = d;
                                        out_ref_index = kh* pd.kh + kw;
                                    }
                                }
                            } else if (p.aalgorithm == pooling_avg_include_padding ||
                                       p.aalgorithm == pooling_avg_exclude_padding) {
                                out_ref += d;
                                num_summands++;
                            }
                        }
                    }

                    if (p.aalgorithm == pooling_avg_include_padding) {
                        num_summands = pd.kw * pd.kh;
                    }

                    if (p.aalgorithm == pooling_avg_include_padding ||
                        p.aalgorithm == pooling_avg_exclude_padding) {
                        out_ref /= num_summands;
                    }
                    EXPECT_NEAR(out, out_ref, 1e-6);
                    if(p.aalgorithm == pooling_max
                        && p.aprop_kind == forward_training) {
                        EXPECT_EQ(out_index, out_ref_index) << " n = " << n
                             << " c = " << c << " oh = " << oh << " ow = " << ow;
                    }
                }
            }
        }
    }
}

template <typename data_t>
class pooling_test : public ::testing::TestWithParam<pool_test_params> {
protected:
    virtual void SetUp()
    {
        pool_test_params p
                = ::testing::TestWithParam<pool_test_params>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_TRUE(p.aprop_kind == prop_kind::forward_training
                || p.aprop_kind == prop_kind::forward_scoring);
        auto eng = engine(p.engine_kind, 0);
        memory::data_type data_type = data_traits<data_t>::data_type;

        test_pool_desc_t pd = p.test_pd;

        auto p_src_desc
                = create_md({ pd.mb, pd.c, pd.ih, pd.iw }, data_type, p.src_format);
        auto p_dst_desc
                = create_md({ pd.mb, pd.c, pd.oh, pd.ow }, data_type, p.dst_format);

        std::vector<int> padR = { pd.padt, pd.padl };
        for (int i = 0; i < 2; ++i) {
        if ((pd.ih + pd.padt + padR[0] - pd.kh)/pd.strh + 1 < pd.oh) ++padR[0];
        if ((pd.iw + pd.padl + padR[1] - pd.kw)/pd.strw + 1 < pd.ow) ++padR[1];
        }

        auto pool_desc = pooling_forward::desc(p.aprop_kind, p.aalgorithm, p_src_desc,
                p_dst_desc, {pd.strh, pd.strw}, {pd.kh, pd.kw}, {pd.padt, pd.padl},
                padR, padding_kind::zero);

        auto pool_prim_desc = pooling_forward::primitive_desc(pool_desc, eng);

        bool with_workspace = p.aprop_kind == prop_kind::forward_training &&
                p.aalgorithm == pooling_max;
        auto p_workspace_desc = with_workspace ?
            pool_prim_desc.workspace_primitive_desc() :
            memory::primitive_desc( {{}, data_type, p.dst_format}, eng);
        auto p_src = memory({p_src_desc, eng});
        auto p_workspace = memory(p_workspace_desc);
        auto p_dst = memory({p_dst_desc, eng});

        fill_data<data_t>(p_src.get_primitive_desc().get_size()/ sizeof(data_t),
                (data_t *)p_src.get_data_handle());

        auto pool = with_workspace ?
                pooling_forward(pool_prim_desc, p_src, p_dst, p_workspace) :
                pooling_forward(pool_prim_desc, p_src, p_dst);

        std::vector<primitive> pipeline;
        pipeline.push_back(pool);

        stream(stream::kind::lazy).submit(pipeline).wait();

        check_pool_fwd<data_t>(p, p_src, p_dst, p_workspace);
    }
};

using pooling_test_float = pooling_test<float>;
using pooling_test_s8 = pooling_test<int8_t>;
using pooling_test_u8 = pooling_test<uint8_t>;
using pool_test_params_float = pool_test_params;

TEST_P(pooling_test_s8, TestsPooling)
{
}

INSTANTIATE_TEST_CASE_P(
        TestPoolingAlexnetForwardS8, pooling_test_s8, ::testing::Values(
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {1, 96, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {1, 256, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {1, 256, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardMaxS8, pooling_test_s8, ::testing::Values(
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
            algorithm::pooling_max, memory::format::nhwc, memory::format::nhwc,
            {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardAvgS8, pooling_test_s8, ::testing::Values(
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
            algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } },
            pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
            algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

TEST_P(pooling_test_u8, TestsPooling)
{
}

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardMaxU8, pooling_test_u8, ::testing::Values(
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
            algorithm::pooling_max, memory::format::nhwc, memory::format::nhwc,
            {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardAvgU8, pooling_test_u8, ::testing::Values(
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
            algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } },
            pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
            algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

TEST_P(pooling_test_float, TestsPooling)
{
}

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardMax, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 4, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 4, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
            ));


INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardMaxNHWC, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, { 2, 4, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardMaxBlocked, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 32, 13, 13, 12, 12, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 32, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 32, 3, 3, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 32, 3, 3, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 1, 1 } }

            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardMaxBlockedPerf, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardAvgBlockedPerf, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, { 1, 8, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1 } }
            , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardMaxBlocked16, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 32, 13, 13, 12, 12, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 32, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 32, 3, 3, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 32, 3, 3, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 1, 1 } }

            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardMaxBlocked16Perf, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardAvgBlocked16Perf, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_avg_include_padding, memory::format::nChw16c,
            memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_avg_exclude_padding, memory::format::nChw16c,
            memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_avg_include_padding, memory::format::nChw16c,
            memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_avg_exclude_padding, memory::format::nChw16c,
            memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));


INSTANTIATE_TEST_CASE_P(
        TestPoolingAlexnetForwardMaxNCHW, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAlexnetForwardMaxBlocked, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAlexnetForwardMaxBlocked16, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxBlockedStride1, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 55, 55, 53, 53, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 55, 55, 53, 53, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 27, 27, 25, 25, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 27, 27, 25, 25, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 13, 13, 11, 11, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 13, 13, 11, 11, 3, 3, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxCIFAR10NCHW, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAvgCIFAR10NCHW, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 32, 16, 15, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 32, 16, 15, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxCIFAR10Blocked, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAvgCIFAR10Blocked, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxCIFAR10Blocked16, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAvgCIFAR10Blocked16, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxGoogleNetV1NCHW, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxGoogleNetV1Blocked, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxGoogleNetV1Blocked16, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxResnet50NCHW, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxResnet50Blocked, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxResnet50Blocked16, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAvgGoogleNetV1NCHW, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAvgGoogleNetV1Blocked, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAvgGoogleNetV1Blocked16, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAvgResnet50NCHW, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAvgResnet50Blocked, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAvgResnet50Blocked16, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAsymmPadding, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, {1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1}}
            ,pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1}}
            ,pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1}}

            ,pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, {1, 8, 3, 14, 1, 8, 3, 3, 0, 1, 1, 2}}
            ,pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 8, 3, 14, 1, 8, 3, 3, 0, 1, 1, 2}}
            ,pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 8, 3, 14, 1, 8, 3, 3, 0, 1, 1, 2}}

            ,pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, {1, 96, 3, 100, 1, 51, 3, 3, 0, 1, 1, 2}}
            ,pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 96, 3, 100, 1, 51, 3, 3, 0, 1, 1, 2}}
            ,pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 96, 3, 100, 1, 51, 3, 3, 0, 1, 1, 2}}

            ,pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, {1, 96, 3, 102, 1, 52, 3, 3, 0, 1, 1, 2}}
            ,pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 96, 3, 102, 1, 52, 3, 3, 0, 1, 1, 2}}
            ,pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 96, 3, 102, 1, 52, 3, 3, 0, 1, 1, 2}}

            ,pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, {1, 96, 9, 103, 7, 52, 3, 3, 0, 1, 1, 2}}
            ,pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 96, 9, 103, 7, 52, 3, 3, 0, 1, 1, 2}}
            ,pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 96, 9, 103, 7, 52, 3, 3, 0, 1, 1, 2}}

            ,pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, {1, 96, 300, 500, 151, 251, 3, 3, 1, 1, 2, 2} }
            ,pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 96, 300, 500, 151, 251, 3, 3, 1, 1, 2, 2} }
            ,pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 96, 300, 500, 151, 251, 3, 3, 1, 1, 2, 2} }

            ));

}
