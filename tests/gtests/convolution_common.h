#if 0
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
#endif

#define EXPAND_SIZES(mb, ng, ic, ih, iw, oc, oh, ow, kh, kw, ph, pw, sh, sw) \
    { mb, ng, ic, ih, iw, oc, oh, ow, kh, kw, ph, pw, sh, sw }
#define EXPAND_FORMATS(src, weights, bias, dst) \
    { memory::format::src, memory::format::weights, \
    memory::format::bias, memory::format::dst }

#define ENGINE engine::kind::cpu
#define ALGORITHM convolution_direct

#ifdef DIRECTION_FORWARD
#define FMT_WEIGHTS_BLOCKED OIhw8i8o
#define FMT_WEIGHTS_BLOCKED_G gOIhw8i8o
#define FMT_WEIGHTS_BLOCKED16 OIhw16i16o
#define FMT_WEIGHTS_BLOCKED16_G gOIhw16i16o
#define TEST_CASE_NAME_PREFIX Forward
#elif defined DIRECTION_BACKWARD_DATA
#define FMT_WEIGHTS_BLOCKED OIhw8o8i
#define FMT_WEIGHTS_BLOCKED_G gOIhw8o8i
#define FMT_WEIGHTS_BLOCKED16 OIhw16o16i
#define FMT_WEIGHTS_BLOCKED16_G gOIhw16o16i
#define TEST_CASE_NAME_PREFIX BackwardData
#elif defined DIRECTION_BACKWARD_WEIGHTS
#define FMT_WEIGHTS_BLOCKED OIhw8i8o
#define FMT_WEIGHTS_BLOCKED_G gOIhw8i8o
#define FMT_WEIGHTS_BLOCKED16 OIhw16i16o
#define FMT_WEIGHTS_BLOCKED16_G gOIhw16i16o
#define TEST_CASE_NAME_PREFIX BackwardWeights
#endif

#define FMT_BIAS x
#define FMT_NO_BIAS format_undef
#define FMT_DATA_BLOCKED nChw8c
#define FMT_DATA_BLOCKED16 nChw16c

#define PARAMS(src, weights, bias, dst, ...) \
    test_convolution_params_t { ENGINE, ALGORITHM, \
    EXPAND_FORMATS(src, weights, bias, dst), EXPAND_SIZES(__VA_ARGS__) }

#define CONCAT_WITH_UNDERSCORE_(a,b) a ## _ ## b
#define CONCAT_WITH_UNDERSCORE(a,b) CONCAT_WITH_UNDERSCORE_(a,b)

#define INST_TEST_CASE_(str, ...) INSTANTIATE_TEST_CASE_P( \
        str, convolution_test, ::testing::Values(__VA_ARGS__))
#define INST_TEST_CASE(str, ...) INST_TEST_CASE_( \
        CONCAT_WITH_UNDERSCORE(TEST_CASE_NAME_PREFIX, str), __VA_ARGS__)

#include "convolution_simple_small.h"
#include "convolution_alexnet.h"
#include "convolution_googlenet_v1.h"
#include "convolution_googlenet_v2.h"
#include "convolution_resnet.h"
#include "convolution_cifar10.h"
