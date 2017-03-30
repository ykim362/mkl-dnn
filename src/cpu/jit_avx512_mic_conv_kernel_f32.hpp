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

#ifndef JIT_AVX512_MIC_CONV_KERNEL_F32_HPP
#define JIT_AVX512_MIC_CONV_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx512_mic_conv_fwd_kernel_f32 : public jit_generator {

    jit_avx512_mic_conv_fwd_kernel_f32(jit_conv_conf_t ajcp) : jcp(ajcp)
    {
        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *)) this->getCode();
    }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, bool with_relu = false,
            double relu_negative_slope = 0.);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    enum { typesize = sizeof(float) };

    reg64_t param = abi_param1;
    reg64_t reg_inp = r8;
    reg64_t reg_ker = r9;
    reg64_t reg_out = r10;

    reg64_t reg_inp_prf = r11;
    reg64_t reg_ker_prf = r12;
    reg64_t reg_out_prf = r13;

    reg64_t aux_reg_inp = r14;
    reg64_t aux_reg_ker = r15;

    reg64_t aux_reg_inp_prf = rsi;
    reg64_t aux_reg_ker_prf = rdx;

    reg64_t reg_current_ic = rsi;
    reg64_t reg_bias = rdx;

    reg64_t reg_kj = rax;
    reg64_t reg_oi = rbx;
    reg64_t reg_kh = abi_not_param1;

    reg64_t reg_tmp = rbp;

    reg64_t reg_ic_loop = rdx;
    reg64_t reg_inp_loop = rsi;

    reg64_t reg_init_flag = r13;
    reg64_t reg_bias_ptr = param;

    reg64_t aux_reg_ic = r12;
    reg64_t reg_binp = rax;
    reg64_t reg_bout = r11;
    reg64_t aux1_reg_inp = rbx;
    reg64_t aux_reg_out = abi_not_param1;

    inline void prepare_output(int ur_w);
    inline void store_output(int ur_w, char pad_label);
    inline int compute_loop(int ur_w, int pad_l, int pad_r,
            const char *kh_label, char pad_label);

    void generate();
};

struct jit_avx512_mic_conv_bwd_data_kernel_f32: public jit_generator {
    jit_avx512_mic_conv_bwd_data_kernel_f32(jit_conv_conf_t ajcp): jcp(ajcp)
    {
        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &diff_src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &diff_dst_d);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    enum { typesize = sizeof(float) };

    reg64_t param      = abi_param1;
    reg64_t reg_dst     = r8;
    reg64_t reg_ker     = r9;
    reg64_t reg_src     = r10;

    reg64_t reg_dst_prf = r11;
    reg64_t reg_ker_prf = r12;
    reg64_t reg_src_prf = r13;

    reg64_t aux_reg_dst = r14;
    reg64_t aux_reg_ker = r15;

    reg64_t aux_reg_dst_prf = rsi;
    reg64_t aux_reg_ker_prf = rdx;

    reg64_t reg_kj = rax;
    reg64_t reg_oi = rbx;
    reg64_t reg_kh = abi_not_param1;

    reg64_t reg_current_ic = rsi;

    reg64_t reg_tmp = rbp;

    inline void prepare_output(int ur_w);
    inline void store_output(int ur_w);
    inline int compute_loop(int ur_w, int l_overflow, int r_overflow);
    void generate();
};

struct jit_avx512_mic_conv_bwd_weights_kernel_f32 : public jit_generator {
    jit_avx512_mic_conv_bwd_weights_kernel_f32(jit_conv_conf_t ajcp) : jcp(ajcp)
    {
        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }

    static status_t init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &diff_weights_d,
        const memory_desc_wrapper &diff_dst_d);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    enum {typesize = sizeof(float)};

    reg64_t param = abi_param1;
    reg64_t reg_input = rax;
    reg64_t reg_kernel = rdx;
    reg64_t reg_output = rsi;
    reg64_t b_ic = abi_not_param1;
    reg64_t kj = r8;
    reg64_t reg_kh  = r9;
    reg64_t reg_ur_w_trips  = r10;
    reg64_t reg_oj = r15;
    reg64_t reg_ih_count = rbx;

    reg64_t aux_reg_bcast_data = r14;
    reg64_t aux_reg_load_data  = r15;

    reg64_t bk_loop = rdx;
    reg64_t bcast_loop = rsi;

    reg64_t reg_init_flag = r13;

    reg64_t aux_bk_loop = r12;
    reg64_t reg_bbcast = rax;
    reg64_t reg_bload = r11;
    reg64_t aux1_reg_bcast_data = rbx;
    reg64_t aux_reg_out_data = abi_not_param1;
    reg64_t reg_output_loadblk_step = abi_param1;

    inline void compute_oh_step_unroll_ow_icblock(int ic_block_step,
        int max_ur_w);
    inline void oh_step_comeback_pointers();
    inline void compute_oh_step_unroll_ow(int ic_block_step, int max_ur_w);
    inline void compute_ic_block_step(int ur_w, int pad_l, int pad_r,
        int ic_block_step, int input_offset, int kernel_offset,
        int output_offset);
    inline void compute_oh_step_common(int ic_block_step, int max_ur_w);
    inline void compute_oh_step_disp();
    inline void compute_oh_loop_common();
    void generate();
};


}
}
}

#endif
