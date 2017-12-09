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

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "mkldnn.h"

#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"
#include "norm.hpp"

#include "bnorm/bnorm.hpp"

namespace bnorm {

static int prepare_fwd(const prb_t *p, dnn_mem_t &src, dnn_mem_t &mean,
        dnn_mem_t &var, dnn_mem_t &ss) {
    /** Idea: choose src[] values so that both mean and variance are computed
     * exactly (independently of the order of the computations).
     *
     * The `exactness` is achieved via [a1]: src[i] + src[i+1] = 2 * mean.
     *
     * The variation in src is allowed in the last flex_bits bits.
     * If the sequence (L) is too big (flex_bits <= min_flex_bits), the mean
     * value is set to 0 and src is partially filled with zeros (according to
     * density so that at least want_flex_bits is reserved for src variation.
     * Once src is set, variance is computed.
     *
     * ALG_0: mean is set to 0
     * ALG_1: mean is set to 2^p, where p \in {-2, -1, ..., 4}
     * ALG_AUTO: choose between ALG_0 and ALG_1 automatically */
    const int exact_bits = 24;
    const int L = p->mb * p->ih * p->iw;
    const int logL = (int)ceilf(log2f(L));

    assert(logL <= 0 || (1<<(logL-1)) < L);
    assert(L <= (1<<logL));

    const int min_flex_bits = 3;
    const int want_flex_bits = 6;

    check_alg_t alg = p->check_alg;
    if (alg == ALG_AUTO) /* choose appropriate checking algorithm */
        alg = (exact_bits - logL) / 2 - 1 >= min_flex_bits ? ALG_1 : ALG_0;

    const int flex_bits = alg == ALG_0
        ? want_flex_bits : ((exact_bits - logL) / 2 - 1);

    if (flex_bits < min_flex_bits)
        return FAIL;

    const int flex_mask = (1 << flex_bits) - 1;

    /* density: (exact_bits - log_2(L * density)) / 2 >= flex_bits */
    const float density = alg == ALG_0
        ? 1.f * (1 << (exact_bits - 2 * flex_bits)) / L : 1.f;
    assert((exact_bits - ceilf(log2f(L * density))) / 2 >= flex_bits);

    print(6, "check_alg: %s, density = %g, flex_bits = %d\n",
            check_alg2str(ALG_0), density, flex_bits);

#   pragma omp parallel for
    for (int c = 0; c < p->ic; ++c) {
        const float m = ((float *)mean)[c] =
            alg == ALG_0 ? 0.f : 0.25f * (1 << (c % 7));
        float v = 0; /* current variance */

        for (int mb = 0; mb < p->mb; ++mb) {
            int l_base = mb * p->ih * p->iw + c * 239 * 2; // l[0] must be even
            float *s = (float *)src + data_off(p, mb, c, 0, 0);

            for (int h = 0; h < p->ih; ++h)
            for (int w = 0; w < p->iw; ++w) {
                const int sp = h * p->iw + w;
                const int l = l_base + sp;

                if (alg == ALG_0 && (l/2 * 257) % 379 > density * 379) {
                    s[sp] = 0;
                    continue;
                }

                const int gen = l / 2 * 1637 & flex_mask;
                const int sgn = l % 2 == 0 ? 1 : -1; /* [a1] */
                const float f = 1.f * sgn * gen / (1 << flex_bits);

                s[sp] = alg == ALG_0 ? f : m * (1.f + f);
                v += (s[sp] - m) * (s[sp] - m);
            }
        }

        if (L % 2 == 1) {
            /* if L is odd -- fix the last value so that mean == m */
            const auto off = data_off(p, p->mb - 1, c, p->ih - 1, p->iw - 1);
            float &s = ((float *)src)[off];
            v += m * m - (s - m) * (s - m);
            s = m;
        }

        ((float *)var)[c] = v / (p->mb * p->ih * p->iw);

        if (p->flags & USE_SCALESHIFT) {
            ((float *)ss)[c] = 1.f / 8 * (1 << (c % 7));
            ((float *)ss)[p->ic + c] = ((float *)ss)[c] / 64;
        } else {
            ((float *)ss)[c] = 1;
            ((float *)ss)[p->ic + c] = 0;
        }
    }

    return OK;
}

/** @brief L = 2^k * P, P % 2 != 0 */
static void decompose2(int L, int &k, int &P) {
    P = L;
    for (k = 0; P % 2 == 0; ++k)
        P /= 2;
}

static int prepare_bwd(const prb_t *p, dnn_mem_t &src, dnn_mem_t &d_dst,
        dnn_mem_t &mean, dnn_mem_t &var, dnn_mem_t &ss) {
    const int exact_bits = 24;

    const int L = p->mb * p->ih * p->iw;
    if (L < 2)
        return FAIL;

    /** Stabilization idea...
     * Since
     *      d_src = func(d_beta / L, d_gamma' / L, ...)
     * try to make d_beta = L / 2^t_beta and d_gamma' = L / 2^t_gamma,
     * where both t_beta and t_gamma are in {1, .., max_k}.
     * Currently, with no obvious reason, max_k set to 4.
     *
     * Here d_gamma' = d_gamma / sqrt(var + eps).
     * We might hope that division by L would be exact in that case,
     * but that might happen iff L is less than 2^exact_bits, hence
     * restriction [r1]. */

    int k, P;
    decompose2(L, k, P);

    int log2P = (int)ceilf(log2f(P));
    if (log2P >= exact_bits)
        return FAIL; /* [r1] */

    const int max_k = 4;
    if (k > max_k && exact_bits - log2P > max_k + 4) {
        log2P += (k - max_k);
        P <<= k - max_k;
        k = max_k;
    }

    print(5, "prep_bwd: k:%d, P:%d log2P:%d\n", k, P, log2P);

#   pragma omp parallel for
    for (int c = 0; c < p->ic; ++c) {
        const float m = ((float *)mean)[c] = c % 2;

        /* var + eps \in {1/4, 1, 4} */
        const float ve_denom = 4.f / (1 << 2 * (c % 3));
        ((float *)var)[c] = ve_denom - p->eps;

        const int db_p2 = (c * 127 % 7);
        const float factor_dd = 1.f / (1 << db_p2);
        const float factor_f = 0.5f;

        const float target_db = factor_dd * P;
        const float target_dg = ve_denom * 2 * target_db;

        float dg = 0, db = 0; /* current d_beta and d_gamma */
        for (int mb = 0; mb < p->mb; ++mb) {
            const int l_base = mb * p->ih * p->iw;

            const auto off = data_off(p, mb, c, 0, 0);
            float *s = (float *)src + off;
            float *dd = (float *)d_dst + off;

            for (int h = 0; h < p->ih; ++h)
            for (int w = 0; w < p->iw; ++w) {
                const int sp = h * p->iw + w;
                if (l_base + sp + 2 >= L) continue; /* last 2 are special */
                const int l = l_base + sp * 7 + c * 19 + mb * 13;

                const int sgn_dd = db < target_db ? 1 : -1;
                dd[sp] = sgn_dd * factor_dd * (1 + (l * 3 % 32));
                db += dd[sp];

                const int sgn_f = dg < target_dg ? 1 : -1;
                const float f = sgn_f * factor_f * (2 + (l * 7 % 15));
                dg += f * dd[sp];
                s[sp] = f + m;
            }
        }

        if (1) {
            /* the last 2 elements in src and d_dst are set, so that:
             *      db == target_db
             *      dg == target_dg
             * For this we need to solve the system:
             *      d_dst[l1]           + d_dst[l0]           = target_db - db
             *      d_dst[l1] * src[l1] + d_dst[l0] * src[l0] = target_dg - dg
             *
             * Here l0 -- last index, l1 -- last but one.
             * More over, let's assume src[l1] = 1 and src[l0] = -1. */
            size_t l0 = data_off(p, p->mb - 1, c, p->ih - 1, p->iw - 1);
            size_t l1 = l0 - 1;
            if (p->ih == 1 && p->iw == 1)
                l1 = data_off(p, p->mb - 2, c, p->ih - 1, p->iw - 1);

            ((float *)src)[l1] = 1.f;
            ((float *)src)[l0] = -1.f;

            float f1 = ((target_db - db) + (target_dg - dg)) /2;
            float f0 = ((target_db - db) - (target_dg - dg)) /2;

            ((float *)d_dst)[l1] = f1 + m;
            ((float *)d_dst)[l0] = f0 + m;
        }

        if (p->flags & USE_SCALESHIFT) {
            ((float *)ss)[c] = 1.f / 2 * (1 << (c % 7));
            ((float *)ss)[p->ic + c] = ((float *)ss)[c] / 64;
        } else {
            ((float *)ss)[c] = 1;
            ((float *)ss)[p->ic + c] = 0;
        }
    }

    return OK;
}

static int compare(const prb_t *p, data_kind_t kind, const dnn_mem_t &fp_mem,
        const dnn_mem_t &dt_mem, res_t *r) {
    const char *skind = data_kind2str(kind);
    const float eps = p->dir & FLAG_FWD
        ? (kind == DATA ? 5e-7 : 0)
        : (kind == DATA ? 2e-7 : 0);

    /* With all the stability tricks bwd_d is still pretty unstable.
     * So let's rely on relative error in L1, L2, and L_inf norms.
     * TODO: make computations for bwd_d more stable and use `L0` here. */
    const bool rely_on_norm = false
        || (kind == DATA && (p->dir & FLAG_BWD) && (p->flags | GLOB_STATS));

    const size_t nelems = kind == DATA
        ? (size_t)p->mb * p->ic * p->ih * p->iw
        : (size_t)p->ic * (kind == SS ? 2 : 1);
    r->total += rely_on_norm ? 1 : nelems;

    diff_norm_t diff_norm;

    for (size_t i = 0; i < nelems; ++i) {
        const float fp = ((const float *)fp_mem)[i];
        const float dt = ((const float *)dt_mem)[i];
        diff_norm.update(fp, dt);

        if (rely_on_norm)
            continue;

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        const bool ok = (fabs(fp) > 1e-5 ? rel_diff : diff) <= eps;

        r->errors += !ok;

        bool dump = false
            || (!ok && (r->errors < 10 || verbose >= 10))
            || (verbose >= 50 && i < 30);
        if (dump) {
            const int ind_str_len = 32;
            char ind_str[ind_str_len] = {'\0'};
            if (kind == DATA) {
                int mb, c, h, w;
                inv_data_off(p, i, mb, c, h, w);
                snprintf(ind_str, ind_str_len, "%d,%d,%d,%d", mb, c, h, w);
            } else if (kind == SS) {
                snprintf(ind_str, ind_str_len, "%d,%d",
                        (int)i / p->ic, (int)i % p->ic);
            } else {
                snprintf(ind_str, ind_str_len, "%d", (int)i);
            }

            print(0, "[%lu][%s%s][%s] fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                    i, p->dir & FLAG_BWD ? "D_" : "", skind, ind_str,
                    fp, dt, diff, rel_diff);
        }
    }

    diff_norm.done();

    if (rely_on_norm) {
        r->errors += false
            || diff_norm.rel_diff(norm_t::L1) > eps
            || diff_norm.rel_diff(norm_t::L2) > eps
            || diff_norm.rel_diff(norm_t::L8) > eps;
    }

    if (r->errors || verbose >= 5) {
        const int vl = r->errors ? 0 : 2;
        print(vl, "@@@ [%s%s] diff: l0(``%g``) "
                "l1:(%g,%g,%g,``%g``) "
                "l2:(%g,%g,%g,``%g``) "
                "l8:(%g,%g,%g,``%g``)\n",
                p->dir & FLAG_BWD ? "D_" : "", skind,
                diff_norm.rel_diff(norm_t::L0),
                diff_norm.a_[norm_t::L1], diff_norm.b_[norm_t::L1],
                diff_norm.diff_[norm_t::L1], diff_norm.rel_diff(norm_t::L1),
                diff_norm.a_[norm_t::L2], diff_norm.b_[norm_t::L2],
                diff_norm.diff_[norm_t::L2], diff_norm.rel_diff(norm_t::L2),
                diff_norm.a_[norm_t::L8], diff_norm.b_[norm_t::L8],
                diff_norm.diff_[norm_t::L8], diff_norm.rel_diff(norm_t::L8));
    }

    if (r->errors)
        r->state = FAILED;

    if (r->state == UNTESTED)
        r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

static int init_pd(const prb_t *p, mkldnn_batch_normalization_desc_t &bd,
        mkldnn_primitive_desc_t &bpd, res_t *r) {
    mkldnn_memory_desc_t data_d;
    mkldnn_dims_t data_dims = {p->mb, p->ic, p->ih, p->iw};
    DNN_SAFE(mkldnn_memory_desc_init(&data_d, 4, data_dims, p->dt, p->fmt),
            WARN);

    auto flags = (mkldnn_batch_normalization_flag_t)p->flags;
    if (p->dir & FLAG_FWD) {
        auto prop = p->dir & FLAG_INF
            ? mkldnn_forward_inference : mkldnn_forward_training;
        DNN_SAFE(mkldnn_batch_normalization_forward_desc_init(&bd, prop,
                    &data_d, p->eps, flags), WARN);

    } else {
        auto prop = p->dir & FLAG_WEI
            ? mkldnn_backward : mkldnn_backward_data;
        DNN_SAFE(mkldnn_batch_normalization_backward_desc_init(&bd, prop,
                    &data_d, &data_d, p->eps, flags), WARN);
    }

    auto mkldnn_attr = create_mkldnn_attr(p->attr, 1, NULL);

    mkldnn_primitive_desc_t hint_fwd_pd = NULL;
    if (p->dir & FLAG_BWD) {
        mkldnn_batch_normalization_desc_t bd_fwd;
        DNN_SAFE(mkldnn_batch_normalization_forward_desc_init(&bd_fwd,
                    mkldnn_forward_training, &data_d, p->eps, flags), WARN);
        DNN_SAFE(mkldnn_primitive_desc_create_v2(&hint_fwd_pd, &bd_fwd, NULL,
                    engine, NULL), WARN);
    }
    mkldnn_status_t init_status = mkldnn_primitive_desc_create_v2(&bpd, &bd,
            mkldnn_attr, engine, hint_fwd_pd);

    mkldnn_primitive_desc_destroy(hint_fwd_pd);
    mkldnn_primitive_attr_destroy(mkldnn_attr);

    if (init_status == mkldnn_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    const char *impl_str = query_impl_info(bpd);
    if (maybe_skip(skip_impl, impl_str)) {
        print(2, "SKIPPED: mkldnn implementation: %s\n", impl_str);
        DNN_SAFE(mkldnn_primitive_desc_destroy(bpd), WARN);
        return r->state = SKIPPED, OK;
    } else {
        print(5, "mkldnn implementation: %s\n", impl_str);
    }

    return OK;
}

int doit(const prb_t *p, res_t *r) {
    res_t res_zero{};
    *r = res_zero;

    mkldnn_batch_normalization_desc_t bd;
    mkldnn_primitive_desc_t bpd;
    mkldnn_primitive_t b{};

    SAFE(init_pd(p, bd, bpd, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED)
        return OK;

    const auto fp = mkldnn_f32;
    auto &data_dt_d = bd.data_desc;

    const mkldnn_dims_t dims1d = {p->ic};
    const mkldnn_dims_t dims2d = {2, p->ic};

    dnn_mem_t data_fp(data_dt_d, fp, mkldnn_nchw),
              data_dt(data_dt_d);
    dnn_mem_t d_data_fp(data_dt_d, fp, mkldnn_nchw),
              d_data_dt(data_dt_d);

    dnn_mem_t mean_fp(1, dims1d, fp, mkldnn_x),
              mean_dt(mean_fp.md_);
    dnn_mem_t var_fp(1, dims1d, fp, mkldnn_x),
              var_dt(var_fp.md_);

    dnn_mem_t ss_fp(2, dims2d, fp, mkldnn_nc),
              ss_dt(ss_fp.md_);
    dnn_mem_t d_ss_fp(2, dims2d, fp, mkldnn_nc),
              d_ss_dt(d_ss_fp.md_);

    if (p->dir & FLAG_FWD) {
        if (prepare_fwd(p, data_fp, mean_fp, var_fp, ss_fp) != OK)
            return r->state = MISTRUSTED, OK;

        mkldnn_primitive_at_t inputs[4];
        const_mkldnn_primitive_t outputs[3];

        int idx = 0;

        SAFE(data_dt.reorder(data_fp), WARN);
        inputs[idx++] = {data_dt.p_, 0};

        if (p->flags & GLOB_STATS) {
            SAFE(mean_dt.reorder(mean_fp), WARN);
            SAFE(var_dt.reorder(var_fp), WARN);
            inputs[idx++] = {mean_dt.p_, 0};
            inputs[idx++] = {var_dt.p_, 0};
        }
        if (p->flags & USE_SCALESHIFT) {
            SAFE(ss_dt.reorder(ss_fp), WARN);
            inputs[idx++] = {ss_dt.p_, 0};
        }

        idx = 0;
        outputs[idx++] = data_dt.p_; /* always in-place so far... */
        if (!(p->flags & GLOB_STATS)) {
            outputs[idx++] = mean_dt.p_;
            outputs[idx++] = var_dt.p_;
        }

        DNN_SAFE(mkldnn_primitive_create(&b, bpd, inputs, outputs), WARN);
        SAFE(execute(b), WARN);
        if (bench_mode & CORR) {
            compute_ref_fwd(p, data_fp, mean_fp, var_fp, ss_fp, data_fp);
            if (!(p->flags & GLOB_STATS)) {
                SAFE(compare(p, MEAN, mean_fp, mean_dt, r), WARN);
                SAFE(compare(p, VAR, var_fp, var_dt, r), WARN);
            }
            dnn_mem_t data(data_dt.md_, fp, mkldnn_nchw);
            SAFE(data.reorder(data_dt), WARN);
            SAFE(compare(p, DATA, data_fp, data, r), WARN);
        }
    } else {
        if (prepare_bwd(p, data_fp, d_data_fp, mean_fp, var_fp, ss_fp) != OK)
            return r->state = MISTRUSTED, OK;

        mkldnn_primitive_at_t inputs[5];
        const_mkldnn_primitive_t outputs[2];

        int idx = 0;

        SAFE(data_dt.reorder(data_fp), WARN);
        inputs[idx++] = {data_dt.p_, 0};

        SAFE(mean_dt.reorder(mean_fp), WARN);
        SAFE(var_dt.reorder(var_fp), WARN);
        inputs[idx++] = {mean_dt.p_, 0};
        inputs[idx++] = {var_dt.p_, 0};

        SAFE(d_data_dt.reorder(d_data_fp), WARN);
        inputs[idx++] = {d_data_dt.p_, 0};

        if (p->flags & USE_SCALESHIFT) {
            SAFE(ss_dt.reorder(ss_fp), WARN);
            inputs[idx++] = {ss_dt.p_, 0};
        }

        idx = 0;
        outputs[idx++] = d_data_dt.p_; /* always in-place so far... */
        if ((p->flags & USE_SCALESHIFT) && (p->dir & FLAG_WEI))
            outputs[idx++] = d_ss_dt.p_;

        DNN_SAFE(mkldnn_primitive_create(&b, bpd, inputs, outputs), WARN);
        SAFE(execute(b), WARN);
        if (bench_mode & CORR) {
            compute_ref_bwd(p, data_fp, mean_fp, var_fp, d_data_fp, ss_fp,
                    d_data_fp, d_ss_fp);
            if ((p->flags & USE_SCALESHIFT) && (p->dir & FLAG_WEI))
                SAFE(compare(p, SS, d_ss_fp, d_ss_dt, r), WARN);
            dnn_mem_t d_data(d_data_dt.md_, fp, mkldnn_nchw);
            SAFE(d_data.reorder(d_data_dt), WARN);
            SAFE(compare(p, DATA, d_data_fp, d_data, r), WARN);
        }
    }

    if (bench_mode & PERF) {
        auto &t = r->timer;
        t.reset();
        while (true) {
            SAFE(execute(b), WARN);
            t.stamp();
            const bool stop = false
                || (fix_times_per_prb && t.times() >= fix_times_per_prb)
                || (!fix_times_per_prb
                        && t.total_ms() >= max_ms_per_prb
                        && t.times() >= min_times_per_prb);
            if (stop) break;
        }
    }

    return OK;
}

}
