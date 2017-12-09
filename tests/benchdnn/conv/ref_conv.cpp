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

#include "conv/conv.hpp"

namespace conv {

void compute_ref_fwd(const prb_t *p, dnn_mem_t &src_m,
        dnn_mem_t &wei_m, dnn_mem_t &bia_m, dnn_mem_t &dst_m) {
    auto ker = [&](float &d, int g, int mb, int oc, int oh, int ow) {
        for (int ic = 0; ic < p->ic/p->g; ++ic) {
            for (int kh = 0; kh < p->kh; ++kh) {
                const int ih = oh * p->sh - p->ph + kh * (p->dh + 1);
                if (ih < 0 || ih >= p->ih) continue;

                for (int kw = 0; kw < p->kw; ++kw) {
                    const int iw = ow * p->sw - p->pw + kw * (p->dw + 1);
                    if (iw < 0 || iw >= p->iw) continue;

                    size_t src_off = src_off_f(p, mb, g, ic, ih, iw);
                    size_t wei_off = wei_off_f(p, g, oc, ic, kh, kw);
                    d += ((float*)src_m)[src_off] * ((float*)wei_m)[wei_off];
                }
            }
        }
    };

    auto maybe_scale = [&](float &d, int oc) {
        if (!p->attr.oscale.is_def()) {
            using policy_t = attr_t::scale_t::policy_t;
            const auto &s = p->attr.oscale;
            if (s.policy == policy_t::COMMON) {
                d *= s.scale;
            } else {
                d *= p->scales[oc];
            }
        }
    };

    auto maybe_post_ops = [&](float &conv_res, float dst) {
        const auto &ops = p->attr.post_ops;
        for (int idx = 0; idx < ops.len; ++idx) {
            using pk = attr_t::post_ops_t::kind_t;
            const auto &e = ops.entry[idx];
            switch (e.kind) {
            case pk::SUM:
                conv_res += e.sum.scale * dst;
                break;
            case pk::RELU:
                conv_res = e.eltwise.scale * (conv_res < 0 ? 0 : conv_res);
                break;
            default:
                assert(!"unknown attr::post_ops::kind");
            }
        }
    };

#   pragma omp parallel for collapse(5)
    for (int g = 0; g < p->g; ++g) {
    for (int mb = 0; mb < p->mb; ++mb) {
        for (int oc = 0; oc < p->oc/p->g; ++oc) {
        for (int oh = 0; oh < p->oh; ++oh) {
        for (int ow = 0; ow < p->ow; ++ow) {
            const size_t dst_off = dst_off_f(p, mb, g, oc, oh, ow);
            float &dst = ((float*)dst_m)[dst_off];

            float conv_res = 0;
            ker(conv_res, g, mb, oc, oh, ow);

            if (p->dir & FLAG_BIA) {
                const size_t bia_off = bia_off_f(p, g, oc);
                conv_res += ((float*)bia_m)[bia_off];
            }

            if (p->merge == RELU && conv_res < 0)
                conv_res = 0;

            maybe_scale(conv_res, g * p->oc / p->g + oc);
            maybe_post_ops(conv_res, dst);

            dst = conv_res;
        }
        }
        }
    }
    }
}

void compute_ref_bwd_d(const prb_t *p, dnn_mem_t &diff_src_m,
        dnn_mem_t &wei_m, dnn_mem_t &diff_dst_m) {
    auto ker = [&](float &ds, int g, int mb, int ic, int ih, int iw) {
        for (int oc = 0; oc < p->oc/p->g; ++oc) {
            for (int kh = 0; kh < p->kh; ++kh) {
                int oh = ih - kh * (p->dh + 1) + p->ph;
                if (oh < 0 || oh % p->sh) continue;
                oh /= p->sh;
                if (oh >= p->oh) continue;

                for (int kw = 0; kw < p->kw; ++kw) {
                    int ow = iw - kw * (p->dw + 1) + p->pw;
                    if (ow < 0 || ow % p->sw) continue;
                    ow /= p->sw;
                    if (ow >= p->ow) continue;

                    size_t dst_off = dst_off_f(p, mb, g, oc, oh, ow);
                    size_t wei_off = wei_off_f(p, g, oc, ic, kh, kw);
                    ds += ((float*)diff_dst_m)[dst_off]
                        * ((float*)wei_m)[wei_off];
                }
            }
        }
    };

#   pragma omp parallel for collapse(5)
    for (int g = 0; g < p->g; ++g) {
    for (int mb = 0; mb < p->mb; ++mb) {
        for (int ic = 0; ic < p->ic/p->g; ++ic) {
        for (int ih = 0; ih < p->ih; ++ih) {
        for (int iw = 0; iw < p->iw; ++iw) {
            size_t src_off = src_off_f(p, mb, g, ic, ih, iw);
            float &ds = ((float*)diff_src_m)[src_off];
            ds = 0;
            ker(ds, g, mb, ic, ih, iw);
        }
        }
        }
    }
    }
}

void compute_ref_bwd_w(const prb_t *p, dnn_mem_t &src_m,
        dnn_mem_t &diff_wei_m, dnn_mem_t &diff_bia_m, dnn_mem_t &diff_dst_m) {
    auto ker = [&](float &dw, int g, int oc, int ic, int kh, int kw) {
        for (int mb = 0; mb < p->mb; ++mb) {
            for (int oh = 0; oh < p->oh; ++oh) {
            for (int ow = 0; ow < p->ow; ++ow) {
                const int ih = oh * p->sh - p->ph + kh * (p->dh + 1);
                const int iw = ow * p->sw - p->pw + kw * (p->dw + 1);
                if (ih < 0 || ih >= p->ih) continue;
                if (iw < 0 || iw >= p->iw) continue;

                size_t src_off = src_off_f(p, mb, g, ic, ih, iw);
                size_t dst_off = dst_off_f(p, mb, g, oc, oh, ow);
                dw += ((float*)diff_dst_m)[dst_off]
                    * ((float*)src_m)[src_off];
            }
            }
        }
    };

#   pragma omp parallel for collapse(5)
    for (int g = 0; g < p->g; ++g) {
        for (int oc = 0; oc < p->oc/p->g; ++oc) {
        for (int ic = 0; ic < p->ic/p->g; ++ic) {
            for (int kh = 0; kh < p->kh; ++kh) {
            for (int kw = 0; kw < p->kw; ++kw) {
                size_t wei_off = wei_off_f(p, g, oc, ic, kh, kw);
                float &dw = ((float*)diff_wei_m)[wei_off];
                dw = 0;
                ker(dw, g, oc, ic, kh, kw);
            }
            }
        }
        }
    }

    if (!(p->dir & FLAG_BIA)) return;

#   pragma omp parallel for collapse(2)
    for (int g = 0; g < p->g; ++g) {
        for (int oc = 0; oc < p->oc/p->g; ++oc) {
            size_t bia_off = bia_off_f(p, g, oc);
            float &db = ((float*)diff_bia_m)[bia_off];
            db = 0;

            for (int mb = 0; mb < p->mb; ++mb) {
                for (int oh = 0; oh < p->oh; ++oh) {
                for (int ow = 0; ow < p->ow; ++ow) {
                    size_t dst_off = dst_off_f(p, mb, g, oc, oh, ow);
                    db += ((float*)diff_dst_m)[dst_off];
                }
                }
            }
        }
    }
}

}
