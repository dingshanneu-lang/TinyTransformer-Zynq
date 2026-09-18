/*
 * xform_core.cpp - Fixed-point 2-layer TinyTransformer inference core.
 *
 * LOW-RESOURCE serial variant for Zynq-7010 (xc7z010clg400-1).
 * Original fully-parallel version kept at xform_core_parallel_backup.cpp.
 *
 * Interface (s_axilite, bundle CTRL):
 *   weights[522]  int16 Q12  (layout identical to gen_reference.py export)
 *   in[16]        int16 Q12  (4x4 features, row-major)
 *   out[16]       int16 Q12  ([0]=class0 logit, [1]=class1 logit, rest 0)
 *
 * Numerics mirror gen_reference.py FixedQuant bit-for-bit:
 *   - all accumulation in int64
 *   - qround(a,b) = (a + (b>>1)) / b   (C-truncating), b is a power of two
 *   - restoring (shift-subtract) 64-bit division implements tdiv(., b) with
 *     the exact C-truncating semantics; glow positions matched HLS stability
 *   - sigmoid / exp from xform_tables.h LUTs
 *   - layernorm inv-sqrt via integer newton in 24 iterations
 */
#include "xform_tables.h"
#include <stdint.h>

#define Q8S   256

/* C-truncating divide a/b, b>0. Identical to python tdiv(a,b).
 * Implemented as restoring shift-subtract to avoid the 64-bit divider macro.
 * #pragma UNROLL factor=1 keeps a single independent multiplier-free netlist. */
static int64_t tdiv(int64_t a, int64_t b) {
    uint64_t ua, ub, q = 0, r = 0;
    int neg;
    if (b <= 0) return 0;
    neg = (a < 0);
    ua = neg ? (uint64_t)(-(a + 1)) + 1 : (uint64_t)a;
    ub = (uint64_t)b;
    for (int i = 63; i >= 0; i--) {
#pragma HLS UNROLL factor=1
        r = (r << 1) | ((ua >> i) & 1);
        if (r >= ub) {
            r -= ub;
            q |= (1ULL << i);
        }
    }
    return neg ? -(int64_t)q : (int64_t)q;
}
/* qround(a,b)=(a+(b>>1))/b with b=2^k; identical to python qround. */
static int64_t qround_p2(int64_t a, int k) {
    return tdiv(a + (int64_t)(1ULL << (k - 1)), (int64_t)(1ULL << k));
}
static int16_t clip16(int64_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}
static int64_t invsqrt(int64_t v) {
    if (v <= 0) return 0;
    int64_t s = v;
    for (int i = 0; i < 24; i++) {
#pragma HLS UNROLL factor=1
        if (s == 0) break;
        s = (s + tdiv(v, s)) >> 1;
    }
    while (s * s > v) s--;
    while ((s + 1) * (s + 1) <= v) s++;
    if (s <= 0) return 0;
    return tdiv((1LL << 24) + (s >> 1), s);
}

static int16_t sigmoid(int16_t x) {
#pragma HLS BIND_STORAGE variable=SIG_LUT type=ROM_1P impl=BRAM
    // SIG_LUT is 256 entries, Q12. Input x is Q12 (-16384..16383).
    // Map Q12 [-16384, 16383] -> index [0, 255] via >> 7
    int idx = (int)((x + 16384) >> 7);
    if (idx < 0) idx = 0;
    if (idx > 255) idx = 255;
    return SIG_LUT[idx];
}
static int16_t gelu(int16_t x) {
    int64_t z = tdiv((int64_t)x * 6971, 4096);
    int16_t sg = sigmoid((int16_t)z);
    return clip16(qround_p2((int64_t)x * sg, 12));
}

static void softmax_row(const int32_t scores[4], int16_t p[4]) {
#pragma HLS BIND_STORAGE variable=EXP_LUT type=ROM_1P impl=BRAM
    int64_t mx = scores[0];
    for (int i = 1; i < 4; i++) if (scores[i] > mx) mx = scores[i];
    int64_t es[4];
    int64_t tot = 0;
    int idx;
    for (int i = 0; i < 4; i++) {
        int64_t arg = mx - scores[i];
        idx = (arg > 511 * 64) ? 511 : (int)(arg >> 6);
        es[i] = EXP_LUT[idx];
        tot += es[i];
    }
    if (tot <= 0) tot = 1;
    int64_t invN = tdiv((1LL << 24) + (tot >> 1), tot);
    for (int i = 0; i < 4; i++) p[i] = clip16(qround_p2(es[i] * invN, 12));
}

static void layernorm_vec(const int16_t x[4], const int16_t gamma[4],
                          const int16_t beta[4], int16_t y[4]) {
    int64_t s = 0;
    for (int i = 0; i < 4; i++) s += x[i];
    int64_t mu = qround_p2(s, 2);
    int64_t ss = 0;
    for (int i = 0; i < 4; i++) { int64_t d = x[i] - mu; ss += d * d; }
    int64_t var = tdiv(ss, 4);
    int64_t inv = invsqrt(var);
    for (int i = 0; i < 4; i++) {
        int64_t d = x[i] - mu;
        int64_t yn = qround_p2(d * inv, 12);
        y[i] = clip16(qround_p2(yn * gamma[i], 12) + beta[i]);
    }
}

void xform_core(const int16_t weights[522], const int16_t in[16], int16_t out[16]) {
#pragma HLS INTERFACE s_axilite port=weights bundle=CTRL
#pragma HLS INTERFACE s_axilite port=in      bundle=CTRL
#pragma HLS INTERFACE s_axilite port=out     bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return  bundle=CTRL
#pragma HLS ALLOCATION function instances=qround_p2 limit=1
#pragma HLS ALLOCATION function instances=tdiv limit=2
#pragma HLS ALLOCATION function instances=invsqrt limit=1
#pragma HLS ALLOCATION instances=div limit=2 operation
#pragma HLS ALLOCATION instances=mul limit=2 operation
#pragma HLS ALLOCATION instances=add limit=4 operation
#pragma HLS ALLOCATION instances=sub limit=2 operation
#pragma HLS ALLOCATION instances=icmp limit=2 operation

    const int E = 4, FF = 16, seq = 4, layers = 2;

    int16_t w[522];
    int16_t x[16];           // x[t*E+e]
    int16_t track[16];
    int16_t res[16];

    for (int i = 0; i < 522; i++) w[i] = weights[i];
    for (int i = 0; i < 16; i++)  x[i] = in[i];

    int off = 0;
    for (int layer = 0; layer < layers; layer++) {
        const int16_t *qkv_w = &w[off];       off += 48;
        const int16_t *qkv_b = &w[off];       off += 12;
        const int16_t *out_w = &w[off];       off += 16;
        const int16_t *out_b = &w[off];       off += 4;
        const int16_t *fc1_w = &w[off];       off += 64;
        const int16_t *fc1_b = &w[off];       off += 16;
        const int16_t *fc2_w = &w[off];       off += 64;
        const int16_t *fc2_b = &w[off];       off += 4;
        const int16_t *ln1_g = &w[off];       off += 4;
        const int16_t *ln1_b = &w[off];       off += 4;
        const int16_t *ln2_g = &w[off];       off += 4;
        const int16_t *ln2_b = &w[off];       off += 4;

        int16_t Q[4][4], K[4][4], V[4][4];

        // projections (qkv row e -> cols 0..3/4..7/8..11), serial
        for (int t = 0; t < seq; t++) {
        #pragma HLS LOOP_FLATTEN off
            for (int e = 0; e < E; e++) {
                int64_t aq = 0, ak = 0, av = 0;
                for (int j = 0; j < E; j++) {
                    int64_t qv = (int64_t)qkv_w[e * 12 + 0 * 4 + j];
                    int64_t kv = (int64_t)qkv_w[e * 12 + 1 * 4 + j];
                    int64_t vv = (int64_t)qkv_w[e * 12 + 2 * 4 + j];
                    aq += qv * x[t * E + j];
                    ak += kv * x[t * E + j];
                    av += vv * x[t * E + j];
                }
                Q[t][e] = clip16(qround_p2(aq, 12) + qkv_b[e]);
                K[t][e] = clip16(qround_p2(ak, 12) + qkv_b[E + e]);
                V[t][e] = clip16(qround_p2(av, 12) + qkv_b[2 * E + e]);
            }
        }

        // attention scores (scaled /2, d_head=4), serial
        int32_t scores[4][4];
        int16_t attn[4][4];
        for (int t = 0; t < seq; t++) {
        #pragma HLS LOOP_FLATTEN off
            for (int k = 0; k < seq; k++) {
                int64_t acc = 0;
                for (int e = 0; e < E; e++) {
                    acc += (int64_t)Q[t][e] * K[k][e];
                }
                scores[t][k] = (int32_t)tdiv(qround_p2(acc, 12), 2);
            }
        }

        for (int t = 0; t < seq; t++) {
        #pragma HLS LOOP_FLATTEN off
            int16_t sm[4];
            softmax_row(scores[t], sm);
            for (int e = 0; e < E; e++) {
                int64_t acc = 0;
                for (int k = 0; k < seq; k++) {
                    acc += (int64_t)sm[k] * V[k][e];
                }
                attn[t][e] = clip16(qround_p2(acc, 12));
            }
        }

        // out projection + residual + ln1
        for (int t = 0; t < seq; t++) {
        #pragma HLS LOOP_FLATTEN off
            for (int e = 0; e < E; e++) {
                int64_t acc = 0;
                for (int j = 0; j < E; j++) {
                    acc += (int64_t)out_w[j * E + e] * attn[t][j];
                }
                int64_t z = clip16(qround_p2(acc, 12) + out_b[e]);
                res[t * E + e] = clip16((int64_t)x[t * E + e] + z);
            }
        }
        for (int t = 0; t < seq; t++)
            layernorm_vec(&res[t * E], ln1_g, ln1_b, &track[t * E]);

        // ffn
        for (int t = 0; t < seq; t++) {
            int16_t h[16], g[16], u[4];
            for (int f = 0; f < FF; f++) {
                #pragma HLS LOOP_FLATTEN off
                int64_t acc = 0;
                for (int j = 0; j < E; j++) {
                    acc += (int64_t)fc1_w[j * FF + f] * track[t * E + j];
                }
                h[f] = clip16(qround_p2(acc, 12) + fc1_b[f]);
                g[f] = gelu(h[f]);
            }
            for (int e = 0; e < E; e++) {
                #pragma HLS LOOP_FLATTEN off
                int64_t acc = 0;
                for (int f = 0; f < FF; f++) {
                    acc += (int64_t)fc2_w[f * E + e] * g[f];
                }
                u[e] = clip16(qround_p2(acc, 12) + fc2_b[e]);
            }
            for (int e = 0; e < E; e++) {
                int64_t r = clip16((int64_t)track[t * E + e] + u[e]);
                res[t * E + e] = r;
            }
            layernorm_vec(&res[t * E], ln2_g, ln2_b, &x[t * E]);
        }
    }

    // output projection: 16 -> 2, serial
    const int16_t *op_w = &w[off];  off += 32;
    const int16_t *op_b = &w[off];  off += 2;
    int16_t logits[2];
    for (int c = 0; c < 2; c++) {
        #pragma HLS LOOP_FLATTEN off
        int64_t acc = 0;
        for (int i = 0; i < 16; i++) {
            acc += (int64_t)op_w[i * 2 + c] * x[i];
        }
        logits[c] = clip16(qround_p2(acc, 12) + op_b[c]);
    }
    for (int i = 0; i < 16; i++) out[i] = 0;
    out[0] = logits[0];
    out[1] = logits[1];
}