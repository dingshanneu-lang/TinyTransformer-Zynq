#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_reference.py
================
Generates:
  1. Fixed-point LUT tables (sigmoid, exp) written into xform_tables.h
     so that the HLS C++ core and this Python reference are bit-identical.
  2. Full 522-int16 weight export (fixes train.py `expected=440` truncation).
  3. Test vectors (inputs + float reference logits + fixed-point reference logits)
     consumed by the HLS C testbench (tb.cpp).

Fixed-point convention: Q12 (1 << 12 == 4096).
  int16 storage, int32/int64 accumulators, final result re-quantized to Q12.

Usage:
  python gen_reference.py            # outputs into ./ref/
"""
import os
import json
import numpy as np
import torch

import sys
_ROOTS = [os.path.dirname(os.path.abspath(__file__))]
for _ in range(4):
    _ROOTS.append(os.path.dirname(_ROOTS[-1]))
sys.path.insert(0, _ROOTS[3])  # D:\ZynqTinyTransformerClassification = ...\TCPServer\pl\xform_core_hls\..\..\..\..

from train import TinyTransformer, MODEL_CONFIG

PROJECT_HLS = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(PROJECT_HLS, "ref")
DATA_DIR = "D:/ZynqTinyTransformerClassification/weights"
WEIGHT_BIN = os.path.join(DATA_DIR, "xform_weights.bin")

Q12 = 4096
QLOG2 = 12

# --------------------------------------------------------------------------
# fixed point helpers (must mirror xform_core.cpp exactly)
# --------------------------------------------------------------------------
def tdiv(a, b):
    """C-style truncation division (b > 0)."""
    if b <= 0:
        raise ValueError("b must be positive")
    if a >= 0:
        return a // b
    return -((-a) // b)

def qround(a, b):
    """(a + b/2) / b  truncating, exactly as C: (a + (b>>1)) / b."""
    return tdiv(a + (b >> 1), b)

def clip16(v):
    v = int(v)
    return max(-32768, min(32767, v))

def invsqrt(v):
    """integer 1/sqrt(v) scaled by 2^24, v>=0. Mirrors C"""
    if v <= 0:
        return 0
    s = v
    # integer sqrt via newton: s ~ floor(sqrt(v))
    for _ in range(24):
        if s == 0:
            break
        s = (s + tdiv(v, s)) >> 1
    while s * s > v:
        s -= 1
    while (s + 1) * (s + 1) <= v:
        s += 1
    if s <= 0:
        return 0
    return tdiv((1 << 24) + (s >> 1), s)  # round(2^24 / s)

# --------------------------------------------------------------------------
# LUT construction (identical formulas feed xform_tables.h)
# --------------------------------------------------------------------------
def make_sigmoid_lut(n=128):
    """sigmoid over x in [-4,4), step 1/16. index = (x+16384)>>8 (Q12)."""
    lut = []
    for i in range(n):
        xr = (i + 0.5) / 16.0 - 4.0
        v = 1.0 / (1.0 + np.exp(-xr))
        lut.append(int(np.clip(np.round(v * Q12), -32768, 32767)))
    return lut

def make_exp_lut(n=512):
    """exp(-t) for t in [0,8) step 1/64. index = t_q12 >> 6."""
    lut = []
    for i in range(n):
        t = i / 64.0
        v = np.exp(-t)
        lut.append(int(np.clip(np.round(v * Q12), 0, 32767)))
    return lut

SIG_LUT = make_sigmoid_lut(128)
EXP_LUT = make_exp_lut(512)

# --------------------------------------------------------------------------
# fixed-point forward (mirror of xform_core.cpp)
# --------------------------------------------------------------------------
class FixedQuant:
    def __init__(self, w):
        self.w = w  # np.int16 array 522

    @staticmethod
    def sigmoid(x):
        xc = max(-16384, min(16383, x))
        idx = (xc + 16384) >> 8
        return SIG_LUT[idx]

    @staticmethod
    def gelu(x):
        z = tdiv(int(x) * 6971, Q12)          # 1.702*x  (1.702*4096=6971)
        sg = FixedQuant.sigmoid(z)
        return clip16(qround(int(x) * sg, Q12))

    @staticmethod
    def softmax(row):
        mx = max(row)
        es = []
        for v in row:
            arg = mx - v                       # >= 0
            idx = min(511, (int(arg) >> 6))
            es.append(EXP_LUT[idx])
        tot = sum(es)
        if tot <= 0:
            tot = 1
        invN = tdiv((1 << 24) + (tot >> 1), tot)   # round(2^24/tot)
        out = [clip16(qround(int(e) * invN, Q12)) for e in es]
        return out

    @staticmethod
    def linear(act, w, b, din, dout):
        """act[din] (Q12), w[dout][din] (Q12). returns Q12."""
        out = []
        for o in range(dout):
            acc = 0
            for i in range(din):
                acc += int(w[o][i]) * int(act[i])
            out.append(clip16(qround(acc, Q12) + int(b[o])))
        return out

    @staticmethod
    def layernorm(x, gamma, beta):
        """x[4] Q12 -> y[4] Q12, gamma/beta Q12."""
        s = sum(int(v) for v in x)
        mu = qround(s, 4)
        ss = 0
        for v in x:
            d = int(v) - mu
            ss += d * d
        var = tdiv(ss, 4)
        inv = invsqrt(var)
        y = []
        for i in range(4):
            v = x[i]
            d = int(v) - mu
            yn = qround(d * inv, Q12)                 # (d*inv+2048)/4096
            y.append(clip16(qround(int(yn) * int(gamma[i]), Q12) + int(beta[i])))
        return y

    def forward(self, xin):
        """xin: 16 int16 (4x4) Q12 -> [2 logits] Q12"""
        E, LAY, L, seq, H, FF = 4, 522, 2, 4, 1, 16
        w = self.w
        x = [[int(xin[t * E + e]) for e in range(E)] for t in range(seq)]

        off = 0
        for layer in range(L):
            qkv_w = w[off:off + 48];   off += 48   # [4,12] row-major (e,col)
            qkv_b = w[off:off + 12];   off += 12
            out_w = w[off:off + 16];   off += 16   # [4,4]
            out_b = w[off:off + 4];    off += 4
            fc1_w = w[off:off + 64];   off += 64
            fc1_b = w[off:off + 16];   off += 16
            fc2_w = w[off:off + 64];   off += 64
            fc2_b = w[off:off + 4];    off += 4
            ln1_g = w[off:off + 4];    off += 4
            ln1_b = w[off:off + 4];    off += 4
            ln2_g = w[off:off + 4];    off += 4
            ln2_b = w[off:off + 4];    off += 4

            def wq(e, j): return qkv_w[e * 12 + 0 * 4 + j]
            def wk(e, j): return qkv_w[e * 12 + 1 * 4 + j]
            def wv(e, j): return qkv_w[e * 12 + 2 * 4 + j]

            # ---- attention ----
            Q = [[clip16(qround(sum(wq(e, j) * x[t][j] for j in range(E)), Q12) + int(qkv_b[e]))
                  for e in range(E)] for t in range(seq)]
            K = [[clip16(qround(sum(wk(e, j) * x[t][j] for j in range(E)), Q12) + int(qkv_b[E + e]))
                  for e in range(E)] for t in range(seq)]
            V = [[clip16(qround(sum(wv(e, j) * x[t][j] for j in range(E)), Q12) + int(qkv_b[2 * E + e]))
                  for e in range(E)] for t in range(seq)]

            # scores (scaled by 1/sqrt(d_head)=0.5, d_head=4)
            scores = [[tdiv(qround(sum(Q[t][e] * K[k][e] for e in range(E)), Q12), 2)
                       for k in range(seq)] for t in range(seq)]

            attn = []
            for t in range(seq):
                sm = FixedQuant.softmax(scores[t])
                row = [clip16(qround(sum(int(sm[k]) * V[k][e] for k in range(seq)), Q12))
                       for e in range(E)]
                attn.append(row)

            z = [[clip16(qround(sum(int(out_w[j * E + e]) * attn[t][j] for j in range(E)), Q12)
                         + int(out_b[e])) for e in range(E)] for t in range(seq)]
            r = [[clip16(int(x[t][e]) + int(z[t][e])) for e in range(E)] for t in range(seq)]
            y = [FixedQuant.layernorm(r[t], ln1_g, ln1_b) for t in range(seq)]

            # ---- ffn ----
            x2 = []
            for t in range(seq):
                h = [clip16(qround(sum(int(fc1_w[j * FF + f]) * y[t][j] for j in range(E)), Q12)
                            + int(fc1_b[f])) for f in range(FF)]
                g = [FixedQuant.gelu(hv) for hv in h]
                u = [clip16(qround(sum(int(fc2_w[f * E + e]) * g[f] for f in range(FF)), Q12)
                            + int(fc2_b[e])) for e in range(E)]
                r2 = [clip16(int(y[t][e]) + int(u[e])) for e in range(E)]
                x2.append(FixedQuant.layernorm(r2, ln2_g, ln2_b))
            x = x2

        # ---- output proj: 16 -> 2 ----
        op_w = w[off:off + 32]; off += 32   # [16,2] row-major (i,c)
        op_b = w[off:off + 2];  off += 2
        assert off == LAY, f"weight count {off} != {LAY}"
        flat = [x[t][e] for t in range(seq) for e in range(E)]
        logits = [clip16(qround(sum(int(op_w[i * 2 + c]) * flat[i] for i in range(16)), Q12)
                         + int(op_b[c])) for c in range(2)]
        return logits

# --------------------------------------------------------------------------
# weight export
# --------------------------------------------------------------------------
def quantize(tensor):
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    scale = 32767.0 / 8.0
    return np.clip(arr * scale, -32768, 32767).astype(np.int16)

def export_full_weights(model, path):
    sd = model.state_dict()
    cfg = MODEL_CONFIG
    lst = []
    for i in range(cfg["num_layers"]):
        qkv_w = quantize(sd[f"layers.{i}.attention.in_proj_weight"].T)
        qkv_b = quantize(sd[f"layers.{i}.attention.in_proj_bias"])
        out_w = quantize(sd[f"layers.{i}.attention.out_proj.weight"].T)
        out_b = quantize(sd[f"layers.{i}.attention.out_proj.bias"])
        fc1_w = quantize(sd[f"layers.{i}.ffn.0.weight"].T)
        fc1_b = quantize(sd[f"layers.{i}.ffn.0.bias"])
        fc2_w = quantize(sd[f"layers.{i}.ffn.2.weight"].T)
        fc2_b = quantize(sd[f"layers.{i}.ffn.2.bias"])
        ln1_w = quantize(sd[f"layers.{i}.norm1.weight"])
        ln1_b = quantize(sd[f"layers.{i}.norm1.bias"])
        ln2_w = quantize(sd[f"layers.{i}.norm2.weight"])
        ln2_b = quantize(sd[f"layers.{i}.norm2.bias"])
        lst += [qkv_w.ravel(), qkv_b.ravel(), out_w.ravel(), out_b.ravel(),
                fc1_w.ravel(), fc1_b.ravel(), fc2_w.ravel(), fc2_b.ravel(),
                ln1_w.ravel(), ln1_b.ravel(), ln2_w.ravel(), ln2_b.ravel()]
    out_w = quantize(sd["output_proj.weight"].T)
    out_b = quantize(sd["output_proj.bias"])
    lst += [out_w.ravel(), out_b.ravel()]

    allw = np.concatenate(lst).astype(np.int16)
    print(f"[+] total weights: {allw.size} int16 -> {allw.size * 2} bytes")
    assert allw.size == 522, allw.size
    allw.tofile(path)
    print(f"[+] wrote {path}")
    return allw

# --------------------------------------------------------------------------
def main():
    os.makedirs(REF_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    torch.manual_seed(20260910)
    model = TinyTransformer(MODEL_CONFIG)
    model.eval()

    # save model + export full weights
    torch.save({"model": model.state_dict()},
               os.path.join(DATA_DIR, "best_tinytransformer_fpga.pth"))
    w = export_full_weights(model, WEIGHT_BIN)

    # write LUT tables into xform_tables.h
    lines = ["// AUTO-GENERATED by gen_reference.py - do not edit manually",
             "#ifndef XFORM_TABLES_H", "#define XFORM_TABLES_H",
             "#include <stdint.h>",
             "static const int16_t SIG_LUT[128] = {",
             ",".join(str(v) for v in SIG_LUT), "};",
             "static const int16_t EXP_LUT[512] = {",
             ",".join(str(v) for v in EXP_LUT), "};",
             "#endif"]
    with open(os.path.join(PROJECT_HLS, "xform_tables.h"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("[+] wrote xform_tables.h")

    fx = FixedQuant(w.tolist())

    # test vectors
    n = 8
    vectors = []
    for s in range(n):
        xin = [np.random.randint(-23400, 23401) for _ in range(16)]
        xt = torch.tensor(np.array(xin, dtype=np.int16),
                          dtype=torch.float32) / Q12
        xt = xt.view(1, 4, 4).float()
        with torch.no_grad():
            logits_f = model(xt)[0].tolist()
        logits_q = fx.forward(xin)
        vectors.append({"in": xin, "float": logits_f, "q12": logits_q})

    with open(os.path.join(REF_DIR, "test_vectors.json"), "w") as f:
        json.dump(vectors, f)
    np.array(w, dtype=np.int16).tofile(os.path.join(REF_DIR, "weights_int16.bin"))

    # sanity: fixed vs float argmax agreement
    agree = [1 if (v["q12"][0] > v["q12"][1]) == (v["float"][0] > v["float"][1])
             else 0 for v in vectors]
    print(f"[+] test vectors: {n}, fixed-vs-float argmax agreement "
          f"{sum(agree)}/{n}")
    for s, v in enumerate(vectors[:3]):
        print(f"  case {s}: float={[round(x,3) for x in v['float']]} "
              f"q12={v['q12']}")

if __name__ == "__main__":
    main()