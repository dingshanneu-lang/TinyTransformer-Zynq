#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_real_infer.py - 用真实猫狗图片测试端到端推理 (PC特征 -> FPGA -> 分类)。
用法: python test_real_infer.py [host]
"""
import sys, os, glob, random
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tiny_transformer_client import TinyTransformerClient, preprocess_image, classify_output

HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.0.102"
PORT = 5000
WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "xform_weights.bin")


def main():
    # 收集 val 图片
    val = []
    for cls, lb in [("cats", "猫"), ("dogs", "狗")]:
        for p in glob.glob(os.path.join(os.path.dirname(__file__), "data", "processed", "val", cls, "*.*")):
            val.append((p, lb))
    print(f"[+] {len(val)} val images")

    client = TinyTransformerClient(HOST, PORT, timeout=30)
    if not client.connect():
        print("[-] connect failed"); return 1
    if not client.hello():
        print("[-] hello failed"); return 1
    if not client.send_config():
        print("[-] config failed"); return 1
    if not client.send_weights(WEIGHTS):
        print("[-] weights failed"); return 1

    ok = 0
    for p, lb in val:
        x = preprocess_image(p)
        out = client.inference(x)
        if out is None:
            print(f"  {os.path.basename(p)}: inference failed"); continue
        label, cp, dp, conf = classify_output(out)
        good = (label == lb)
        ok += good
        print(f"  {os.path.basename(p):32s} true={lb} pred={label} "
              f"cat={cp:.2f} dog={dp:.2f} {'OK' if good else 'WRONG'}  logits=({out[0,0]},{out[0,1]})")

    print(f"\n真实图片准确率: {ok}/{len(val)}")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
