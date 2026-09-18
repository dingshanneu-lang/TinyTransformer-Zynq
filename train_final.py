#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_final.py - 正式训练 TinyTransformer 猫狗分类

要点:
  - 从 data/processed/train 内部随机划分 holdout 作为验证集
    (官方 data/processed/val 分布异常, 不用于训练/选型)
  - Stage1: 训练 MobileNetV2(冻结) -> 1280->16 proj
  - Stage2: 训练 TinyTransformer (2层, 522参数) + 特征缩放
  - 导出 522 int16 Q12 权重 (与板上 xform_core_opt_prj IP 一致) + proj/scale

产物:
  weights/xform_weights.bin   (522 int16)
  weights/best_final.pth      (TinyTransformer + proj + scale)
"""
import sys, argparse, random, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

ROOT = Path(r"D:\ZynqTinyTransformerClassification")
sys.path.insert(0, str(ROOT))
from train import TinyTransformer, MODEL_CONFIG
from train_v2 import Backbone, tf_train, tf_val, evaluate, export522

DATA = ROOT / "data" / "processed"
W_DIR = ROOT / "weights"
W_DIR.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class SplitDS(Dataset):
    def __init__(self, items, tf):
        self.items = items; self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, l = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj-epochs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--proj-lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--holdout", type=int, default=2000)
    ap.add_argument("--out", default="xform_weights.bin")
    args = ap.parse_args()

    items = []
    for cls, lb in [("cats", 0), ("dogs", 1)]:
        for p in sorted((DATA / "train" / cls).glob("*.jpg")):
            items.append((p, lb))
    random.Random(99).shuffle(items)
    val_items, tr_items = items[:args.holdout], items[args.holdout:]
    print(f"[*] device={DEV} train={len(tr_items)} holdout={len(val_items)}")

    tr = DataLoader(SplitDS(tr_items, tf_train), batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
    hv = DataLoader(SplitDS(val_items, tf_val), batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    bb = Backbone().to(DEV); tt = TinyTransformer(MODEL_CONFIG).to(DEV)
    crit = nn.CrossEntropyLoss()

    # ---- Stage1: proj ----
    print("[*] Stage1: train 1280->16 proj")
    head = nn.Linear(16, 2).to(DEV)
    opt = optim.AdamW(list(bb.proj.parameters()) + list(head.parameters()),
                      lr=args.proj_lr, weight_decay=1e-4)
    for ep in range(args.proj_epochs):
        bb.train()
        for x, y in tr:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad(); loss = crit(head(bb.proj(bb.raw(x))), y); loss.backward(); opt.step()
        ok = n = 0
        with torch.no_grad():
            for x, y in hv:
                x, y = x.to(DEV), y.to(DEV)
                ok += (head(bb.proj(bb.raw(x))).argmax(1) == y).sum().item(); n += y.size(0)
        print(f"  proj ep{ep}: holdout={100*ok/n:.2f}%")

    # ---- Stage2: TinyTransformer ----
    print("[*] Stage2: train TinyTransformer")
    for p in bb.parameters():
        p.requires_grad = False
    scale = nn.Parameter(torch.tensor(4.0, device=DEV))
    opt = optim.AdamW(list(tt.parameters()) + [scale], lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = 0.0
    for ep in range(args.epochs):
        tt.train(); ok = n = 0; ls = 0
        for x, y in tr:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad(); out = tt(bb(x) * scale); loss = crit(out, y); loss.backward(); opt.step()
            ls += loss.item() * y.size(0)
            ok += (out.argmax(1) == y).sum().item(); n += y.size(0)
        sched.step()
        va = evaluate(bb, tt, hv, DEV, scale)
        print(f"Epoch {ep:2d}: loss={ls/n:.4f} train={100*ok/n:.2f}% holdout={va:.2f}% scale={scale.item():.3f}")
        if va > best:
            best = va
            torch.save({"model": tt.state_dict(), "proj": bb.proj.state_dict(),
                        "scale": float(scale.item()), "config": MODEL_CONFIG,
                        "holdout_acc": va}, W_DIR / "best_final.pth")

    ck = torch.load(W_DIR / "best_final.pth", map_location=DEV)
    tt.load_state_dict(ck["model"])
    w = export522(tt, W_DIR / args.out)
    print(f"[+] exported {w.size} int16 -> {W_DIR/args.out}")
    print(f"[+] best holdout = {best:.2f}%  scale={ck['scale']:.3f}")
    json.dump({"best_holdout": best, "scale": ck["scale"], "holdout": args.holdout},
              open(W_DIR / "final_meta.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
