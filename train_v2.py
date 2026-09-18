#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_v2.py - 稳健训练 TinyTransformer 猫狗分类

Stage 1: 冻结 MobileNetV2 backbone, 训练 1280->16 proj (配一个临时线性头), 让 16-d 特征可分
Stage 2: 冻结 proj, 端到端训练 TinyTransformer (2 层, 522 参数) + 特征缩放
导出: 522 int16 Q12 权重 + proj 权重 (供 PC 端提特征)
"""
import os, sys, argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

ROOT = Path(r"D:\ZynqTinyTransformerClassification")
sys.path.insert(0, str(ROOT))
from train import TinyTransformer, MODEL_CONFIG

DATA_DIR = ROOT / "data" / "processed"
W_DIR = ROOT / "weights"
W_DIR.mkdir(exist_ok=True)
SCALE = 32767.0 / 8.0

tf_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
tf_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ImgDS(Dataset):
    def __init__(self, split, tf, subset=None, seed=0):
        self.samples = []
        for cls, lb in [("cats", 0), ("dogs", 1)]:
            for p in sorted((DATA_DIR / split / cls).glob("*.jpg")):
                self.samples.append((p, lb))
        if subset:
            import random
            random.Random(seed).shuffle(self.samples)
            self.samples = self.samples[:subset]
        self.tf = tf

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        p, l = self.samples[i]
        return self.tf(Image.open(p).convert("RGB")), l


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = m.features
        for p in self.features.parameters():
            p.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(1280, 16)

    def raw(self, x):
        with torch.no_grad():
            f = self.features(x)
        return self.pool(f).flatten(1)

    def forward(self, x):
        return self.proj(self.raw(x)).view(-1, 4, 4)


def quantize(t):
    a = t.detach().cpu().numpy().astype(np.float32)
    return np.clip(a * SCALE, -32768, 32767).astype(np.int16)


def export522(model, path):
    sd = model.state_dict(); cfg = MODEL_CONFIG; lst = []
    for i in range(cfg["num_layers"]):
        lst += [quantize(sd[f"layers.{i}.attention.in_proj_weight"].T).ravel(),
                quantize(sd[f"layers.{i}.attention.in_proj_bias"]).ravel(),
                quantize(sd[f"layers.{i}.attention.out_proj.weight"].T).ravel(),
                quantize(sd[f"layers.{i}.attention.out_proj.bias"]).ravel(),
                quantize(sd[f"layers.{i}.ffn.0.weight"].T).ravel(),
                quantize(sd[f"layers.{i}.ffn.0.bias"]).ravel(),
                quantize(sd[f"layers.{i}.ffn.2.weight"].T).ravel(),
                quantize(sd[f"layers.{i}.ffn.2.bias"]).ravel(),
                quantize(sd[f"layers.{i}.norm1.weight"]).ravel(),
                quantize(sd[f"layers.{i}.norm1.bias"]).ravel(),
                quantize(sd[f"layers.{i}.norm2.weight"]).ravel(),
                quantize(sd[f"layers.{i}.norm2.bias"]).ravel()]
    lst += [quantize(sd["output_proj.weight"].T).ravel(),
            quantize(sd["output_proj.bias"]).ravel()]
    w = np.concatenate(lst).astype(np.int16)
    assert w.size == 522
    w.tofile(path)
    return w


@torch.no_grad()
def evaluate(bb, tt, loader, dev, scale):
    tt.eval(); ok = n = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        f = bb(x) * scale
        ok += (tt(f).argmax(1) == y).sum().item(); n += y.size(0)
    return 100.0 * ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--proj-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--proj-lr", type=float, default=1e-3)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="xform_weights.bin")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = DataLoader(ImgDS("train", tf_train, args.subset or None, 1), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, pin_memory=True)
    va = DataLoader(ImgDS("val", tf_val, (args.subset // 4) or None, 2), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f"[*] device={dev} train={len(tr.dataset)} val={len(va.dataset)}")

    bb = Backbone().to(dev)
    tt = TinyTransformer(MODEL_CONFIG).to(dev)
    crit = nn.CrossEntropyLoss()

    # ---- Stage 1: proj + 临时头 ----
    print("[*] Stage1: train 1280->16 proj")
    head = nn.Linear(16, 2).to(dev)
    opt = optim.AdamW(list(bb.proj.parameters()) + list(head.parameters()),
                      lr=args.proj_lr, weight_decay=1e-4)
    for ep in range(args.proj_epochs):
        bb.train(); head.train(); ok = n = 0; ls = 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            z = bb.proj(bb.raw(x))
            out = head(z)
            loss = crit(out, y)
            loss.backward(); opt.step()
            ls += loss.item() * y.size(0)
            ok += (out.argmax(1) == y).sum().item(); n += y.size(0)
        # val
        vok = vn = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(dev), y.to(dev)
                vok += (head(bb.proj(bb.raw(x))).argmax(1) == y).sum().item(); vn += y.size(0)
        print(f"  proj ep{ep}: loss={ls/n:.4f} train={100*ok/n:.2f}% val={100*vok/vn:.2f}%")

    # ---- Stage 2: TinyTransformer on frozen proj ----
    print("[*] Stage2: train TinyTransformer on frozen 16-d features")
    for p in bb.parameters():
        p.requires_grad = False
    scale = torch.nn.Parameter(torch.tensor(4.0, device=dev))
    opt = optim.AdamW(list(tt.parameters()) + [scale], lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = 0
    for ep in range(args.epochs):
        tt.train(); ok = n = 0; ls = 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            f = bb(x) * scale
            out = tt(f)
            loss = crit(out, y)
            loss.backward(); opt.step()
            ls += loss.item() * y.size(0)
            ok += (out.argmax(1) == y).sum().item(); n += y.size(0)
        sched.step()
        va_acc = evaluate(bb, tt, va, dev, scale)
        print(f"Epoch {ep:2d}: loss={ls/n:.4f} train={100*ok/n:.2f}% val={va_acc:.2f}% scale={scale.item():.3f}")
        if va_acc > best:
            best = va_acc
            torch.save({"model": tt.state_dict(), "proj": bb.proj.state_dict(),
                        "scale": float(scale.item()), "config": MODEL_CONFIG},
                       W_DIR / "best_v2.pth")

    ck = torch.load(W_DIR / "best_v2.pth", map_location=dev)
    tt.load_state_dict(ck["model"])
    w = export522(tt, W_DIR / args.out)
    print(f"[+] exported {w.size} int16 -> {W_DIR/args.out}")
    print(f"[+] best val={best:.2f}% scale={ck['scale']:.3f}")
    json.dump({"best_val": best, "scale": ck["scale"]}, open(W_DIR / "v2_meta.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
