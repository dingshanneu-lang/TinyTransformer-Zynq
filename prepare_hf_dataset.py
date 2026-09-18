#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_hf_dataset.py - 从 HuggingFace 镜像下载猫狗数据集并整理为项目目录结构。

输出:
  data/processed/train/cats/*.jpg
  data/processed/train/dogs/*.jpg
  data/processed/val/cats/*.jpg
  data/processed/val/dogs/*.jpg

用法:
  python prepare_hf_dataset.py                # 小样本 (50 张, 快速验证)
  python prepare_hf_dataset.py --full         # 完整版 (microsoft/cats_vs_dogs)
"""
import os, sys, argparse, random
from pathlib import Path
import io
import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT = Path(__file__).parent
OUT = PROJECT / "data" / "processed"
SEED = 42


def save_images(images, labels, split_dirs):
    """images: list of PIL, labels: list of int(0=cat,1=dog)"""
    from PIL import Image
    n = len(images)
    idx = list(range(n))
    random.seed(SEED); random.shuffle(idx)
    nval = max(1, int(n * 0.2))
    val_idx = set(idx[:nval])

    counts = {"train": [0, 0], "val": [0, 0]}
    for i, (im, lb) in enumerate(zip(images, labels)):
        split = "val" if i in val_idx else "train"
        cls = "cats" if lb == 0 else "dogs"
        dst = split_dirs[split] / cls
        im.save(dst / f"{split}_{cls}_{i:05d}.jpg", "JPEG")
        counts[split][lb] += 1

    print(f"[+] train: cats={counts['train'][0]} dogs={counts['train'][1]}")
    print(f"[+] val:   cats={counts['val'][0]} dogs={counts['val'][1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="使用完整数据集")
    args = ap.parse_args()

    for split in ["train", "val"]:
        for cls in ["cats", "dogs"]:
            (OUT / split / cls).mkdir(parents=True, exist_ok=True)
    split_dirs = {"train": OUT / "train", "val": OUT / "val"}

    from huggingface_hub import hf_hub_download
    import pandas as pd
    from PIL import Image

    repo = "microsoft/cats_vs_dogs" if args.full else "hf-internal-testing/cats_vs_dogs_sample"
    # 完整版是多分片 parquet，小样本是单文件
    if args.full:
        from huggingface_hub import list_repo_files
        files = [f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
    else:
        files = ["data/train-00000-of-00001.parquet"]

    images, labels = [], []
    for fn in files:
        print(f"[*] download {fn}")
        p = hf_hub_download(repo, fn, repo_type="dataset")
        df = pd.read_parquet(p)
        print(f"    rows={len(df)} cols={list(df.columns)}")
        for _, row in df.iterrows():
            img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
            im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            images.append(im)
            labels.append(int(row["labels"]))
        if not args.full:
            break

    print(f"[+] loaded {len(images)} images")
    save_images(images, labels, split_dirs)
    print("[+] done")


if __name__ == "__main__":
    main()
