#!/usr/bin/env python3
"""
Kaggle Dogs vs Cats 数据集下载与准备脚本

使用方法:
1. 安装 kaggle CLI: pip install kaggle
2. 配置 kaggle.json (从 https://www.kaggle.com/settings/account 下载)
   - Windows: %USERPROFILE%\\.kaggle\\kaggle.json
   - Linux/Mac: ~/.kaggle/kaggle.json
   - chmod 600 ~/.kaggle/kaggle.json
3. 运行: python download_dataset.py

或者手动下载:
- 访问 https://www.kaggle.com/c/dogs-vs-cats/data
- 下载 train.zip 和 test1.zip
- 解压到 data/raw/
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
import subprocess
import argparse

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

def check_kaggle_cli():
    """检查 kaggle CLI 是否可用"""
    try:
        result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_kaggle_credentials():
    """检查 kaggle 凭据"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    return kaggle_json.exists()

def download_with_kaggle():
    """使用 kaggle CLI 下载数据集"""
    print("[*] 使用 kaggle CLI 下载 Dogs vs Cats 数据集...")
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # 下载训练集
    print("[*] 下载训练集...")
    result = subprocess.run([
        "kaggle", "competitions", "download", "-c", "dogs-vs-cats",
        "-p", str(RAW_DIR)
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[-] 下载失败: {result.stderr}")
        return False
    
    print("[+] 下载完成")
    return True

def extract_dataset():
    """解压数据集"""
    print("[*] 解压数据集...")
    
    train_zip = RAW_DIR / "train.zip"
    test_zip = RAW_DIR / "test1.zip"
    
    if train_zip.exists():
        print("[*] 解压 train.zip...")
        with zipfile.ZipFile(train_zip, 'r') as zf:
            zf.extractall(RAW_DIR / "train")
        print("[+] train.zip 解压完成")
    
    if test_zip.exists():
        print("[*] 解压 test1.zip...")
        with zipfile.ZipFile(test_zip, 'r') as zf:
            zf.extractall(RAW_DIR / "test")
        print("[+] test1.zip 解压完成")

def organize_dataset(val_split=0.2, seed=42):
    """整理数据集为 train/val/test 结构"""
    import random
    
    print("[*] 整理数据集结构...")
    
    train_dir = RAW_DIR / "train"
    if not train_dir.exists():
        print(f"[-] 训练目录不存在: {train_dir}")
        return False
    
    # 获取所有图片
    cat_images = list(train_dir.glob("cat*.jpg"))
    dog_images = list(train_dir.glob("dog*.jpg"))
    
    print(f"    猫图片: {len(cat_images)}")
    print(f"    狗图片: {len(dog_images)}")
    
    # 打乱并划分
    random.seed(seed)
    random.shuffle(cat_images)
    random.shuffle(dog_images)
    
    cat_val_count = int(len(cat_images) * val_split)
    dog_val_count = int(len(dog_images) * val_split)
    
    cat_val = cat_images[:cat_val_count]
    cat_train = cat_images[cat_val_count:]
    dog_val = dog_images[:dog_val_count]
    dog_train = dog_images[dog_val_count:]
    
    # 创建目标目录
    for split in ["train", "val"]:
        for cls in ["cats", "dogs"]:
            (PROCESSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    
    # 复制文件
    def copy_files(files, dst_dir):
        for f in files:
            shutil.copy2(f, dst_dir / f.name)
    
    print("[*] 复制训练集...")
    copy_files(cat_train, PROCESSED_DIR / "train" / "cats")
    copy_files(dog_train, PROCESSED_DIR / "train" / "dogs")
    
    print("[*] 复制验证集...")
    copy_files(cat_val, PROCESSED_DIR / "val" / "cats")
    copy_files(dog_val, PROCESSED_DIR / "val" / "dogs")
    
    # 处理测试集 (无标签)
    test_dir = RAW_DIR / "test"
    if test_dir.exists():
        test_images = list(test_dir.glob("*.jpg"))
        (PROCESSED_DIR / "test").mkdir(parents=True, exist_ok=True)
        print(f"[*] 复制测试集 ({len(test_images)} 张)...")
        for f in test_images:
            shutil.copy2(f, PROCESSED_DIR / "test" / f.name)
    
    print(f"[+] 数据集整理完成")
    print(f"    训练集: 猫 {len(cat_train)}, 狗 {len(dog_train)}")
    print(f"    验证集: 猫 {len(cat_val)}, 狗 {len(dog_val)}")
    if test_dir.exists():
        print(f"    测试集: {len(test_images)}")
    
    return True

def create_sample_dataset(num_samples=100):
    """创建小规模样本数据集用于快速测试"""
    print(f"[*] 创建样本数据集 ({num_samples} 每类)...")
    
    train_dir = RAW_DIR / "train"
    if not train_dir.exists():
        print(f"[-] 训练目录不存在: {train_dir}")
        return False
    
    cat_images = list(train_dir.glob("cat*.jpg"))[:num_samples]
    dog_images = list(train_dir.glob("dog*.jpg"))[:num_samples]
    
    sample_dir = DATA_DIR / "sample"
    for split in ["train", "val"]:
        for cls in ["cats", "dogs"]:
            (sample_dir / split / cls).mkdir(parents=True, exist_ok=True)
    
    # 80% train, 20% val
    cat_split = int(len(cat_images) * 0.8)
    dog_split = int(len(dog_images) * 0.8)
    
    import shutil
    for i, img in enumerate(cat_images):
        dst = sample_dir / ("train" if i < cat_split else "val") / "cats" / img.name
        shutil.copy2(img, dst)
    
    for i, img in enumerate(dog_images):
        dst = sample_dir / ("train" if i < dog_split else "val") / "dogs" / img.name
        shutil.copy2(img, dst)
    
    print(f"[+] 样本数据集创建完成: {sample_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="下载并准备 Dogs vs Cats 数据集")
    parser.add_argument("--manual", action="store_true", help="手动模式 (仅整理已下载的数据)")
    parser.add_argument("--sample", action="store_true", help="创建小样本数据集用于测试")
    parser.add_argument("--val-split", type=float, default=0.2, help="验证集比例 (默认 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Kaggle Dogs vs Cats 数据集准备工具")
    print("=" * 50)
    
    if not args.manual:
        # 检查 kaggle CLI
        if not check_kaggle_cli():
            print("[-] 未检测到 kaggle CLI")
            print("    安装方法: pip install kaggle")
            print("    或手动下载: https://www.kaggle.com/c/dogs-vs-cats/data")
            print("    下载后将 train.zip 和 test1.zip 放入 data/raw/")
            if not RAW_DIR.exists():
                return 1
        else:
            print("[+] kaggle CLI 可用")
            
            if not check_kaggle_credentials():
                print("[-] 未配置 kaggle 凭据")
                print("    请从 https://www.kaggle.com/settings/account 下载 kaggle.json")
                print(f"    放置到: {Path.home() / '.kaggle' / 'kaggle.json'}")
                print("    Windows: %USERPROFILE%\\.kaggle\\kaggle.json")
                return 1
            
            print("[+] kaggle 凭据已配置")
            
            if not download_with_kaggle():
                return 1
    
    # 解压
    extract_dataset()
    
    # 整理
    if not organize_dataset(args.val_split, args.seed):
        return 1
    
    # 创建样本集
    if args.sample:
        create_sample_dataset()
    
    print("\n" + "=" * 50)
    print("数据集准备完成!")
    print(f"处理后数据位置: {PROCESSED_DIR}")
    print("结构:")
    print(f"  {PROCESSED_DIR}/")
    print(f"  ├── train/")
    print(f"  │   ├── cats/")
    print(f"  │   └── dogs/")
    print(f"  ├── val/")
    print(f"  │   ├── cats/")
    print(f"  │   └── dogs/")
    print(f"  └── test/")
    print("=" * 50)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())