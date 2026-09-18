#!/usr/bin/env python3
"""
猫狗分类训练脚本
支持 CNN 和 TinyTransformer 模型，导出 FPGA 兼容的权重文件
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
from tqdm import tqdm

# 导入 PyTorch 嵌入模型
try:
    from torch_embed_models import create_torch_embed_model, EmbedModelWrapper
    TORCH_EMBED_AVAILABLE = True
except ImportError:
    TORCH_EMBED_AVAILABLE = False
    create_torch_embed_model = None
    EmbedModelWrapper = None

# 项目路径
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

# 模型配置 (与 FPGA 端 network.h 保持一致)
# FPGA 端接收 PC 端嵌入模型提取的 16 维特征 (4x4)，不包含输入投影
MODEL_CONFIG = {
    "num_layers": 2,
    "seq_len": 4,
    "embed_dim": 4,
    "ffn_dim": 16,
    "num_heads": 1,
    "num_classes": 2,
}

class TinyTransformer(nn.Module):
    """TinyTransformer 模型 (与 FPGA 端一致)
    
    FPGA 端架构:
    - 输入: (seq_len, embed_dim) = (4, 4) int16 特征 (已由 PC 端嵌入模型提取)
    - Transformer 层 (num_layers=2)
    - 输出投影: (seq_len * embed_dim) -> num_classes
    
    注意: 不包含 input_proj (3072->16)，那是 PC 端嵌入模型的工作
    """
    
    def __init__(self, config=MODEL_CONFIG):
        super().__init__()
        self.config = config
        self.num_layers = config["num_layers"]
        self.seq_len = config["seq_len"]
        self.embed_dim = config["embed_dim"]
        self.ffn_dim = config["ffn_dim"]
        self.num_heads = config["num_heads"]
        self.num_classes = config["num_classes"]
        
        # Transformer 层 (直接处理 4x4 特征)
        self.layers = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.ffn_dim, self.num_heads)
            for _ in range(self.num_layers)
        ])
        
        # 输出头: (seq_len * embed_dim) -> num_classes
        self.output_proj = nn.Linear(self.seq_len * self.embed_dim, self.num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (B, seq_len, embed_dim) = (B, 4, 4) - 已经是提取好的特征
        B = x.shape[0]
        
        # Transformer 层
        for layer in self.layers:
            x = layer(x)
        
        # 全局池化 + 分类
        x = x.view(B, -1)  # (B, seq_len * embed_dim) = (B, 16)
        x = self.output_proj(x)  # (B, num_classes)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ffn_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class SimpleCNN(nn.Module):
    """简单 CNN 基线模型"""
    
    def __init__(self, num_classes=2, input_size=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 32x32 -> 32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MobileNetV2Classifier(nn.Module):
    """MobileNetV2 迁移学习模型"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        # 替换分类头
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


def get_data_loaders(data_dir, batch_size=32, num_workers=4, image_size=32):
    """获取数据加载器"""
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    if not train_dir.exists():
        raise FileNotFoundError(f"训练目录不存在: {train_dir}")
    
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform) if val_dir.exists() else None
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    
    print(f"类别: {train_dataset.classes}")
    print(f"训练样本: {len(train_dataset)}")
    if val_dataset:
        print(f"验证样本: {len(val_dataset)}")
    
    return train_loader, val_loader, train_dataset.classes


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, model_type="cnn", embed_model=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # TinyTransformer 需要先提取特征
        if model_type == "tinytransformer" and embed_model is not None:
            with torch.no_grad():
                # images: (B, 3, 32, 32) -> embed_model -> (B, 4, 4)
                features = embed_model.extract_batch(images)
            outputs = model(features)
        else:
            outputs = model(images)
        
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({"loss": f"{running_loss/total:.4f}", "acc": f"{100.*correct/total:.2f}%"})
    
    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device, epoch, model_type="cnn", embed_model=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            if model_type == "tinytransformer" and embed_model is not None:
                features = embed_model.extract_batch(images)
                outputs = model(features)
            else:
                outputs = model(images)
            
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({"loss": f"{running_loss/total:.4f}", "acc": f"{100.*correct/total:.2f}%"})
    
    return running_loss / len(loader), 100. * correct / total


def export_weights_tinytransformer(model, output_path, config=MODEL_CONFIG):
    """导出 TinyTransformer 权重为 FPGA 兼容的二进制格式
    
    FPGA 端期望的权重布局 (network.c: transformer_weights_t):
    - 每层: attention (qkv_proj, out_proj), ffn (fc1, fc2), norms
    - output_proj_weight: [num_classes, seq_len*embed_dim] -> [2, 16]
    - output_proj_bias: [num_classes] -> [2]
    
    总计约 440 个 int16 = 880 bytes
    (不包含 input_proj，那是 PC 端嵌入模型的工作)
    """
    print(f"[*] 导出 TinyTransformer 权重到 {output_path}")
    
    state_dict = model.state_dict()
    weights_list = []
    
    # 量化参数
    scale = 32767.0 / 8.0  # 假设权重范围 [-8, 8]
    
    def quantize(tensor):
        """将 float32 tensor 量化为 int16"""
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        arr = np.clip(arr * scale, -32768, 32767).astype(np.int16)
        return arr
    
    # 1. Transformer layers (无 input_proj)
    for i in range(config["num_layers"]):
        # Attention: qkv_proj (in_proj_weight), out_proj
        # PyTorch MultiheadAttention: in_proj_weight [3*embed_dim, embed_dim]
        qkv_w = quantize(state_dict[f"layers.{i}.attention.in_proj_weight"].T)
        qkv_b = quantize(state_dict[f"layers.{i}.attention.in_proj_bias"])
        out_w = quantize(state_dict[f"layers.{i}.attention.out_proj.weight"].T)
        out_b = quantize(state_dict[f"layers.{i}.attention.out_proj.bias"])
        weights_list.append(qkv_w.ravel())
        weights_list.append(qkv_b.ravel())
        weights_list.append(out_w.ravel())
        weights_list.append(out_b.ravel())
        
        # FFN
        fc1_w = quantize(state_dict[f"layers.{i}.ffn.0.weight"].T)
        fc1_b = quantize(state_dict[f"layers.{i}.ffn.0.bias"])
        fc2_w = quantize(state_dict[f"layers.{i}.ffn.2.weight"].T)
        fc2_b = quantize(state_dict[f"layers.{i}.ffn.2.bias"])
        weights_list.append(fc1_w.ravel())
        weights_list.append(fc1_b.ravel())
        weights_list.append(fc2_w.ravel())
        weights_list.append(fc2_b.ravel())
        
        # LayerNorms
        ln1_w = quantize(state_dict[f"layers.{i}.norm1.weight"])
        ln1_b = quantize(state_dict[f"layers.{i}.norm1.bias"])
        ln2_w = quantize(state_dict[f"layers.{i}.norm2.weight"])
        ln2_b = quantize(state_dict[f"layers.{i}.norm2.bias"])
        weights_list.append(ln1_w.ravel())
        weights_list.append(ln1_b.ravel())
        weights_list.append(ln2_w.ravel())
        weights_list.append(ln2_b.ravel())
    
    # 2. output_proj
    out_w = quantize(state_dict["output_proj.weight"].T)
    out_b = quantize(state_dict["output_proj.bias"])
    weights_list.append(out_w.ravel())
    weights_list.append(out_b.ravel())
    
    # 合并所有权重
    all_weights = np.concatenate(weights_list)
    print(f"    总权重数量: {len(all_weights)} (int16)")
    print(f"    文件大小: {len(all_weights) * 2} bytes")
    
    # 验证大小 (应为 440 个 int16 = 880 bytes)
    expected = 440
    if len(all_weights) != expected:
        print(f"[!] 警告: 权重数量 {len(all_weights)} != 期望 {expected}")
        # 截断或填充
        if len(all_weights) > expected:
            all_weights = all_weights[:expected]
        else:
            all_weights = np.pad(all_weights, (0, expected - len(all_weights)), mode='constant')
    
    # 写入二进制文件
    all_weights.astype(np.int16).tofile(output_path)
    print(f"[+] 权重已导出: {output_path}")


def export_weights_cnn(model, output_path):
    """导出 CNN 权重 (仅用于参考，FPGA 端需对应修改)"""
    print(f"[*] 导出 CNN 权重到 {output_path}")
    state_dict = model.state_dict()
    all_params = []
    for k, v in state_dict.items():
        arr = v.detach().cpu().numpy().astype(np.float32).ravel()
        all_params.append(arr)
    all_params = np.concatenate(all_params)
    scale = 32767.0 / 8.0
    quantized = np.clip(all_params * scale, -32768, 32767).astype(np.int16)
    quantized.tofile(output_path)
    print(f"[+] CNN 权重已导出: {len(quantized)} int16")


def main():
    parser = argparse.ArgumentParser(description="猫狗分类训练脚本")
    parser.add_argument("--model", choices=["tinytransformer", "cnn", "mobilenetv2"], 
                       default="tinytransformer", help="模型类型")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="数据集目录")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--output", type=str, default="weights.bin", help="输出权重文件名")
    parser.add_argument("--image-size", type=int, default=32, help="输入图片尺寸")
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--resume", type=str, help="恢复训练的检查点路径")
    parser.add_argument("--pretrained", action="store_true", help="使用预训练权重 (仅 MobileNetV2)")
    parser.add_argument("--embed-model", choices=["random", "mobilenetv2", "vit"], 
                       default="random", help="嵌入模型类型 (仅 TinyTransformer)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"猫狗分类训练 - {args.model.upper()}")
    print("=" * 60)
    print(f"设备: {args.device}")
    print(f"数据目录: {args.data_dir}")
    print(f"训练轮数: {args.epochs}")
    print(f"批大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    if args.model == "tinytransformer":
        print(f"嵌入模型: {args.embed_model}")
    print("=" * 60)
    
    # 数据加载
    train_loader, val_loader, classes = get_data_loaders(
        Path(args.data_dir), args.batch_size, args.num_workers, args.image_size
    )
    
    # 创建嵌入模型 (仅 TinyTransformer 需要)
    embed_model = None
    if args.model == "tinytransformer":
        if not TORCH_EMBED_AVAILABLE:
            print("[-] torch_embed_models 不可用，使用随机投影")
            from torch_embed_models import TorchRandomProjection
            embed_model = TorchRandomProjection().to(args.device)
        else:
            print(f"[*] 创建嵌入模型: {args.embed_model}")
            embed_model = create_torch_embed_model(args.embed_model, device=args.device).to(args.device)
            embed_model.eval()  # 冻结嵌入模型
        print(f"嵌入模型参数量: {sum(p.numel() for p in embed_model.parameters()):,}")
    
    # 模型创建
    if args.model == "tinytransformer":
        model = TinyTransformer(MODEL_CONFIG).to(args.device)
    elif args.model == "cnn":
        model = SimpleCNN(num_classes=len(classes), input_size=args.image_size).to(args.device)
    elif args.model == "mobilenetv2":
        model = MobileNetV2Classifier(num_classes=len(classes), pretrained=args.pretrained).to(args.device)
    
    print(f"\n主模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 恢复训练
    start_epoch = 0
    best_acc = 0.0
    if args.resume and Path(args.resume).exists():
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        print(f"[+] 恢复训练从 Epoch {start_epoch}, 最佳准确率: {best_acc:.2f}%")
    
    # 训练循环
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch, args.model, embed_model
        )
        
        # 验证
        val_loss, val_acc = 0, 0
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, args.device, epoch, args.model, embed_model)
        
        scheduler.step()
        
        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}% | LR={scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
                "config": MODEL_CONFIG,
                "classes": classes,
            }
            torch.save(checkpoint, WEIGHTS_DIR / f"best_{args.model}.pth")
            print(f"  [*] 保存最佳模型 (Acc: {best_acc:.2f}%)")
    
    # 导出 FPGA 权重
    output_path = WEIGHTS_DIR / args.output
    if args.model == "tinytransformer":
        export_weights_tinytransformer(model, output_path, MODEL_CONFIG)
    else:
        export_weights_cnn(model, output_path)
    
    # 保存训练历史
    with open(WEIGHTS_DIR / f"history_{args.model}.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"训练完成! 最佳验证准确率: {best_acc:.2f}%")
    print(f"权重文件: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()