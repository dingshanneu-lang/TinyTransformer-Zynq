#!/usr/bin/env python3
"""
PyTorch 兼容的嵌入模型封装
用于训练时的特征提取 (可微分)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class TorchRandomProjection(nn.Module):
    """随机投影嵌入 (可微分版本，用于训练)"""
    
    def __init__(self, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        # 3 通道均值 -> 16 维特征
        self.proj = nn.Parameter(torch.randn(3, 16) * 0.1, requires_grad=False)
    
    def forward(self, x):
        # x: (B, 3, H, W) -> 全局平均池化 -> (B, 3) -> 投影 -> (B, 16)
        pooled = x.mean(dim=[2, 3])  # (B, 3)
        features = pooled @ self.proj  # (B, 16)
        return features.view(-1, 4, 4)  # (B, 4, 4)


class TorchMobileNetV2Embed(nn.Module):
    """MobileNetV2 特征提取器 (PyTorch 版本，可微分)"""
    
    def __init__(self, device='cpu', freeze_backbone=True):
        super().__init__()
        self.device = device
        
        # 加载预训练 MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.backbone = mobilenet.features  # 去掉分类头
        self.backbone.eval()
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 全局平均池化 + 投影到 16 维
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(1280, 16)  # MobileNetV2 输出 1280 维
        
        # ImageNet 归一化 (输入已归一化，这里用于反归一化后重新处理)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x):
        # x: (B, 3, H, W) - 已经是 ImageNet 归一化的 tensor
        # 反归一化到 [0, 1] 范围
        x = x * self.std + self.mean
        x = torch.clamp(x, 0, 1)
        
        # 通过 MobileNetV2 backbone
        features = self.backbone(x)  # (B, 1280, H', W')
        features = self.pool(features).flatten(1)  # (B, 1280)
        features = self.proj(features)  # (B, 16)
        return features.view(-1, 4, 4)  # (B, 4, 4)


class TorchViTEmbed(nn.Module):
    """Vision Transformer 特征提取器 (需要 timm)"""
    
    def __init__(self, model_name='vit_tiny_patch16_224', device='cpu', freeze_backbone=True):
        super().__init__()
        self.device = device
        
        try:
            import timm
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
            embed_dim = self.backbone.embed_dim
            
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            
            self.proj = nn.Linear(embed_dim, 16)
            
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        except ImportError:
            print("[-] timm 未安装，回退到随机投影")
            self.backbone = None
            self.proj = TorchRandomProjection()
    
    def forward(self, x):
        if self.backbone is None:
            return self.proj(x)
        
        # x: (B, 3, H, W) - 已经是 ImageNet 归一化的 tensor
        x = x * self.std + self.mean
        x = torch.clamp(x, 0, 1)
        
        # ViT 需要 224x224 输入
        if x.shape[-1] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        features = self.backbone(x)  # (B, embed_dim)
        features = self.proj(features)  # (B, 16)
        return features.view(-1, 4, 4)  # (B, 4, 4)


def create_torch_embed_model(model_type: str = 'random', device='cpu', **kwargs) -> nn.Module:
    """创建 PyTorch 嵌入模型"""
    models = {
        'random': TorchRandomProjection,
        'mobilenetv2': TorchMobileNetV2Embed,
        'vit': TorchViTEmbed,
    }
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    return models[model_type](device=device, **kwargs)


# 兼容 embed_models.py 接口的批量提取函数
class EmbedModelWrapper:
    """包装器：使 embed_models.py 的 EmbedModel 兼容批量处理"""
    
    def __init__(self, embed_model, device='cpu'):
        self.embed_model = embed_model
        self.device = device
    
    def extract_batch(self, images: torch.Tensor) -> torch.Tensor:
        """批量提取特征
        
        Args:
            images: (B, 3, H, W) float32 tensor, 已做 ImageNet 归一化
            
        Returns:
            (B, 4, 4) float32 tensor
        """
        B = images.shape[0]
        features_list = []
        
        for i in range(B):
            # 转为 numpy (H, W, C) 格式
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            # 反归一化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)
            img_np = (img_np * 255).astype(np.uint8)
            
            # 提取特征
            feat = self.embed_model.extract(img_np.astype(np.float32) / 255.0)  # (16,)
            features_list.append(feat)
        
        features = torch.from_numpy(np.stack(features_list)).float().to(self.device)
        return features.view(B, 4, 4)


if __name__ == '__main__':
    # 测试
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"测试设备: {device}")
    
    # 测试随机投影
    embed = TorchRandomProjection().to(device)
    dummy = torch.randn(2, 3, 32, 32).to(device)
    out = embed(dummy)
    print(f"RandomProjection: {dummy.shape} -> {out.shape}")
    
    # 测试 MobileNetV2
    try:
        embed = TorchMobileNetV2Embed(device=device).to(device)
        out = embed(dummy)
        print(f"MobileNetV2: {dummy.shape} -> {out.shape}")
    except Exception as e:
        print(f"MobileNetV2 测试失败: {e}")