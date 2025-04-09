import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchvision.models import resnet50
from Models.OTEModel import TextModel, ImageModel

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, input, target):
        # 如果使用label_smoothing，需要将target转换为one-hot编码
        if self.label_smoothing > 0:
            # 获取类别数
            num_classes = input.size(1)
            # 创建one-hot编码
            target_one_hot = torch.zeros_like(input).scatter_(1, target.unsqueeze(1), 1)
            # 应用label smoothing
            target_one_hot = target_one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            # 计算交叉熵损失
            ce_loss = -(target_one_hot * torch.log_softmax(input, dim=1)).sum(dim=1)
        else:
            # 确保target是长整型
            target = target.long()
            # 计算交叉熵损失
            ce_loss = torch.nn.functional.cross_entropy(input, target, reduction='none')
        
        # 计算focal loss
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AdaptiveFusion(nn.Module):
    def __init__(self, config):
        super(AdaptiveFusion, self).__init__()
        self.hidden_size = config.middle_hidden_size
        
        # 文本到图像的注意力
        self.text_to_image_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.attention_nhead,
            dropout=config.attention_dropout
        )
        
        # 图像到文本的注意力
        self.image_to_text_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.attention_nhead,
            dropout=config.attention_dropout
        )
        
        # 自适应门控机制
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fusion_dropout)
        )
        
        # 残差连接
        self.residual = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        )
        
        # 特征增强层
        self.enhancement = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2, eps=config.layer_norm_eps),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        )
        
        # 跨模态交互层
        self.cross_modal = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=config.attention_nhead,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.attention_dropout,
            activation='gelu',
            layer_norm_eps=config.layer_norm_eps
        )

    def forward(self, text_features, image_features):
        batch_size = text_features.size(0)
        
        # 调整特征维度
        text_features = text_features.unsqueeze(0)  # [1, batch_size, hidden_size]
        image_features = image_features.unsqueeze(0)  # [1, batch_size, hidden_size]
        
        # 文本到图像的注意力
        text_to_image_attn, _ = self.text_to_image_attention(
            text_features,
            image_features,
            image_features
        )
        
        # 图像到文本的注意力
        image_to_text_attn, _ = self.image_to_text_attention(
            image_features,
            text_features,
            text_features
        )
        
        # 调整维度
        text_enhanced = text_to_image_attn.squeeze(0)  # [batch_size, hidden_size]
        image_enhanced = image_to_text_attn.squeeze(0)  # [batch_size, hidden_size]
        
        # 自适应门控机制
        gate_values = self.gate(torch.cat([text_enhanced, image_enhanced], dim=1))
        gated_text = text_enhanced * gate_values
        gated_image = image_enhanced * (1 - gate_values)
        
        # 特征融合
        fused_features = self.fusion_layer(torch.cat([gated_text, gated_image], dim=1))
        
        # 残差连接
        residual = self.residual(torch.cat([text_features.squeeze(0), image_features.squeeze(0)], dim=1))
        fused_features = fused_features + residual
        
        # 特征增强
        enhanced_features = self.enhancement(fused_features)
        
        # 跨模态交互
        cross_modal_features = self.cross_modal(enhanced_features.unsqueeze(0)).squeeze(0)
        
        # 最终特征
        final_features = enhanced_features + cross_modal_features
        
        return final_features


class EnhancedModel(nn.Module):
    def __init__(self, config):
        super(EnhancedModel, self).__init__()
        
        # 使用原有的文本和图像模型
        self.text_model = TextModel(config)
        self.img_model = ImageModel(config)
        
        # 自适应融合
        self.adaptive_fusion = AdaptiveFusion(config)
        
        # 情感分类器
        self.classifier = nn.Sequential(
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.middle_hidden_size, config.out_hidden_size),
            nn.LayerNorm(config.out_hidden_size, eps=config.layer_norm_eps),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.out_hidden_size, config.out_hidden_size // 2),
            nn.LayerNorm(config.out_hidden_size // 2, eps=config.layer_norm_eps),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.out_hidden_size // 2, config.num_labels)
        )
        
        # 使用FocalLoss
        self.loss_func = FocalLoss(
            gamma=config.focal_loss_gamma,
            label_smoothing=config.label_smoothing
        )

    def forward(self, texts, texts_mask, imgs, labels=None):
        # 获取文本特征
        text_features = self.text_model(texts, texts_mask)
        
        # 获取图像特征
        image_features = self.img_model(imgs)
        
        # 自适应融合
        fused_features = self.adaptive_fusion(text_features, image_features)
        
        # 情感分类
        logits = self.classifier(fused_features)
        
        if labels is not None:
            loss = self.loss_func(logits, labels)
            return logits, loss
        return logits 