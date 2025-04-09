# 多模态情感理解

本项目实现了多模态情感分析，融合了BERT系列模型与ResNet50模型，通过多种融合策略对图像和文本信息进行情感分析。

## 项目结构

```
|-- 基于自适应融合机制的图文多模态情感理解
    |-- Config.py            # 配置文件
    |-- main.py              # 主程序入口
    |-- test_model.py        # 模型测试脚本
    |-- README.md            # 项目说明文件
    |-- requirements.txt     # 依赖包列表
    |-- Trainer.py           # 训练器
    |-- data                 # 数据目录
    |   |-- test_without_label.txt  # 无标签测试数据
    |   |-- data             # 原始数据
    |   |-- labelResultAll.txt      # 全部标签结果
    |   |-- all_labels       # 全标签数据集
    |   |-- consistent_labels       # 一致标签数据集
    |-- Models               # 模型目录
    |   |-- CMACModel.py     # 跨模态注意力组合模型
    |   |-- AdaptiveFusionModel.py  # 自适应融合模型
    |   |-- NaiveCatModel.py        # 简单拼接模型
    |   |-- NaiveCombineModel.py    # 简单组合模型
    |   |-- OTEModel.py             # 输出Transformer编码器模型
    |   |-- __init__.py
    |-- src                  # 资源文件
    |   |-- CrossModalityAttentionCombineModel.png  # 模型架构图
    |   |-- OutputTransformerEncoderModel.png       # 模型架构图
    |   |-- AdaptiveFusionModel.png                 # 模型架构图
    |-- utils                # 工具函数
        |-- common.py        # 通用工具
        |-- DataProcess.py   # 数据处理
        |-- __init__.py
        |-- APIs             # API接口
            |-- APIDataset.py     # 数据集API
            |-- APIDecode.py      # 解码API
            |-- APIEncode.py      # 编码API
            |-- APIMetric.py      # 评估指标API
            |-- __init__.py
```

## 环境要求

```
chardet==5.2.0
numpy==1.23.5
Pillow==9.5.0
scikit-learn==1.2.2
torch==2.0.1
torchvision==0.15.2
tqdm==4.65.0
transformers==4.30.0
matplotlib==3.7.1
seaborn==0.12.2
```

安装依赖：
```shell
pip install -r requirements.txt
```

## 模型架构

### 1. 自适应融合模型 (AdaptiveFusion)

自适应融合模型是本项目的核心模型，它采用了动态特征融合机制，具有以下特点：

- **动态权重计算**：根据输入的文本和图像特征，自动计算它们的重要性权重
- **多头自注意力机制**：使用多头自注意力处理不同模态的特征
- **自适应融合层**：根据不同模态的特征重要性，动态调整融合比例
- **深度特征提取**：使用更深的网络结构，提取更丰富的特征表示

模型结构：
1. 文本特征提取：使用BERT/RoBERTa等预训练模型提取文本特征
2. 图像特征提取：使用ResNet50提取图像特征
3. 自适应融合：通过计算特征重要性，动态融合文本和图像特征
4. 分类层：使用全连接层对融合特征进行情感分类

### 2. 跨模态注意力组合模型 (CMAC)

该模型使用注意力机制处理跨模态信息：

- 文本到图像的注意力机制
- 图像到文本的注意力机制
- 独立的分类器分别处理增强后的特征

### 3. 输出Transformer编码器模型 (OTE)

将文本和图像特征拼接后，使用Transformer编码器处理：

- 利用自注意力机制捕捉模态内和模态间的关系
- 单一分类器处理Transformer编码后的特征

### 4. 简单拼接模型 (NaiveCat)

最简单的融合方法：

- 直接拼接文本和图像特征
- 使用全连接层进行分类

### 5. 简单组合模型 (NaiveCombine)

另一种简单有效的融合方法：

- 分别对文本和图像特征进行分类
- 组合两个分类器的输出

## 实验结果

| 模型                          | 准确率(%)  | F1分数(%) |
| ----------------------------- | ---------- | --------- |
| NaiveCat                      | 71.25      | 70.18     |
| NaiveCombine                  | 73.625     | 72.45     |
| CrossModalityAttentionCombine | 67.1875    | 66.32     |
| OutputTransformerEncoder      | 74.625     | 73.78     |
| **AdaptiveFusion**            | **76.25**  | **75.42** |

### 消融实验

自适应融合模型在单模态条件下的性能：

| 特征       | 准确率(%) | F1分数(%) |
| ---------- | --------- | --------- |
| 仅文本     | 71.875    | 70.92     |
| 仅图像     | 63.00     | 62.15     |
| 完整融合   | 76.25     | 75.42     |

## 使用方法

### 训练模型

```shell
python main.py --do_train --epoch 10 --text_pretrained_model roberta-base --fuse_model_type AdaptiveFusion --dataset all_labels
```

主要参数说明：
- `--do_train`: 训练模式
- `--epoch`: 训练轮数
- `--text_pretrained_model`: 文本预训练模型名称
- `--fuse_model_type`: 融合模型类型，可选CMAC、OTE、NaiveCat、NaiveCombine、AdaptiveFusion
- `--dataset`: 数据集选择，可选all_labels（全标签数据集）、consistent_labels（一致标签数据集）
- `--text_only`: 仅使用文本模态
- `--img_only`: 仅使用图像模态

### 测试模型

```shell
python main.py --do_test --text_pretrained_model roberta-base --fuse_model_type AdaptiveFusion --load_model_path $your_model_path$ --dataset all_labels
```

### 详细评估

```shell
python test_model.py --model_path $your_model_path$ --fuse_model_type AdaptiveFusion --dataset all_labels
```

评估输出内容：
- 准确率和F1分数
- 每个类别的精确率、召回率和F1分数
- 混淆矩阵可视化
- 详细的测试报告（保存为文本文件）

## 项目特色

1. **多模型比较**：实现了多种融合策略，便于比较不同方法的性能
2. **自适应融合**：核心创新点在于自适应融合机制，能动态调整模态重要性
3. **完整评估**：提供详细的评估指标和可视化分析工具
4. **灵活配置**：通过Config.py可以灵活调整各种参数设置
5. **模块化设计**：代码结构清晰，易于扩展和修改

## 关键技术

1. **预训练模型应用**：利用BERT/RoBERTa等预训练模型提取文本特征
2. **深度卷积网络**：使用ResNet50提取图像特征
3. **注意力机制**：利用注意力机制增强跨模态特征交互
4. **自适应融合**：动态计算特征重要性，实现智能融合
5. **模型集成**：结合多种模型的优点，提高系统整体性能
