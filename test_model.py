import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from Config import config
from utils.common import read_from_file
from utils.DataProcess import Processor
from main import get_model

def parse_args():
    parser = argparse.ArgumentParser(description='测试多模态情感分析模型')
    parser.add_argument('--model_path', type=str, default=None, help='模型路径')
    parser.add_argument('--dataset', type=str, default='all_labels', choices=['all_labels', 'consistent_labels'], help='数据集选择')
    parser.add_argument('--text_pretrained_model', type=str, default='roberta-base', help='文本预训练模型')
    parser.add_argument('--fuse_model_type', type=str, default='OTEModel', help='融合模型类型')
    parser.add_argument('--text_only', action='store_true', help='仅使用文本模态')
    parser.add_argument('--img_only', action='store_true', help='仅使用图像模态')
    parser.add_argument('--output_dir', type=str, default='output/test_results', help='测试结果输出目录')
    return parser.parse_args()

def plot_confusion_matrix(cm, labels, output_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def test_model(args):
    # 设置配置
    config.bert_name = args.text_pretrained_model
    config.fuse_model_type = args.fuse_model_type
    config.only = 'img' if args.img_only else None
    config.only = 'text' if args.text_only else None
    if args.img_only and args.text_only:
        config.only = None
    
    # 设置数据集路径
    config.test_data_path = os.path.join(config.root_path, f'data/{args.dataset}/test.json')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查模型路径是否存在
    if args.model_path is None:
        # 尝试找到默认模型路径
        default_model_path = os.path.join(config.output_path, args.fuse_model_type, "pytorch_model.bin")
        if os.path.exists(default_model_path):
            args.model_path = default_model_path
            print(f"使用默认模型路径: {args.model_path}")
        else:
            raise ValueError("请提供有效的模型路径 --model_path 参数")
    
    # 初始化模型和处理器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"成功加载模型: {args.model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        raise
    
    model = model.to(device)
    model.eval()
    
    processor = Processor(config)
    
    # 检查测试数据文件是否存在
    if not os.path.exists(config.test_data_path):
        raise FileNotFoundError(f"测试数据文件不存在: {config.test_data_path}")
    
    # 加载测试数据
    print(f"正在加载测试数据集: {args.dataset}")
    try:
        test_data = read_from_file(config.test_data_path, config.data_dir, config.only)
        test_loader = processor(test_data, config.test_params)
    except Exception as e:
        print(f"加载测试数据失败: {e}")
        raise
    
    # 测试模型
    print("开始测试模型...")
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试中"):
            guids, texts, texts_mask, imgs, labels = batch
            texts = texts.to(device)
            texts_mask = texts_mask.to(device)
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # 根据模型返回值的不同处理预测结果
            outputs = model(texts, texts_mask, imgs)
            
            # 如果模型返回元组 (pred_labels, loss)，取第一个元素
            if isinstance(outputs, tuple):
                pred = outputs[0]
            else:
                # 如果模型直接返回预测值
                pred = outputs
            
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(pred.cpu().numpy())
    
    # 计算评估指标
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    # 获取分类报告
    class_report = classification_report(true_labels, pred_labels, output_dict=True)
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    
    # 保存结果
    result_file = os.path.join(args.output_dir, f"{args.fuse_model_type}_{args.dataset}_results.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"模型: {args.fuse_model_type}\n")
        f.write(f"数据集: {args.dataset}\n")
        f.write(f"文本预训练模型: {args.text_pretrained_model}\n")
        f.write(f"模态: {'仅文本' if args.text_only else '仅图像' if args.img_only else '文本+图像'}\n\n")
        
        f.write(f"准确率: {accuracy:.4f}\n")
        f.write(f"F1分数: {f1:.4f}\n\n")
        
        f.write("分类报告:\n")
        for label, metrics in class_report.items():
            if isinstance(metrics, dict):
                f.write(f"{label}:\n")
                for metric_name, value in metrics.items():
                    f.write(f"  {metric_name}: {value:.4f}\n")
                f.write("\n")
    
    # 绘制混淆矩阵
    cm_file = os.path.join(args.output_dir, f"{args.fuse_model_type}_{args.dataset}_confusion_matrix.png")
    label_names = ['negative', 'neutral', 'positive']
    plot_confusion_matrix(cm, label_names, cm_file)
    
    # 打印结果
    print(f"\n测试结果已保存到: {result_file}")
    print(f"混淆矩阵已保存到: {cm_file}")
    print(f"\n准确率: {accuracy:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    return accuracy, f1

if __name__ == "__main__":
    args = parse_args()
    test_model(args) 