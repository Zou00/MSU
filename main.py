import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('./utils')
sys.path.append('./utils/APIs')

import torch

import argparse
from Config import Config
config = Config()
from utils.common import data_format, read_from_file, train_val_split, save_model, write_to_file
from utils.DataProcess import Processor
from Trainer import Trainer


# args
parser = argparse.ArgumentParser()
parser.add_argument('--do_train', action='store_true', help='训练模型')
parser.add_argument('--text_pretrained_model', default='roberta-base', help='文本分析模型', type=str)
parser.add_argument('--fuse_model_type', default='EnhancedOTE', help='融合模型类别(CMAC/OTE/EnhancedOTE/NaiveCat/NaiveCombine)', type=str)
parser.add_argument('--lr', default=5e-5, help='设置学习率', type=float)
parser.add_argument('--weight_decay', default=1e-2, help='设置权重衰减', type=float)
parser.add_argument('--epoch', default=10, help='设置训练轮数', type=int)
parser.add_argument('--dataset', default='all_labels', help='选择数据集(all_labels/consistent_labels)', type=str)
parser.add_argument('--use_focal_loss', action='store_true', help='是否使用Focal Loss')
parser.add_argument('--focal_loss_gamma', default=2.0, help='Focal Loss的gamma参数', type=float)
parser.add_argument('--label_smoothing', default=0.1, help='标签平滑参数', type=float)
parser.add_argument('--gradient_clip', default=1.0, help='梯度裁剪阈值', type=float)
parser.add_argument('--do_test', action='store_true', help='预测测试集数据')
parser.add_argument('--load_model_path', default=None, help='已经训练好的模型路径', type=str)
parser.add_argument('--text_only', action='store_true', help='仅用文本预测')
parser.add_argument('--img_only', action='store_true', help='仅用图像预测')
args = parser.parse_args()

# 设置数据集路径
config.train_data_path = os.path.join(config.root_path, f'data/{args.dataset}/train.json')
config.val_data_path = os.path.join(config.root_path, f'data/{args.dataset}/val.json')
config.test_data_path = os.path.join(config.root_path, f'data/{args.dataset}/test.json')

# 设置模型参数
config.learning_rate = args.lr
config.weight_decay = args.weight_decay
config.epoch = args.epoch
config.bert_name = args.text_pretrained_model
config.fuse_model_type = args.fuse_model_type
config.load_model_path = args.load_model_path
config.only = 'img' if args.img_only else None
config.only = 'text' if args.text_only else None
if args.img_only and args.text_only: config.only = None

# 设置数据增强参数
config.use_focal_loss = args.use_focal_loss
config.focal_loss_gamma = args.focal_loss_gamma
config.label_smoothing = args.label_smoothing
config.gradient_clip = args.gradient_clip

print('TextModel: {}, ImageModel: {}, FuseModel: {}, Dataset: {}'.format(
    config.bert_name, 'ResNet50', config.fuse_model_type, args.dataset))

# Initilaztion
processor = Processor(config)

def get_model(config):
    """获取模型实例"""
    if config.fuse_model_type == 'CMAC' or config.fuse_model_type == 'CrossModalityAttentionCombine':
        from Models.CMACModel import Model
    elif config.fuse_model_type == 'OTE' or config.fuse_model_type == 'OutputTransformerEncoder':
        from Models.OTEModel import Model
    elif config.fuse_model_type == 'EnhancedOTE':
        from Models.EnhancedOTEModel import EnhancedModel as Model
    elif config.fuse_model_type == 'NaiveCat':
        from Models.NaiveCatModel import Model
    else:
        from Models.NaiveCombineModel import Model
    
    return Model(config)

if __name__ == '__main__':
    if args.do_train:
        # 读取训练数据
        train_data = read_from_file(config.train_data_path, config.data_dir, config.only)
        val_data = read_from_file(config.val_data_path, config.data_dir, config.only)
        
        # 获取数据加载器
        train_loader = processor(train_data, config.train_params)
        val_loader = processor(val_data, config.val_params)
        
        # 获取模型
        model = get_model(config)
        
        # 训练模型
        trainer = Trainer(config, processor, model, train_loader)
        trainer.train(train_loader, val_loader)
        
    if args.do_test:
        # 读取测试数据
        test_data = read_from_file(config.test_data_path, config.data_dir, config.only)
        test_loader = processor(test_data, config.test_params)
        
        # 获取模型
        model = get_model(config)
        
        # 加载模型权重
        if config.load_model_path is not None:
            model.load_state_dict(torch.load(config.load_model_path))
        
        # 测试模型
        trainer = Trainer(config, processor, model, test_loader)
        trainer.test(test_loader)