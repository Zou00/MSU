import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import os
from datetime import datetime
from utils.common import save_model

# 配置日志
def setup_logger(config):
    logger = logging.getLogger('trainer')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建logs目录
        log_dir = os.path.join(config.root_path, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 使用时间戳创建日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{config.fuse_model_type}_{timestamp}.log')
        
        # 创建文件处理器，指定UTF-8编码
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

class Trainer():

    def __init__(self, config, processor, model, train_loader, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        self.config = config
        self.processor = processor
        self.model = model.to(device)
        self.device = device
        self.logger = setup_logger(config)
        
        # 使用统一参数
        self.learning_rate = config.learning_rate
        self.epoch = config.epoch
        self.warmup_steps = config.warmup_steps
        self.batch_size = config.train_params['batch_size']
        
        # 设置默认值
        self.dropout_rate = getattr(config, 'dropout_rate', 0.1)
        self.label_smoothing = getattr(config, 'label_smoothing', 0.1)
        
        # 打印模型配置信息
        self.logger.info("=" * 50)
        self.logger.info("模型初始化配置:")
        self.logger.info("-" * 30)
        self.logger.info(f"设备: {device}")
        self.logger.info(f"模型类型: {config.fuse_model_type}")
        self.logger.info(f"学习率: {self.learning_rate}")
        self.logger.info(f"权重衰减: {config.weight_decay}")
        self.logger.info(f"训练轮数: {self.epoch}")
        self.logger.info(f"Warmup步数: {self.warmup_steps}")
        self.logger.info(f"Batch大小: {self.batch_size}")
        self.logger.info(f"Dropout率: {self.dropout_rate}")
        self.logger.info(f"Label Smoothing: {self.label_smoothing}")
        self.logger.info("=" * 50)
       
        # 分离不同模块的参数
        bert_params = set(self.model.text_model.bert.parameters())
        resnet_params = set(self.model.img_model.full_resnet.parameters())
        other_params = list(set(self.model.parameters()) - bert_params - resnet_params)
        
        # 设置不同模块的学习率和权重衰减
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in self.model.text_model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': self.learning_rate, 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.text_model.bert.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': self.learning_rate, 'weight_decay': 0.0},
            {'params': [p for n, p in self.model.img_model.full_resnet.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': self.learning_rate, 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.img_model.full_resnet.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': self.learning_rate, 'weight_decay': 0.0},
            {'params': other_params,
                'lr': self.learning_rate, 'weight_decay': 0.01},
        ]
        
        # 使用AdamW优化器
        self.optimizer = AdamW(params, lr=self.learning_rate)
        
        # 计算总训练步数
        num_training_steps = len(train_loader) * self.epoch
        
        # 使用带warmup的学习率调度器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # 设置损失函数
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing).to(device)

    def train(self, train_loader, val_loader):
        best_f1 = 0
        patience = 5
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(self.epoch):
            self.logger.info('\n' + '=' * 50)
            self.logger.info(f'Epoch {epoch+1}/{self.epoch}')
            self.logger.info('-' * 30)
            
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_true_labels, train_pred_labels = [], []
            
            for batch in tqdm(train_loader, desc='----- [Training] '):
                guids, texts, texts_mask, imgs, labels = batch
                texts = texts.to(self.device)
                texts_mask = texts_mask.to(self.device)
                imgs = imgs.to(self.device)
                labels = labels.to(self.device).long()
                
                outputs = self.model(texts, texts_mask, imgs, labels=labels)
                if isinstance(outputs, tuple):
                    pred, loss = outputs
                else:
                    pred = outputs
                    loss = self.criterion(pred, labels)
                
                train_loss += loss.item()
                train_true_labels.extend(labels.cpu().tolist())
                
                # 处理预测结果
                if len(pred.shape) == 1:
                    train_pred_labels.extend(pred.cpu().tolist())
                else:
                    train_pred_labels.extend(pred.argmax(dim=-1).cpu().tolist())
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # 应用梯度裁剪
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.gradient_clip)
                
                self.optimizer.step()
                self.scheduler.step()
            
            avg_train_loss = train_loss / len(train_loader)
            train_metrics = self.processor.metric(train_true_labels, train_pred_labels)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            val_true_labels, val_pred_labels = [], []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='----- [Validating] '):
                    guids, texts, texts_mask, imgs, labels = batch
                    texts = texts.to(self.device)
                    texts_mask = texts_mask.to(self.device)
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device).long()
                    
                    outputs = self.model(texts, texts_mask, imgs, labels=labels)
                    if isinstance(outputs, tuple):
                        pred, loss = outputs
                    else:
                        pred = outputs
                        loss = self.criterion(pred, labels)
                    
                    val_loss += loss.item()
                    val_true_labels.extend(labels.cpu().tolist())
                    
                    # 处理预测结果
                    if len(pred.shape) == 1:
                        val_pred_labels.extend(pred.cpu().tolist())
                    else:
                        val_pred_labels.extend(pred.argmax(dim=-1).cpu().tolist())
            
            avg_val_loss = val_loss / len(val_loader)
            val_metrics = self.processor.metric(val_true_labels, val_pred_labels)
            
            # 记录训练和验证指标
            self.logger.info(f'Train Loss: {avg_train_loss:.4f}')
            self.logger.info(f'Train Metrics: {train_metrics}')
            self.logger.info(f'Val Loss: {avg_val_loss:.4f}')
            self.logger.info(f'Val Metrics: {val_metrics}')
            
            # 保存最佳模型
            current_f1 = val_metrics['weighted avg']['f1-score']
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_epoch = epoch + 1
                patience_counter = 0
                save_model(self.config.output_path, self.config.fuse_model_type, self.model)
                self.logger.info(f'保存最佳模型，F1: {best_f1:.4f}')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                    break
            
            self.logger.info(f'Best F1: {best_f1:.4f} at epoch {best_epoch}')

    def predict(self, test_loader):
        self.model.eval()
        self.logger.info("\n" + "=" * 50)
        self.logger.info("开始预测...")
        self.logger.info("-" * 30)
        
        pred_guids, pred_labels = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc='----- [Predicting] '):
                guids, texts, texts_mask, imgs, labels = batch
                texts = texts.to(self.device)
                texts_mask = texts_mask.to(self.device)
                imgs = imgs.to(self.device)
                
                pred = self.model(texts, texts_mask, imgs)
                
                pred_guids.extend(guids)
                pred_labels.extend(pred.tolist())

        self.logger.info("预测完成")
        self.logger.info("-" * 30)
        return [(guid, label) for guid, label in zip(pred_guids, pred_labels)]
