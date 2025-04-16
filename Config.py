import os

class Config:
    # 根目录
    root_path = os.getcwd()
    data_dir = os.path.join(root_path, './data/data/')
    # 数据集路径将在运行时设置
    train_data_path = None
    val_data_path = None
    test_data_path = None
    output_path = os.path.join(root_path, 'output')
    output_test_path = os.path.join(output_path, 'test.txt')
    load_model_path = None

    # 一般超参
    epoch = 20
    learning_rate = 3e-5
    weight_decay = 0.01
    num_labels = 3
    loss_weight = [1.85, 5.52, 3.58]  # 基于逆频率计算的权重

    # Fuse相关
    fuse_model_type = 'EnhancedOTE'
    only = None
    middle_hidden_size = 64
    hidden_size = 64
    attention_nhead = 8
    attention_dropout = 0.4
    fuse_dropout = 0.5
    out_hidden_size = 128
    fusion_dropout = 0.2
    classifier_dropout = 0.2

    # BERT相关
    fixed_text_model_params = False
    bert_name = 'roberta-base'
    bert_learning_rate = 5e-6
    bert_dropout = 0.2

    # ResNet相关
    fixed_img_model_params = False
    image_size = 224
    fixed_image_model_params = True
    resnet_learning_rate = 5e-6
    resnet_dropout = 0.2
    img_hidden_seq = 64

    # Dataloader params
    checkout_params = {'batch_size': 4, 'shuffle': False}
    train_params = {'batch_size': 16, 'shuffle': True, 'num_workers': 2}
    val_params = {'batch_size': 16, 'shuffle': False, 'num_workers': 2}
    test_params =  {'batch_size': 8, 'shuffle': False, 'num_workers': 2}

    # 添加warmup_steps属性
    warmup_steps = 300

    # 数据增强相关参数
    use_focal_loss = True
    focal_loss_gamma = 2.0
    label_smoothing = 0.1
    gradient_clip = 1.0
    layer_norm_eps = 1e-12

    def __init__(self):
        # 数据集路径
        self.root_path = os.getcwd()
        self.data_dir = os.path.join(self.root_path, 'data/data')
        self.output_path = os.path.join(self.root_path, 'output')
        
        # 确保输出目录存在
        os.makedirs(self.output_path, exist_ok=True)
        
        # 数据集路径
        self.train_data_path = os.path.join(self.root_path, 'data/all_labels/train.json')
        self.val_data_path = os.path.join(self.root_path, 'data/all_labels/val.json')
        self.test_data_path = os.path.join(self.root_path, 'data/all_labels/test.json')

    
    
