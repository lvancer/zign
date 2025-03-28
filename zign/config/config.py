from zign.config.abc import BaseConfig
import os

class zConfig(BaseConfig):
    
    def __init__(self):
        super().__init__()
        
        self.mode = 'default'
        self.device = "cuda"
        self.dataset_dir = ".data"
        self.save_dir = '.checkpoints'
        self.run_dir = '.runs'
        self.output_dir = '.output'
        self.pretrained_dir = '.pretrained'
        
        self.desc = '' # 训练描述
        
        self.save_iter_freq = 0 # 每训练几次iter保存一次模型，0表示不保存
        self.save_epoch_freq = 1 # 每训练几次epoch保存一次模型

        self.num_epochs = 10
        self.batch_size = 64
        self.shuffle = True
        self.dataset = ''
        
        # 学习率相关
        self.lr = 0.0002
        
        # 图像
        self.image_size = 256 # Resize the input image to this size
        
        # 优化器
        # Adam
        self.opt_adam_beta1 = 0.9
        self.opt_adam_beta2 = 0.999
        self.opt_adam_weight_decay = 0
        self.opt_adam_epsilon = 1e-8
        
        # Scheduler
        self.lr_verbose = False
        # ReduceLROnPlateau
        self.lr_plateau_mode = 'min' # 如果选择 'max'，则表明我们关注的是提升(如准确率)
        self.lr_plateau_patience = 5 # 在监测指标没有改善的情况下，等待多少个 epoch 后再降低学习率
        self.lr_plateau_threshold = 1e-4 # 指标改善的一个阈值，只有当指标改善超过这个阈值时，才认为是有效的改进
        self.lr_plateau_factor = 0.1    # 学习率的降低倍数
        # Linear
        self.lr_linear_decay = 0    # 开始下降学习率的epoch
        # Step
        self.lr_step_size = 0       # 每隔多少个epoch或训练步骤降低学习率
        self.lr_step_gamma = 0.1    # 学习率的降低倍数
        # Cosine
        self.lr_cosine_T_max = 0    # 完成一个完整的余弦周期所需的 epoch 数
        self.lr_cosine_eta_min = 0  # 学习率可以达到的最小值，即学习率将逐渐减小到该值。
        
        
    def pretrained_path(self):
        return os.path.join(self.pretrained_dir, self.mode)
    
    def save_path(self):
        return os.path.join(self.save_dir, self.mode)
        
    def output_path(self):
        return os.path.join(self.output_dir, self.mode)
    
    def dataset_path(self):
        return os.path.join(self.dataset_dir, self.mode)
    
    
    