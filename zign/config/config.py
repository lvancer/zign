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
        self.pretrained = '.pretrained'

        self.num_epochs = 10
        self.batch_size = 64
        self.shuffle = True
        self.lr = 0.0002
        
        self.save_iter_freq = 0
        self.save_epoch_freq = 1
        
    def pretrained_path(self):
        return os.path.join(self.pretrained, self.mode)
    
    def save_path(self):
        return os.path.join(self.save_dir, self.mode)
        
    def output_path(self):
        return os.path.join(self.output_dir, self.mode)
    
    def dataset_path(self):
        return os.path.join(self.dataset_dir, self.mode)