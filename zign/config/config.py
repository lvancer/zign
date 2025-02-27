from zign.config.abc import BaseConfig
import os

class zConfig(BaseConfig):
    
    def __init__(self):
        super().__init__()
        
        self.app_name = os.path.basename(os.path.abspath('.'))
        self.device = "cuda"
        self.dataset_dir = ".data"
        self.save_dir = '.checkpoints'

        self.num_epochs = 10
        self.batch_size = 64
        self.shuffle = True
        self.lr = 0.0002
        
        self.save_iter_freq = 0
        self.save_epoch_freq = 1
        
        
        
        
        