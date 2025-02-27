from typing import TypeVar, Generic, Tuple, Optional
from zign.config import zConfig
from zign import log
import time
from zign.utils import to, io
import torch
from torch import nn
import shutil
import os

Co = TypeVar('Co', bound=zConfig)

class zTrainer(Generic[Co]):

    def __init__(self, config: Optional[Co]):
        self.config = config
    
    def forward(self, data):
        pass
    
    def backward(self, data, outputs):
        pass
    
    def update_learning_rate(self, epoch, losses):
        pass
    
    def update_learning_rate_iter(self, iter, epoch, losses):
        pass
    
    def save_models(self)-> dict[str, nn.Module]:
        pass
    
    def validate_one_iter(self, idx, inputs):
        pass
    
    def validate(self, dataloader):
        with torch.no_grad():
            total_loss = None
            for idx, inputs in enumerate(dataloader):
                losses = self.validate_one_iter(idx, inputs)
                if total_loss is None:
                    total_loss = losses
                else:
                    total_loss = to.apply_operation_on_tensors(total_loss, losses, torch.add)
            total_loss = to.apply_operation_on_tensors(total_loss, len(dataloader), torch.div)
            return total_loss

    def train_one_iter(self, idx, inputs):
        outputs = self.forward(inputs)
        losses = self.backward(inputs, outputs)
        return outputs, losses
    
    def train_one_epoch(self, epoch, dataloader):
        total_loss = None
        for idx, inputs in enumerate(dataloader):
            outputs, losses = self.train_one_iter(idx, inputs)
            if total_loss is None:
                total_loss = losses
            else:
                total_loss = to.apply_operation_on_tensors(total_loss, losses, torch.add)
            self.log_iter_end(epoch, idx, dataloader, losses, outputs)
            self.save_iter(epoch, idx, dataloader)
            self.update_learning_rate_iter(idx, epoch, losses)
        total_loss = to.apply_operation_on_tensors(total_loss, len(dataloader), torch.div)
        return total_loss
    
    def train(self, train_dataset, val_dataset=None):
        dataloader = train_dataset.dataloader(self.config.batch_size, self.config.shuffle)
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            total_losses = self.train_one_epoch(epoch, dataloader)
            end_time = time.time()
            self.log_epoch_end(epoch, total_losses, end_time - start_time)
            self.save_epoch(epoch)
            self.update_learning_rate(epoch, total_losses)
            if val_dataset is not None:
                total_losses = self.validate(val_dataset.dataloader(self.config.batch_size, self.config.shuffle))
                self.log_val_end(epoch, total_losses)
                
    def log_iter_end(self, epoch, idx, dataloader, losses, outputs):
        msg = f"Epoch: {epoch}, Batch[{idx}/{len(dataloader)}], Loss: {to.str_tensors_values(losses)}"
        log.info(msg)

    def log_epoch_end(self, epoch, train_loss, duration):
        msg = f"Epoch: {epoch}, Loss: {to.str_tensors_values(train_loss)}, Epoch time = {duration:.3f}s"
        log.info(msg)
                  
    def log_val_end(self, epoch, losses):
        msg = f"Epoch: {epoch}, Validate Loss: {to.str_tensors_values(losses)},"
        log.info(msg)
                  
    def save_iter(self, epoch, idx, dataloader):
        cur_iter = epoch * len(dataloader) + idx
        if self.config.save_iter_freq > 0 and cur_iter > 0 and cur_iter % self.config.save_iter_freq == 0:
            save_paths = io.save_model(self.save_models(), 'iter_%s' % cur_iter, self.config.save_dir)
            self.save_latest(save_paths)

    def save_epoch(self, epoch):
        if self.config.save_epoch_freq > 0 and epoch % self.config.save_epoch_freq == 0:
            save_paths = io.save_model(self.save_models(), 'epoch_%s' % epoch, self.config.save_dir)
            self.save_latest(save_paths)
            
    def save_latest(self, save_paths):
        for name, save_path in save_paths.items():
            shutil.copyfile(save_path, os.path.join(self.config.save_dir, 'latest_%s.pth' % name))
            log.info('saved model {} at {}'.format(name, save_path))

