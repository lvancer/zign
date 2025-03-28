import json
from typing import TypeVar, Generic, Tuple, Optional
from zign.config import zConfig
import time
from zign.utils import to, io
import torch
from torch import nn
import shutil
import os
from zign.log.tensorboard import zSummaryWriter
import logging
from tqdm import tqdm


Co = TypeVar('Co', bound=zConfig)

class zTrainer(Generic[Co]):

    def __init__(self, config: Optional[Co]):
        self.config = config
        self._summary_writer = zSummaryWriter(self.config)
        self.get_summary_writer().add_text("Config", json.dumps(self.config.to_dict(), indent=4))
        self.init_current_info()
        
    def get_summary_writer(self):
        return self._summary_writer
        
    def init_current_info(self):
        self._current_epoch = 0 # current epoch 1-N
        self._current_iter = 0 # current iteration in one epoch 1-N
        self._num_batch = 0  # number of batches in one epoch
    
    def float_precision_format(self):
        return '.3f'
    
    def update_learning_rate(self, loss, val_losses):
        pass
    
    def update_learning_rate_iter(self, losses):
        pass
    
    def save_models(self)-> dict[str, nn.Module]:
        pass
    
    def graph_models(self):
        pass
        
    def validate_one_iter(self, idx, inputs):
        pass
    
    def validate(self, dataloader):
        with torch.no_grad():
            total_loss = None
            pbar = tqdm(dataloader)
            pbar.set_description(f"[Epoch {self.current_epoch()} Validate]")
            for idx, inputs in enumerate(pbar):
                losses = self.validate_one_iter(idx, inputs)
                if total_loss is None:
                    total_loss = losses
                else:
                    total_loss = to.apply_operation_on_tensors(total_loss, losses, torch.add)
                pbar.set_postfix(to.tensors_to_item(losses, self.float_precision_format()))
            if total_loss is None:
                return None
            return to.apply_operation_on_tensors(total_loss, len(dataloader), torch.div)

    def train_one_iter(self, inputs)-> dict[str, torch.Tensor]:
        return None
    
    def train_one_epoch(self, dataloader):
        total_loss = None
        pbar = tqdm(dataloader)
        pbar.set_description(f"[Epoch {self.current_epoch()}]")
        self._current_iter = 0
        for idx, inputs in enumerate(pbar):
            self._current_iter = idx + 1
            losses = self.train_one_iter(inputs)
            if total_loss is None:
                total_loss = losses
            else:
                total_loss = to.apply_operation_on_tensors(total_loss, losses, torch.add)
            self.save_iter()
            self.update_learning_rate_iter(losses)
            self.log_iter_end(losses)
            pbar.set_postfix(to.tensors_to_item(losses, self.float_precision_format()))
        if total_loss is None:
            return None
        return to.apply_operation_on_tensors(total_loss, self.num_batch(), torch.div)
    
    def train(self, train_dataset, val_dataset=None, val_batch_size: int=0, val_shuffle: bool=False):
        logging.info('starting training')

        if self.graph_models() is not None:
            self.get_summary_writer().add_graphs(self.save_models(), self.graph_models(), self.config.device)
        
        dataloader = train_dataset.dataloader(self.config.batch_size, self.config.shuffle)
        self._num_batch = len(dataloader)
        for epoch in range(1, self.config.num_epochs+1):
            self._current_epoch = epoch
            start_time = time.time()
            self.before_train_epoch()
            losses = self.train_one_epoch(dataloader)
            self.after_train_epoch()
            self.save_epoch()
            val_losses = None
            if val_dataset is not None:
                batch_size = val_batch_size if val_batch_size > 0 else self.config.batch_size
                val_losses = self.validate(val_dataset.dataloader(batch_size, val_shuffle))
            self.update_learning_rate(losses, val_losses)
            self.log_epoch_end(losses, val_losses, time.time() - start_time)
        self.init_current_info()
        self.get_summary_writer().close()
        
    def before_train_epoch(self):
        pass
    
    def after_train_epoch(self):
        pass
    
    def current_step(self):
        return (self.current_epoch() - 1) * self._num_batch + self.current_iter()
    
    def current_epoch(self):
        return self._current_epoch
    
    def current_iter(self):
        return self._current_iter
    
    def num_batch(self):
        return self._num_batch
    
    def log_iter_end(self, losses):
        step = self.current_step()
        self.get_summary_writer().add_losses("Step", losses, step)

    def log_epoch_end(self, train_losses, val_losses, duration):
        msg = f"Epoch: {self.current_epoch()}, Epoch time = {duration:.3f}s, Train Loss: {to.tensors_to_item(train_losses, self.float_precision_format())}"
        if val_losses is not None:
            msg = msg + f", Validate Loss: {to.tensors_to_item(val_losses, self.float_precision_format())}"
        logging.info(msg)
        losses = {f"train_{key}": value for key, value in train_losses.items()}
        if val_losses is not None:
            losses.update({f"val_{key}": value for key, value in val_losses.items()})
        self.get_summary_writer().add_losses("Epoch", losses, self.current_epoch())
    
    def save_iter(self):
        step = self.current_step()
        if self.config.save_iter_freq > 0 and step > 0 and step % self.config.save_iter_freq == 0:
            save_paths = io.save_model(self.save_models(), f"iter_{step}", os.path.join(self.config.save_path(), self.get_summary_writer().folder()))
            self.save_latest(save_paths)

    def save_epoch(self):
        epoch = self.current_epoch()
        if (self.config.save_epoch_freq > 0 and epoch % self.config.save_epoch_freq == 0) or epoch == self.config.num_epochs:
            save_paths = io.save_model(self.save_models(), f"epoch_{epoch}", os.path.join(self.config.save_path(), self.get_summary_writer().folder()))
            self.save_latest(save_paths)
            
    def save_latest(self, save_paths):
        for name, save_path in save_paths.items():
            shutil.copyfile(save_path, os.path.join(self.config.save_path(), self.get_summary_writer().folder(), 'latest_%s.pth' % name))
            logging.info('saved model {} at {}'.format(name, save_path))
            self.get_summary_writer().add_text("Save", save_path, self.current_epoch())
            


        




