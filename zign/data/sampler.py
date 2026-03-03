# from torch.utils.data import Sampler
# import random


# class EpochSplitSampler(Sampler):
#     def __init__(self, dataset_size, num_epochs, epoch, shuffle=True):
#         self.dataset_size = dataset_size
#         self.num_epochs = num_epochs
#         self.epoch = epoch
#         self.shuffle = shuffle
#         self.split_size = dataset_size // num_epochs

#     def __iter__(self):
#         start = self.epoch * self.split_size
#         end = min(start + self.split_size, self.dataset_size)
#         print(start, end)
#         indices = list(range(start, end))
#         if self.shuffle:
#             random.shuffle(indices)
#         return iter(indices)

#     def __len__(self):
#         return self.split_size
  
  
import random
from torch.utils.data import Sampler

class EpochSplitSampler(Sampler):
    def __init__(self, dataset_size, num_epochs, epoch, shuffle=True, even_split=True):
        """
        Args:
            dataset_size (int): 数据集大小
            num_epochs (int): 总共要切分成多少份
            epoch (int): 当前 epoch 索引 (0 ~ num_epochs-1)
            shuffle (bool): 是否在子集内部打乱
            even_split (bool): 是否均匀分配余数
                              True -> 前几份多 1 个样本
                              False -> 全部余数放到最后一份
        """
        self.dataset_size = dataset_size
        self.num_epochs = num_epochs
        self.epoch = epoch
        self.shuffle = shuffle
        self.even_split = even_split

        # 计算每份大小
        base = dataset_size // num_epochs
        remainder = dataset_size % num_epochs

        if even_split:
            # 前 remainder 个子集多 1 个样本
            self.sizes = [base + (1 if i < remainder else 0) for i in range(num_epochs)]
        else:
            # 最后一份包含余数
            self.sizes = [base] * num_epochs
            self.sizes[-1] += remainder

        # 计算当前 epoch 的起止索引
        start = sum(self.sizes[:epoch])
        end = start + self.sizes[epoch]
        self.indices = list(range(start, end))
        if self.shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
