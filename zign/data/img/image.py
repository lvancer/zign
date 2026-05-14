import random
from zign.data import zDataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch


class zImageDataset(zDataset):
    '''
    '''
    
    path = ''
    
    def __init__(self, dataset_dir, file_list, size=(256, 256), **kwargs):
        super().__init__(**kwargs)
        self.dataset_dir = dataset_dir
        self.file_list = file_list
        self.root = os.path.join(self.dataset_dir, self.path)
        self.file_pairs = self.create_file_pairs()
        self.size = size
        if self.cache:
            self._cache = {}
            
    def __len__(self):
        return len(self.file_pairs)
        
    def create_file_pairs(self):
        file_pairs = []
        file_list = os.path.join(self.root, self.file_list)
        for line in open(file_list, 'r'):
            file = line.strip()
            if file == '':
                continue
            pair = self.get_file_pair(file)
            if pair and len(pair) > 0:
                if isinstance(pair[0], list):
                    file_pairs = file_pairs + pair
                else:
                    file_pairs.append(pair)
        if self.sample_ratio < 1 and self.sample_ratio > 0:
            file_pairs = random.sample(file_pairs, int(len(file_pairs) * self.sample_ratio))
        return file_pairs
    
    def get_file_pair(self, file):
        raise NotImplementedError
    
    def load(self, index): 
        pairs = self.file_pairs[index]
        return self.load_image(pairs)
    
    def load_image(self, pairs):
        # 读取所有图像 (PIL -> numpy)
        images = []
        for p in pairs:
            if self.cache and p in self._cache:
                images.append(self._cache[p])
                continue
            with Image.open(p) as img:
                arr = np.array(img.convert("RGB"))
            images.append(arr)
            if self.cache:
                self._cache[p] = arr

        # images = [np.array(Image.open(p).convert("RGB")) for p in pairs]

        # Albumentations 同步变换
        transform = self.get_Atransforms() if not self.is_test() else self.get_Atransforms_test()
        if transform:
            # 构造输入字典
            data = {"image": images[0]}
            for i in range(1, len(images)):
                data[f"image{i}"] = images[i]
                transform.add_targets({f"image{i}": "image"})

            augmented = transform(**data)

            # 取出结果
            images = [augmented["image"]] + [augmented[f"image{i}"] for i in range(1, len(images))]

        # 转换为 tensor
        images = [torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0 for img in images]
        return images
    
    def get_Atransforms(self):
        '''
        Albumentations变换，返回None为不操作
        '''
        raise NotImplementedError
    
    def get_Atransforms_test(self):
        return None
    
    def __getitem__(self, index):
        return self.load(index)

    def shuffle_by_split(self):
        self.file_pairs = random.sample(self.file_pairs, len(self.file_pairs))
        
