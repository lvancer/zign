import torch
from zign.data import zDataset
import torchvision.transforms as transforms

class RandomNoiseDataset(zDataset):
    def __init__(self, num_samples, img_height, img_width, channels=3, transform=None):
        """
        随机噪音图片数据集
        
        Args:
            num_samples (int): 数据集中的样本数量
            img_height (int): 图片的高度
            img_width (int): 图片的宽度
            channels (int, optional): 图片的通道数，默认为3（RGB）
            transform (callable, optional): 数据预处理变换，默认为None
        """
        self.num_samples = num_samples
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.transform = transform
        
    def __len__(self):
        """返回数据集的总样本数"""
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        根据索引获取样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: 噪音图片
        """
        # 生成随机噪音图片，范围在[0, 1]之间
        noise = torch.randn(self.channels, self.img_height, self.img_width)
        # noise = (noise - noise.min()) / (noise.max() - noise.min())  # 归一化到[0,1]
        
        if self.transform:
            noise = self.transform(noise)
            
        return noise



class RandomNoiseWithLabelDataset(RandomNoiseDataset):
    def __init__(self, num_samples, img_height, img_width, channels=3, num_classes=2, transform=None):
        """
        带随机分类标签的随机噪音图片数据集

        Args:
            num_samples (int): 数据集中的样本数量
            img_height (int): 图片的高度
            img_width (int): 图片的宽度
            channels (int, optional): 图片的通道数，默认为3（RGB）
            num_classes (int, optional): 分类任务的类别数量，默认为2
            transform (callable, optional): 数据预处理变换，默认为None
        """
        super().__init__(num_samples, img_height, img_width, channels, transform)
        self.num_classes = num_classes

    def __getitem__(self, idx):
        """
        根据索引获取样本和标签

        Args:
            idx (int): 样本索引

        Returns:
            tuple: 噪音图片和随机分类标签
        """
        # 生成随机噪音图片，范围在[0, 1]之间
        noise = torch.randn(self.channels, self.img_height, self.img_width)
        # noise = (noise - noise.min()) / (noise.max() - noise.min()) # 归一化到[0,1]

        if self.transform:
            noise = self.transform(noise)

        # 生成随机分类标签
        label = torch.randint(0, self.num_classes, ())

        return noise, label
