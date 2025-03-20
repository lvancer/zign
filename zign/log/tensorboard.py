import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from zign.config import zConfig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
from zign.utils import to, io
import torchvision.utils as vutis



class zSummaryWriter(SummaryWriter):
    def __init__(self, config: zConfig, trainer=None):
        """
        初始化训练日志记录器。
        
        Args:
            log_dir (str, optional): 日志文件保存目录。如果为 None，将自动生成一个基于当前时间的目录。
        """
        self._current = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(config.run_dir, config.mode, self._current)
        super().__init__(log_dir=log_dir)
        
    def folder(self):
        return self._current
    
    def add_losses(self, main_tag, losses, step):
        """
        记录损失值到 TensorBoard。
        """
        for key, value in losses.items():
            if isinstance(value, dict):
                self.add_scalars(f"{main_tag}/{key}", value, step)
            else:
                self.add_scalar(f"{main_tag}/{key}", value, step)
    

    def add_graphs(self, models_dict, input_dict, device):
        """
        将多个模型的计算图添加到 TensorBoard 中。
        Args:
            models_dict (dict): 模型字典，键为模型名称，值为模型实例。
            input_dict (dict): 输入内容字典，键为模型名称，值为输入内容。
                - 输入内容可以是单个张量、标量，也可以是包含多个输入的列表或元组。
        """
        # 遍历所有输入内容和对应的模型名称
        for model_name, inputs in input_dict.items():
            # 检查当前模型是否存在
            if model_name in models_dict:
                model = models_dict[model_name]
                if inputs is not None:
                    # 判断输入是否为多个参数（列表或元组）
                    if isinstance(inputs, (list, tuple)):
                        # 逐个处理输入中的每个元素
                        dummy_inputs = []
                        for inp in inputs:
                            if isinstance(inp, torch.Tensor):
                                # 如果是张量，直接使用
                                dummy_inputs.append(inp)
                            else:
                                # 如果是标量，直接使用
                                dummy_inputs.append(inp)
                        # 将输入包装为列表或元组，与 inputs 保持一致
                        if isinstance(inputs, tuple):
                            dummy_inputs = tuple(dummy_inputs)
                    else:
                        # 单个输入的情况，可能是张量或标量
                        dummy_inputs = inputs
                        
                    # 将输入内容移动到指定设备
                    dummy_inputs = to.to_devices(dummy_inputs, device)
                    
                    # 添加计算图到 TensorBoard
                    self.add_graph(model, dummy_inputs)
                    print(f"Successfully added graph for {model_name}")

    def add_images_with_labels(self, tag, images, labels, step, figsize=(10, 5)):
        """
        将多张图片和对应的标签绘制在一个 Figure 中，并添加到 TensorBoard 中
        
        参数:
            tag:
            images: 形状为 (B, C, H, W) 的 Tensor 或 numpy 数组，其中 B 是批量大小，C 是通道数，H 是高度，W 是宽度
            labels: 形状为 (B,) 的 Tensor 或 numpy 数组，包含每张图片的标签
            step: 当前训练步数或 epoch
        """
        
        if isinstance(images, torch.Tensor):
            if len(images.shape) == 3:  # 单张图片
                images = [images]
            else:
                images = [i.squeeze(0) for i in images.split(1, dim=0)]
        
        # # 如果是 numpy 数组，转换为 Tensor
        # if isinstance(images, np.ndarray):
        #     images = torch.from_numpy(images)
        # if isinstance(labels, np.ndarray):
        #     labels = torch.from_numpy(labels)
        # 确保图片的形状是 (B, C, H, W)
        # images = images.cpu() # 将图片移动到 CPU
        
        plt.figure(figsize=figsize)
        
        # 循环遍历每张图片和标签
        for idx, (img, label) in enumerate(zip(images, labels)):
            # 创建子图
            grid = vutis.make_grid(img)
            # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            plt.subplot(1, len(images), idx + 1)
            plt.imshow(ndarr, norm=colors.NoNorm())  # 将图片从 (C, H, W) 转换为 (H, W, C)
            plt.title(f'{label}')  # 将标签显示在图片标题中
            plt.axis('off')  # 关闭坐标轴
        
        # 添加 Figure 到 TensorBoard
        self.add_figure(tag, plt.gcf(), step)
        
        # 关闭当前 Figure
        plt.close()



# class NoNorm(colors.Normalize):
#     """
#     Dummy replacement for `Normalize`, for the case where we want to use
#     indices directly in a `~matplotlib.cm.ScalarMappable`.
#     """
#     def __call__(self, value, clip=None):
#         return value
