import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DarkChannelPrior(nn.Module):
    def __init__(self, kernel_size=15):
        super(DarkChannelPrior, self).__init__()
        self.kernel_size = kernel_size
        # 创建卷积层，用于计算局部最小值
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                               padding=kernel_size // 2, bias=False)
        # 初始化卷积核为均匀分布
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / (kernel_size * kernel_size))  # 将卷积核初始化为均匀分布

    def forward(self, images):
        # 确保输入是 float32 类型
        # images = images.float() / 255.0  # 归一化到 [0, 1]

        # 获取颜色通道
        b, g, r = images[:, 0, :, :], images[:, 1, :, :], images[:, 2, :, :]

        # 计算暗通道
        dark_channel = torch.min(torch.min(b, g), r)  # shape: (batchsize, height, width)

        # 添加维度以进行卷积
        dark_channel_tensor = dark_channel.unsqueeze(1)  # shape: (batchsize, 1, height, width)

        # 使用卷积计算局部最小值
        dark_channel_local_min = self.conv(dark_channel_tensor)

        return dark_channel_local_min  # 返回形状为 (batchsize, height, width)
    
    

def dark_channel_numpy(image, size=15):
    # 转换为numpy数组并归一化
    image = np.array(image) / 255.0

    # 获取颜色通道
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # 计算暗通道
    dark_channel = np.min(np.array([b, g, r]), axis=0)

    # 转换为PyTorch张量并添加通道和批次维度
    dark_channel_tensor = torch.from_numpy(dark_channel).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)

    # 创建卷积核
    kernel = torch.ones((1, 1, size, size), dtype=torch.float32) / (size * size)

    # 使用卷积计算局部最小值
    dark_channel_local_min = F.conv2d(dark_channel_tensor, kernel, padding=size // 2)

    return dark_channel_local_min.squeeze().detach().numpy()
    
    
if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms
    import torchvision
    image = Image.open('d1_image.png')
    
    _transforms = transforms.Compose([
        transforms.ToTensor(), 
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    images = _transforms(image).unsqueeze(0).to('cuda')
    dark_channel_model = DarkChannelPrior(kernel_size=15).to('cuda')
    # 计算暗通道
    dark_channel_images = dark_channel_model(images)

    torchvision.utils.save_image(dark_channel_images[0], f'dark_channel_image_0.png')
        
    # numpy方法
    # 计算暗通道
    dark_channel_image = dark_channel_numpy(image)
    # 处理结果并可视化
    dark_channel_image = Image.fromarray((dark_channel_image * 255).astype(np.uint8))
    dark_channel_image.save('dark_channel_image.png')