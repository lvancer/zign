import torch

def denormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], to_uint8=False, unsqueeze_channel=False):
    """
    反归一化图像 tensor。
    
    Args:
        tensor (torch.Tensor): 归一化后的图像 tensor，形状为 [C, H, W] 或 [B, C, H, W]。
        mean (list or tuple, optional): 归一化时使用的均值，通常为 [mean_r, mean_g, mean_b]。
        std (list or tuple, optional): 归一化时使用的标准差，通常为 [std_r, std_g, std_b]。
        to_uint8 (bool, optional): 是否将结果转换为 uint8 类型（0-255 范围）。默认为 True。
        unsqueeze_channel (bool, optional): 如果输入 tensor 是单通道，是否将其扩展为 [C, H, W] 格式。默认为 False。
    
    Returns:
        torch.Tensor: 反归一化后的图像 tensor，形状为 [C, H, W] 或 [B, C, H, W]。
    """

    
    # 将 mean 和 std 转换为与 tensor 同设备的 tensor
    mean = torch.tensor(mean, device=tensor.device).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(std, device=tensor.device).unsqueeze(-1).unsqueeze(-1)
    
    # 反归一化：x = x * std + mean
    denormalized = tensor * std + mean
    
    # 如果需要，将结果限制在 0-255 范围并转换为 uint8 类型
    if to_uint8:
        denormalized = (denormalized * 255).clamp(0, 255).to(torch.uint8)
    
    # 如果输入是单通道，扩展为 [C, H, W] 格式
    if unsqueeze_channel and denormalized.ndim == 3:
        denormalized = denormalized.unsqueeze(0)
    
    return denormalized