import torch
import os
from datetime import datetime

def read_file_to_list(file_path):
    all_lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # 添加每一行到all_lines列表中，并去除每行末尾的换行符
            all_lines.extend(line.strip() for line in lines)
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
    return all_lines


def save_model(models: dict, key: str, save_dir: str) -> dict:
    """
    保存PyTorch模型到磁盘。

    参数:
        models (dict[str, torch.nn.Module]): 包含要保存的模型的字典，键为字符串，值为模型对象。
        key (str): 文件名中使用的key部分。如果为空，则不添加到文件名中。
        save_dir (str): 模型保存的目录路径。

    返回:
        dict: 包含所有保存文件路径的字典，键为模型名，值为保存的文件路径。
    """
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    saved_paths = {}
    for model_name, model in models.items():
        # 根据key是否为空来构建文件名
        if key:
            filename = f"{key}_{model_name}.pth"
        else:
            filename = f"{model_name}.pth"
        
        # 构建完整的文件路径
        file_path = os.path.join(save_dir, filename)
        
        # 保存模型状态字典
        torch.save(model.state_dict(), file_path)
        
        # 记录保存的文件路径
        saved_paths[model_name] = file_path

    return saved_paths


def load_model(models: dict, key: str, save_dir: str, mode='eval') -> dict:
    """
    从磁盘加载PyTorch模型。
    
    参数:
        models (dict[str, torch.nn.Module]): 包含模型类的字典，键为字符串，值为未加载状态的模型对象。
        key (str): 文件名中使用的key部分。如果为空，则不添加到文件名中。
        save_dir (str): 模型保存的目录路径。
    
    返回:
        dict: 包含所有加载后的模型对象的字典，键为模型名，值为加载后的模型对象。
    """
    loaded_models = {}
    
    for model_name, model in models.items():
        # 根据key是否为空来构建文件名
        if key:
            filename = f"{key}_{model_name}.pth"
        else:
            filename = f"{model_name}.pth"
        
        # 构建完整的文件路径
        file_path = os.path.join(save_dir, filename)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"警告：文件 {file_path} 不存在，跳过加载。")
            continue
        
        # 加载模型状态字典
        model.load_state_dict(torch.load(file_path))
        if mode == 'eval':
            model.eval()  # 设置为评估模式
        
        # 记录加载后的模型
        loaded_models[model_name] = model
    
    return loaded_models

