import torch

def tensors_to_item(obj, str=None):
    """
    将输入对象中的所有torch.Tensor转换为对应的数值。
    支持字典、列表和张量的输入，返回与原结构一致的结果。
    """
    if isinstance(obj, dict):
        return {k: tensors_to_item(v, str) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensors_to_item(item, str) for item in obj]
    elif isinstance(obj, torch.Tensor):
        if str:
            return float(format(obj.item(), str))
            # return float(f"{obj.item():.3f}")
        return obj.item()
    else:
        return obj


def str_tensors_values(tensors, sep=',', format="[{key}: {value:.3f}]"):
    """
    根据传入的tensors类型（单个张量、张量字典或张量列表），打印相应的损失值。
    
    参数:
    tensors (torch.Tensor | dict[str, torch.Tensor] | list[torch.Tensor]): 损失数据。
    
    返回:
    str: 如果是单个张量，则返回格式化后的字符串表示；如果是字典或列表，则打印每个损失值，并返回None。
    """
    
    if isinstance(tensors, torch.Tensor):
        # 对于单个张量，直接返回格式化的损失值
        return f"{tensors.item():.3f}"
    
    elif isinstance(tensors, dict):
        # 对于字典，迭代并打印每个键值对
        output = []
        for key, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"字典中的所有值都应该是张量, 但'{key}'对应的值不是")
            # output.append(f"[{key}]: {value.item():.3f}")
            output.append(format.format(**{'key': key, 'value': value}))
        return sep.join(output)
        
    elif isinstance(tensors, list):
        # 对于列表，迭代并打印每个元素
        output = []
        for idx, value in enumerate(tensors, start=1):
            if not isinstance(value, torch.Tensor):
                raise ValueError("列表中的所有元素都应该为张量")
            # output.append(f"[{idx}]: {value.item():.3f}")
            output.append(format.format(**{'key': idx, 'value': value}))
        return sep.join(output)
        
    else:
        raise TypeError("输入必须是张量、张量字典或张量列表")


def apply_operation_on_tensors(v1, v2, operation):
    
    # 如果v1和v2都是dict，则递归地对每个key应用操作
    if isinstance(v1, dict) and isinstance(v2, dict):
        if v1.keys() == v2.keys():
            return {k: apply_operation_on_tensors(v1[k], v2[k], operation) for k in v1}
        else:
            raise ValueError("Keys of dictionaries do not match")
    
    # 如果v1是dict而v2不是，则对v1的每个值与v2应用操作
    elif isinstance(v1, dict):
        return {k: apply_operation_on_tensors(v1[k], v2, operation) for k in v1}
    
    # 如果v1和v2都是list，则递归地对每个元素应用操作
    elif isinstance(v1, list) and isinstance(v2, list):
        if len(v1) == len(v2):
            return [apply_operation_on_tensors(x, y, operation) for x, y in zip(v1, v2)]
        else:
            raise ValueError("Lists are not of the same length")
    
    # 如果v1是list而v2不是，则对v1的每个元素与v2应用操作
    elif isinstance(v1, list):
        return [apply_operation_on_tensors(x, v2, operation) for x in v1]
    
    # 如果v1和v2都不是dict或list，则直接应用操作
    else:
        if v2 is None:
            return operation(v1)
        return operation(v1, v2)


def to_devices(obj, device):
    """
    将输入的 tensor、model 或嵌套的结构（dict、list、tuple 等）中的所有 tensor 移动到指定的 device。
    
    Args:
        obj: 输入的 tensor、model 或嵌套的结构（dict、list、tuple 等）。
        device: 目标设备，可以是字符串（如 'cuda'、'cpu'）或 torch.device 对象。
    
    Returns:
        移动到目标设备后的对象。
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_devices(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_devices(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(to_devices(item, device) for item in obj)
    else:
        return obj

    


