import torch


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
        return operation(v1, v2)

# 示例用法：
if __name__ == "__main__":
    # 测试用例
    tensor1 = torch.tensor([1.0, 2.0])
    tensor2 = torch.tensor([3.0, 4.0])
    scalar = 5.0
    
    # Dict + Dict
    print(apply_operation_on_tensors({'a': tensor1}, {'a': tensor2}, torch.add))
    
    # List + Scalar
    print(apply_operation_on_tensors([tensor1, tensor2], scalar, torch.add))
    
    # Tensor + Scalar
    print(apply_operation_on_tensors(tensor1, scalar, torch.add))


