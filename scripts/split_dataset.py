import json
import os
import random
import math

def split_dataset(input_file, output_dir, num_parts=100000):
    """
    将JSON数据集随机分割成指定数量的部分
    
    参数:
    input_file: 输入的JSON文件路径
    output_dir: 输出目录路径
    num_parts: 要分割的部分数量
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取JSON数据
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"成功加载 {len(data)} 个样本 from {input_file}")
    
    # 随机打乱数据（固定种子确保可重复性）
    random.seed(42)
    random.shuffle(data)
    
    # 计算每份数据的大小
    total_samples = len(data)
    samples_per_part = total_samples // num_parts
    remainder = total_samples % num_parts
    
    print(f"总样本数: {total_samples}")
    print(f"每份基础样本数: {samples_per_part}")
    print(f"余数: {remainder} (前{remainder}份将多包含1个样本)")
    
    # 分割并保存数据
    start_idx = 0
    for i in range(num_parts):
        # 计算当前部分的结束索引
        end_idx = start_idx + samples_per_part + (1 if i < remainder else 0)
        
        # 获取当前部分的数据
        part_data = data[start_idx:end_idx]
        
        # 保存为JSON文件
        output_file = os.path.join(output_dir, f"{i+1}.json")
        with open(output_file, 'w') as f:
            json.dump(part_data, f, indent=2)
        
        print(f"第 {i+1} 部分: {len(part_data)} 个样本 保存到 {output_file}")
        
        # 更新起始索引
        start_idx = end_idx
    
    print(f"\n数据集已成功分割为 {num_parts} 个部分，保存在 {output_dir}")

if __name__ == "__main__":
    # 配置路径
    data_dir = "/home/liu/dataset/R2R"
    input_file = os.path.join(data_dir, "annotations_r2r_8point.json")
    output_dir = os.path.join(data_dir, "json")
    
    # 执行分割
    split_dataset(input_file, output_dir, num_parts=100000)