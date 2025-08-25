#!/usr/bin/env python3
"""
合并ScanQA数据集的四个JSON文件
"""

import json
import os
from pathlib import Path

def merge_json_files(file_paths, output_path):
    """
    合并多个JSON文件
    
    Args:
        file_paths: 要合并的JSON文件路径列表
        output_path: 输出文件路径
    """
    merged_data = []
    total_count = 0
    
    print("开始合并JSON文件...")
    print("=" * 60)
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"警告: 文件不存在，跳过: {file_path}")
            continue
        
        print(f"正在处理: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 确保data是列表
            if not isinstance(data, list):
                print(f"警告: {file_path} 不是列表格式，尝试转换...")
                if isinstance(data, dict):
                    data = [data]
                else:
                    print(f"错误: 无法处理 {file_path} 的数据格式")
                    continue
            
            merged_data.extend(data)
            print(f"  - 添加了 {len(data)} 条数据")
            total_count += len(data)
            
        except json.JSONDecodeError as e:
            print(f"错误: 无法解析JSON文件 {file_path}: {e}")
        except Exception as e:
            print(f"错误: 处理文件 {file_path} 时出错: {e}")
    
    # 保存合并后的数据
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60)
        print(f"合并完成！")
        print(f"总共合并了 {total_count} 条数据")
        print(f"输出文件: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"错误: 保存合并文件时出错: {e}")
        return False

def verify_merged_file(file_path):
    """
    验证合并后的文件
    """
    print("\n" + "=" * 60)
    print("验证合并后的文件...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"文件加载成功: {file_path}")
        print(f"总数据条数: {len(data)}")
        
        # 显示第一条数据的结构
        if data and isinstance(data[0], dict):
            print(f"第一条数据的键: {list(data[0].keys())}")
            
            # 显示示例数据
            sample = data[0].copy()
            # 如果某些字段太长，截断显示
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 100:
                    sample[key] = value[:100] + "..."
                elif isinstance(value, list) and len(value) > 5:
                    sample[key] = value[:5] + ["..."]
            
            print("第一条数据示例:")
            print(json.dumps(sample, indent=2, ensure_ascii=False))
        
        return True
        
    except Exception as e:
        print(f"验证文件时出错: {e}")
        return False

def main():
    """
    主函数
    """
    # 定义要合并的文件列表
    base_dir = "/home/liu/dataset"
    file_list = [
        "testallrxr/annotations_llava_rxr_modified24.json",
        "All_test/annotations_r2r_24point_test.json",

    ]
    
    # 构建完整路径
    file_paths = [os.path.join(base_dir, filename) for filename in file_list]
    output_path = os.path.join(base_dir, "annotations_alltest_24point.json")
    
    print("ScanQA JSON文件合并工具")
    print("=" * 60)
    print("要合并的文件:")
    for i, path in enumerate(file_paths, 1):
        print(f"  {i}. {Path(path).name}")
    print(f"输出文件: {Path(output_path).name}")
    
    # 执行合并
    success = merge_json_files(file_paths, output_path)
    
    if success:
        # 验证合并结果
        verify_merged_file(output_path)
    else:
        print("合并失败！")

def merge_without_split_info():
    """
    简化版合并函数（保留作为备用）
    """
    pass

if __name__ == "__main__":
    # 运行主合并程序（简单合并，不添加split信息）
    main()