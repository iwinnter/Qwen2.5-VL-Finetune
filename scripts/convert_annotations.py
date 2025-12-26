#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将annotations.json转换为新格式：
- video_id -> id
- 保留frames中的第一张、中间一张、最后一张图片 -> image (数组)
- q和a转换为新的conversations格式
"""

import json

# ========== 请修改为您的实际文件路径 ==========
input_path = "/home/liu/2/annotations.json"                  # 输入文件路径
output_path = "/home/liu/2/annotations_converted_v2.json"    # 输出文件路径
# =============================================

def get_three_frames(frames):
    """获取第一张、中间一张、最后一张图片"""
    if not frames:
        return []
    
    if len(frames) == 1:
        return [frames[0], frames[0], frames[0]]
    
    if len(frames) == 2:
        return [frames[0], frames[0], frames[1]]
    
    first = frames[0]
    last = frames[-1]
    # 中间一张：取中间索引
    mid_index = len(frames) // 2
    middle = frames[mid_index]
    
    return [first, middle, last]

def convert_annotations(input_path, output_path):
    # 读取原始JSON文件
    print(f"正在读取文件: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件未找到 - {input_path}")
        return
    except json.JSONDecodeError as e:
        print(f"错误: JSON解析失败 - {e}")
        return
    
    print(f"成功读取 {len(data)} 条记录")
    
    # 转换数据
    converted_data = []
    
    for item in data:
        video_id = item.get("video_id", "")
        question = item.get("q", "")
        answer = item.get("a", "")
        frames = item.get("frames", [])
        
        # 获取三张图片：第一张、中间一张、最后一张
        three_frames = get_three_frames(frames)
        
        # 构建新的human value
        human_value = (
            "Based on the sequence of historical observations: <image> <image> \n"
            "and current observation: <image>\n"
            "Your navigation assigned task is:\n"
            f"{question}\n"
            "What should be the next action?"
        )
        
        # 构建新格式
        new_item = {
            "id": video_id,
            "image": three_frames,
            "conversations": [
                {
                    "from": "human",
                    "value": human_value
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        
        converted_data.append(new_item)
    
    # 保存转换后的JSON文件
    print(f"正在保存到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成！共转换 {len(converted_data)} 条记录")
    
    # 显示转换示例
    if converted_data:
        print("\n" + "=" * 60)
        print("转换示例 (第一条记录):")
        print("=" * 60)
        print(json.dumps(converted_data[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    convert_annotations(input_path, output_path)
