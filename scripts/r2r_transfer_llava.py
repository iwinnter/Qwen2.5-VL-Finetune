import json
import os
from typing import List, Dict, Any

def convert_r2r_to_llava_full(
    annotations_file: str = "/home/liu/datasets/R2R/annotations.json", 
    train_dir: str = "/home/liu/datasets/R2R/train", 
    output_file: str = "/home/liu/datasets/R2R/annotations_llava.json"
) -> None:
    """
    将R2R数据集完整转换为LLaVA格式，保留所有帧
    
    Args:
        annotations_file: annotations.json文件路径
        train_dir: train文件夹路径  
        output_file: 输出的LLaVA格式json文件路径
    """
    
    print("=== R2R数据集完整转换为LLaVA格式 ===")
    print(f"输入文件: {annotations_file}")
    print(f"训练数据目录: {train_dir}")
    print(f"输出文件: {output_file}")
    print()
    
    # 读取annotations.json
    try:
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        print(f"✓ 成功读取annotations文件，包含 {len(annotations)} 个样本")
    except FileNotFoundError:
        print(f"✗ 错误: 找不到文件 {annotations_file}")
        return
    except json.JSONDecodeError:
        print(f"✗ 错误: {annotations_file} 不是有效的JSON文件")
        return
    
    # 检查train目录
    if not os.path.exists(train_dir):
        print(f"✗ 错误: 找不到训练数据目录 {train_dir}")
        return
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"✓ 创建输出目录: {output_dir}")
        except OSError as e:
            print(f"✗ 错误: 无法创建输出目录 {output_dir}: {e}")
            return

    llava_data = []
    stats = {
        'total_samples': len(annotations),
        'successful_conversions': 0,
        'missing_folders': 0,
        'missing_frames': 0,
        'empty_samples': 0,
        'total_frames_processed': 0
    }
    
    print("开始转换...")
    print("-" * 50)
    
    for idx, item in enumerate(annotations):
        video_id = item['video_id']
        question = item['q']
        answer = item['a']
        frames = item['frames']
        
        # 提取视频文件夹ID（如 "1648-5" -> "1648"）
        video_folder = video_id.split('-')[0]
        folder_path = os.path.join(train_dir, video_folder)
        
        # 检查对应的文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹 {folder_path} 不存在，跳过样本 {video_id}")
            stats['missing_folders'] += 1
            continue
        
        # 构建图片路径列表
        image_paths = []
        missing_frames = []
        
        for frame in frames:
            # 提取帧文件名（如 "1648/frame_0.jpg" -> "frame_0.jpg"）
            frame_filename = os.path.basename(frame)
            frame_path = os.path.join(folder_path, frame_filename)
            
            if os.path.exists(frame_path):
                # 使用相对路径，格式为 "folder/frame_x.jpg"
                relative_path = os.path.join(video_folder, frame_filename)
                image_paths.append(relative_path)
                stats['total_frames_processed'] += 1
            else:
                missing_frames.append(frame_filename)
        
        # 记录缺失的帧
        if missing_frames:
            stats['missing_frames'] += len(missing_frames)
            if len(missing_frames) <= 5:
                print(f"警告: 样本 {video_id} 缺少 {len(missing_frames)} 帧: {missing_frames}")
            else:
                print(f"警告: 样本 {video_id} 缺少 {len(missing_frames)} 帧: {missing_frames[:5]}...")
        
        # 如果没有找到任何有效图片，跳过这个样本
        if not image_paths:
            print(f"错误: 样本 {video_id} 没有找到任何有效图片，跳过")
            stats['empty_samples'] += 1
            continue
        
        # 构建LLaVA格式的对话
        # 为每个图片添加<image>标记
        image_tokens = "".join(["<image>\n" for _ in image_paths])
        
        # 构建人类消息
        human_message = f"{image_tokens}Based on the sequence of {len(image_paths)} images showing navigation through indoor environments, follow this instruction: \"{question}\" What should be the next action?"
        
        # 创建LLaVA格式的数据项
        llava_item = {
            "id": video_id,
            "image": image_paths, # 保持为列表
            "conversations": [
                {
                    "from": "human", 
                    "value": human_message
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        
        llava_data.append(llava_item)
        stats['successful_conversions'] += 1
        
        # 显示进度
        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{len(annotations)} 个样本，成功转换 {stats['successful_conversions']} 个")
    
    # 保存转换结果
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(llava_data, f, ensure_ascii=False, indent=2)
        print(f"✓ 成功保存到 {output_file}")
    except Exception as e:
        print(f"✗ 保存文件时出错: {e}")
        return
    
    # 显示转换统计信息
    print("\n" + "=" * 50)
    print("转换完成！统计信息:")
    print(f"原始样本总数: {stats['total_samples']}")
    print(f"成功转换样本数: {stats['successful_conversions']}")
    print(f"缺失文件夹的样本: {stats['missing_folders']}")
    print(f"没有有效图片的样本: {stats['empty_samples']}")
    print(f"缺失帧总数: {stats['missing_frames']}")
    print(f"成功处理的帧总数: {stats['total_frames_processed']}")
    print(f"转换成功率: {stats['successful_conversions']/stats['total_samples']*100:.1f}%")
    
    if stats['successful_conversions'] > 0:
        avg_frames = stats['total_frames_processed'] / stats['successful_conversions']
        print(f"平均每个样本帧数: {avg_frames:.1f}")
    
    print(f"输出文件: {output_file}")
    print("=" * 50)

def verify_conversion(output_file: str, num_samples: int = 3) -> None:
    """
    验证转换结果，显示前几个样本
    """
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n=== 转换结果验证（显示前{num_samples}个样本）===")
        for i, sample in enumerate(data[:num_samples]):
            print(f"\n样本 {i+1}:")
            print(f"  ID: {sample['id']}")
            print(f"  图片数量: {len(sample['image'])}")
            print(f"  图片路径示例: {sample['image'][:3]}...") # 显示前3个路径
            # 限制显示的消息长度以避免输出过长
            human_msg = sample['conversations'][0]['value']
            gpt_msg = sample['conversations'][1]['value']
            print(f"  人类消息长度: {len(human_msg)} 字符")
            print(f"  人类消息预览: {human_msg[:100]}...")
            print(f"  AI回答长度: {len(gpt_msg)} 字符")
            print(f"  AI回答预览: {gpt_msg[:100]}...")
    except Exception as e:
        print(f"验证时出错: {e}")

if __name__ == "__main__":
    # 执行完整转换
    convert_r2r_to_llava_full()
    
    # 验证转换结果
    verify_conversion("/home/liu/datasets/R2R_llava/r2r_llava_full.json")
