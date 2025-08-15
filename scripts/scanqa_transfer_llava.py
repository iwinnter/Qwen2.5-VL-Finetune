import json
import os
from typing import List, Dict, Any

def convert_scanqa_to_llava_video(
    annotations_file: str = "/home/liu/datasets/ScanQA/annotations.json", 
    video_dir: str = "/home/liu/datasets/ScanQA/videos", 
    output_file: str = "/home/liu/datasets/ScanQA/annotations_llava_scanqa.json"
) -> None:
    """
    将ScanQA数据集转换为LLaVA视频格式
    
    Args:
        annotations_file: ScanQA annotations.json文件路径
        video_dir: 视频文件夹路径  
        output_file: 输出的LLaVA格式json文件路径
    """
    
    print("=== ScanQA数据集转换为LLaVA视频格式 ===")
    print(f"输入文件: {annotations_file}")
    print(f"视频数据目录: {video_dir}")
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
    
    # 检查视频目录
    if not os.path.exists(video_dir):
        print(f"✗ 错误: 找不到视频数据目录 {video_dir}")
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
        'missing_videos': 0,
        'invalid_samples': 0
    }
    
    print("开始转换...")
    print("-" * 50)
    
    for idx, item in enumerate(annotations):
        # 检查必要字段
        if not all(key in item for key in ['question_id', 'q', 'a', 'video_id']):
            print(f"警告: 样本 {idx} 缺少必要字段，跳过")
            stats['invalid_samples'] += 1
            continue
            
        question_id = item['question_id']
        question = item['q']  # ScanQA使用'q'而不是'question'
        video_id = item['video_id']  # ScanQA有video_id字段
        
        # 处理答案 - ScanQA的答案字段为'a'，取第一个答案
        answers = item['a']
        if isinstance(answers, list) and len(answers) > 0:
            answer = str(answers[0])
        elif isinstance(answers, str) and answers.strip():
            answer = answers
        else:
            # 理论上不会到这里，但保险起见
            answer = "I cannot provide a specific answer to this question."
        
        # 构建视频文件路径，使用video_id字段
        video_filename = f"{video_id}.mp4"
        
        video_path = os.path.join(video_dir, video_filename)
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            # 尝试其他可能的文件扩展名
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            video_found = False
            base_name = os.path.splitext(video_filename)[0]
            
            for ext in video_extensions:
                alt_video_path = os.path.join(video_dir, base_name + ext)
                if os.path.exists(alt_video_path):
                    video_filename = base_name + ext
                    video_found = True
                    break
            
            if not video_found:
                print(f"警告: 视频文件 {video_filename} 不存在，跳过样本 {question_id}")
                stats['missing_videos'] += 1
                continue
        
        # 构建LLaVA格式的对话
        # 为视频添加<video>标记
        human_message = f"<video>\n{question}"
        
        # 创建LLaVA格式的数据项
        llava_item = {
            "id": str(question_id),
            "video": video_filename,  # 相对路径
            "conversations": [
                {
                    "from": "human", 
                    "value": f"<video>\n{question}"
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        
        # 如果有额外的元数据，可以添加
        if 'video_id' in item:
            llava_item['video_id'] = item['video_id']
        if 'object_ids' in item and item['object_ids']:
            llava_item['object_ids'] = item['object_ids']
        if 'object_names' in item and item['object_names']:
            llava_item['object_names'] = item['object_names']
        
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
    print(f"缺失视频文件的样本: {stats['missing_videos']}")
    print(f"无效样本数: {stats['invalid_samples']}")
    print(f"转换成功率: {stats['successful_conversions']/stats['total_samples']*100:.1f}%")
    
    print(f"输出文件: {output_file}")
    print("=" * 50)

def convert_scanqa_with_custom_mapping(
    annotations_file: str,
    video_dir: str,
    output_file: str,
    video_mapping_function = None
) -> None:
    """
    使用自定义视频文件映射函数转换ScanQA数据集
    
    Args:
        annotations_file: ScanQA annotations.json文件路径
        video_dir: 视频文件夹路径
        output_file: 输出文件路径
        video_mapping_function: 自定义的视频文件名映射函数
    """
    
    def default_video_mapping(item):
        """默认的视频文件名映射函数"""
        if 'scene_id' in item:
            return f"{item['scene_id']}.mp4"
        elif 'question_id' in item:
            return f"{item['question_id']}.mp4"
        else:
            return None
    
    if video_mapping_function is None:
        video_mapping_function = default_video_mapping
    
    print("=== ScanQA数据集转换为LLaVA视频格式（自定义映射）===")
    print(f"输入文件: {annotations_file}")
    print(f"视频数据目录: {video_dir}")
    print(f"输出文件: {output_file}")
    print()
    
    # 读取annotations
    try:
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        print(f"✓ 成功读取annotations文件，包含 {len(annotations)} 个样本")
    except Exception as e:
        print(f"✗ 读取文件错误: {e}")
        return
    
    # 获取视频目录中的所有视频文件
    video_files = set()
    if os.path.exists(video_dir):
        for file in os.listdir(video_dir):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                video_files.add(file)
        print(f"✓ 找到 {len(video_files)} 个视频文件")
    else:
        print(f"✗ 视频目录不存在: {video_dir}")
        return
    
    llava_data = []
    stats = {
        'total_samples': len(annotations),
        'successful_conversions': 0,
        'missing_videos': 0,
        'invalid_samples': 0
    }
    
    for idx, item in enumerate(annotations):
        try:
            question_id = item.get('question_id', f"sample_{idx}")
            question = item.get('question', '')
            
            if not question:
                stats['invalid_samples'] += 1
                continue
            
            # 使用自定义映射函数获取视频文件名
            video_filename = video_mapping_function(item)
            if not video_filename:
                stats['invalid_samples'] += 1
                continue
            
            # 检查视频文件是否存在
            if video_filename not in video_files:
                # 尝试其他扩展名
                base_name = os.path.splitext(video_filename)[0]
                found = False
                for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                    alt_name = base_name + ext
                    if alt_name in video_files:
                        video_filename = alt_name
                        found = True
                        break
                
                if not found:
                    stats['missing_videos'] += 1
                    continue
            
            # 处理答案
            answers = item.get('answers', [])
            if isinstance(answers, list) and len(answers) > 0:
                if isinstance(answers[0], dict) and 'answer' in answers[0]:
                    answer = answers[0]['answer']
                else:
                    answer = str(answers[0])
            else:
                answer = "I don't know."
            
            # 创建LLaVA格式数据
            llava_item = {
                "id": str(question_id),
                "video": video_filename,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<video>\n{question}"
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ]
            }
            
            llava_data.append(llava_item)
            stats['successful_conversions'] += 1
            
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {e}")
            stats['invalid_samples'] += 1
        
        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{len(annotations)} 个样本")
    
    # 保存结果
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(llava_data, f, ensure_ascii=False, indent=2)
        print(f"✓ 成功保存到 {output_file}")
    except Exception as e:
        print(f"✗ 保存文件时出错: {e}")
        return
    
    # 统计信息
    print("\n" + "=" * 50)
    print("转换完成！统计信息:")
    print(f"原始样本总数: {stats['total_samples']}")
    print(f"成功转换样本数: {stats['successful_conversions']}")
    print(f"缺失视频文件的样本: {stats['missing_videos']}")
    print(f"无效样本数: {stats['invalid_samples']}")
    print(f"转换成功率: {stats['successful_conversions']/stats['total_samples']*100:.1f}%")

def verify_scanqa_conversion(output_file: str, num_samples: int = 3) -> None:
    """
    验证ScanQA转换结果，显示前几个样本
    """
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n=== ScanQA转换结果验证（显示前{num_samples}个样本）===")
        for i, sample in enumerate(data[:num_samples]):
            print(f"\n样本 {i+1}:")
            print(f"  ID: {sample['id']}")
            print(f"  视频文件: {sample['video']}")
            
            human_msg = sample['conversations'][0]['value']
            gpt_msg = sample['conversations'][1]['value']
            print(f"  人类问题: {human_msg}")
            print(f"  AI回答: {gpt_msg}")
            
            # 显示额外的元数据（如果有）
            for key in sample:
                if key not in ['id', 'video', 'conversations']:
                    print(f"  {key}: {sample[key]}")
                    
        print(f"\n总计转换了 {len(data)} 个样本")
        
    except Exception as e:
        print(f"验证时出错: {e}")

if __name__ == "__main__":
    # 基本转换
    convert_scanqa_to_llava_video(
        annotations_file="/home/liu/datasets/ScanQA/ScanQA_merged.json",
        video_dir="/home/liu/datasets/ScanQA/videos",
        output_file="/home/liu/datasets/ScanQA/annotations_llava_scanqa.json"
    )

    
    # 示例：使用自定义视频文件映射
    def custom_video_mapping(item):
        """自定义视频文件映射示例"""
        # 根据您的实际数据结构调整
        if 'video_id' in item:
            return f"{item['video_id']}.mp4"
        return None
    