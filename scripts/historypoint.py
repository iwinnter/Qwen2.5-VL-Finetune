import json
import random
import argparse
import re  # 添加正则表达式模块
from pathlib import Path

def process_r2r_annotations(input_file, output_file, num_random_images):
    """
    处理R2R数据集的注释文件，保留最后一张图片和随机选择的前面N张图片
    
    参数:
    - input_file: 输入JSON文件路径
    - output_file: 输出JSON文件路径
    - num_random_images: 要随机选择的图片数量（不包括最后一张）
    """
    
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_count = 0
    skipped_count = 0
    
    # 处理每个条目
    for item in data:
        if 'image' in item and isinstance(item['image'], list):
            original_images = item['image']
            original_count = len(original_images)
            
            # 总共要保留的图片数量 = 随机选择的数量 + 1（最后一张）
            total_to_keep = num_random_images + 1
            
            if original_count <= total_to_keep:
                # 如果原始图片数量小于等于要保留的数量，保留所有图片
                skipped_count += 1
                print(f"跳过ID {item.get('id', 'unknown')}: 原始图片数量({original_count})少于或等于要保留的数量({total_to_keep})")
                continue
            
            # 获取最后一张图片
            last_image = original_images[-1]
            
            # 从除了最后一张之外的图片中随机选择
            other_images = original_images[:-1]
            selected_images = random.sample(other_images, num_random_images)
            
            # 按原始顺序排序选中的图片
            selected_images.sort(key=lambda x: original_images.index(x))
            
            # 添加最后一张图片
            selected_images.append(last_image)
            
            # 更新图片列表
            item['image'] = selected_images
            
            # 更新对话中的图片标签数量和图像序列描述
            if 'conversations' in item:
                for conversation in item['conversations']:
                    if conversation.get('from') == 'human' and conversation.get('value'):
                        # 生成新的图片标签
                        new_image_tags = '<image>\n' * len(selected_images)
                        
                        # 找到最后一个<image>标签的位置
                        last_tag_end = conversation['value'].rfind('<image>') + 7
                        
                        # 如果找到了<image>标签
                        if last_tag_end > 6:
                            # 获取<image>标签之后的文本内容
                            text_after_images = conversation['value'][last_tag_end:].lstrip('\n')
                            
                            # 重新构建value
                            conversation['value'] = new_image_tags + text_after_images
                        
                        # 修复：更新"Based on the sequence of X images"中的X
                        # 使用正则表达式查找并替换图像数量
                        pattern = r'Based on the sequence of (\d+) images'
                        if re.search(pattern, conversation['value']):
                            # 替换为实际保留的图像数量
                            conversation['value'] = re.sub(
                                pattern, 
                                f'Based on the sequence of {len(selected_images)} images', 
                                conversation['value']
                            )
            
            processed_count += 1
            print(f"处理ID {item.get('id', 'unknown')}: {original_count} -> {len(selected_images)} 张图片")
    
    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成！")
    print(f"- 处理了 {processed_count} 个条目")
    print(f"- 跳过了 {skipped_count} 个条目（图片数量不足）")
    print(f"- 结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='处理R2R数据集注释文件，保留最后一张图片和随机选择的前面N张图片'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='annotations_llava_r2r.json',
        help='输入JSON文件路径 (默认: annotations_llava_r2r.json)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='annotations_llava_r2r_filtered.json',
        help='输出JSON文件路径 (默认: annotations_llava_r2r_filtered.json)'
    )
    
    parser.add_argument(
        '--num-random', '-n',
        type=int,
        default=8,
        help='要随机选择的图片数量，不包括最后一张 (默认: 8)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='随机种子，用于可重复的随机选择 (默认: None)'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.input).exists():
        print(f"错误: 输入文件 '{args.input}' 不存在！")
        return
    
    # 设置随机种子（如果提供）
    if args.seed is not None:
        random.seed(args.seed)
        print(f"使用随机种子: {args.seed}")
    
    print(f"配置:")
    print(f"- 输入文件: {args.input}")
    print(f"- 输出文件: {args.output}")
    print(f"- 随机选择图片数量: {args.num_random}")
    print(f"- 总共保留图片数量: {args.num_random + 1} (包括最后一张)")
    print(f"\n开始处理...\n")
    
    # 处理文件
    process_r2r_annotations(args.input, args.output, args.num_random)

if __name__ == "__main__":
    main()