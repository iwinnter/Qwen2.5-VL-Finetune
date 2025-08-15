import json
import os
import shutil
from pathlib import Path

def modify_json_paths(json_file_path, output_file_path):
    """
    修改JSON文件中的ID和图片路径，在文件夹名前添加"00"前缀
    """
    print(f"正在读取JSON文件: {json_file_path}")
    
    # 读取原始JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 修改每个条目中的ID和图片路径
    modified_ids = 0
    for item in data:
        # 修改ID字段
        if 'id' in item:
            original_id = item['id']
            # 分割ID，通常格式为 "数字-数字"
            id_parts = original_id.split('-')
            if len(id_parts) >= 1 and id_parts[0].isdigit():
                # 在第一个数字前添加"00"前缀
                new_first_part = "00" + id_parts[0]
                # 重新组合ID
                new_id = new_first_part + '-' + '-'.join(id_parts[1:])
                item['id'] = new_id
                modified_ids += 1
                if modified_ids <= 5:  # 只显示前5个修改示例
                    print(f"ID修改示例: {original_id} -> {new_id}")
        
        # 修改图片路径列表
        if 'image' in item:
            modified_images = []
            for img_path in item['image']:
                # 提取文件夹名和文件名
                path_parts = img_path.split('/')
                if len(path_parts) >= 2:
                    folder_name = path_parts[0]
                    file_name = path_parts[1]
                    # 在文件夹名前添加"00"前缀
                    new_folder_name = "00" + folder_name
                    new_path = f"{new_folder_name}/{file_name}"
                    modified_images.append(new_path)
                else:
                    # 如果路径格式不符合预期，保持原样
                    modified_images.append(img_path)
            
            item['image'] = modified_images
    
    # 保存修改后的JSON文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"修改后的JSON文件已保存到: {output_file_path}")
    print(f"共修改了 {len(data)} 个条目，其中 {modified_ids} 个ID被修改")

def rename_train_folders(train_dir_path):
    """
    重命名train文件夹下的所有文件夹，在名称前添加"00"前缀
    """
    train_path = Path(train_dir_path)
    
    if not train_path.exists():
        print(f"错误: train文件夹不存在: {train_dir_path}")
        return
    
    print(f"正在扫描train文件夹: {train_dir_path}")
    
    # 获取所有数字文件夹
    folders_to_rename = []
    for item in train_path.iterdir():
        if item.is_dir() and item.name.isdigit():
            folders_to_rename.append(item)
    
    # 按数字排序
    folders_to_rename.sort(key=lambda x: int(x.name))
    
    print(f"找到 {len(folders_to_rename)} 个需要重命名的文件夹")
    
    # 重命名文件夹
    renamed_count = 0
    for folder in folders_to_rename:
        old_name = folder.name
        new_name = "00" + old_name
        old_path = folder
        new_path = folder.parent / new_name
        
        try:
            # 检查目标文件夹是否已存在
            if new_path.exists():
                print(f"警告: 目标文件夹已存在，跳过: {new_name}")
                continue
            
            # 重命名文件夹
            old_path.rename(new_path)
            renamed_count += 1
            
            # 显示前几个重命名示例
            if renamed_count <= 5:
                print(f"文件夹重命名示例: {old_name} -> {new_name}")
            
            if renamed_count % 1000 == 0:
                print(f"已重命名 {renamed_count} 个文件夹...")
        
        except Exception as e:
            print(f"重命名文件夹失败 {old_name} -> {new_name}: {e}")
    
    print(f"文件夹重命名完成，共重命名 {renamed_count} 个文件夹")

def main():
    """
    主函数：执行完整的修改流程
    """
    # 设置文件路径
    json_file_path = "/home/liu/datasets/RxR/annotations_llava_rxr.json"
    output_json_path = "/home/liu/datasets/RxR/annotations_llava_rxr_modified.json"
    train_dir_path = "/home/liu/datasets/RxR/train"
    
    print("开始修改RxR数据集...")
    print("=" * 50)
    
    # 1. 修改JSON文件中的ID和路径
    try:
        modify_json_paths(json_file_path, output_json_path)
    except Exception as e:
        print(f"修改JSON文件时出错: {e}")
        return
    
    print("\n" + "=" * 50)
    
    # 2. 重命名train文件夹中的文件夹
    try:
        rename_train_folders(train_dir_path)
    except Exception as e:
        print(f"重命名文件夹时出错: {e}")
        return
    
    print("\n" + "=" * 50)
    print("所有修改完成！")
    print(f"修改后的JSON文件: {output_json_path}")
    print(f"train文件夹中的文件夹已重命名")

def verify_changes(json_file_path, train_dir_path):
    """
    验证修改是否正确
    """
    print("验证修改结果...")
    
    # 验证JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sample_item = data[0] if data else None
        if sample_item:
            if 'id' in sample_item:
                print(f"JSON文件示例ID: {sample_item['id']}")
            if 'image' in sample_item:
                print(f"JSON文件示例路径: {sample_item['image'][0]}")
        
    except Exception as e:
        print(f"验证JSON文件时出错: {e}")
    
    # 验证文件夹重命名
    train_path = Path(train_dir_path)
    if train_path.exists():
        folders = [f.name for f in train_path.iterdir() if f.is_dir()]
        folders.sort()
        print(f"train文件夹中的前几个文件夹: {folders[:5]}")
        print(f"train文件夹中的后几个文件夹: {folders[-5:]}")

if __name__ == "__main__":
    # 运行主程序
    main()
    
    # 验证修改结果
    print("\n" + "=" * 50)
    verify_changes("/home/liu/datasets/RxR/annotations_llava_rxr_modified.json", "/home/liu/datasets/RxR/train")