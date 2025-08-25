import json
import os
import shutil
from collections import OrderedDict

def extract_first_n_ids(json_file, train_dir, n=433):
    """
    提取前n个唯一ID及其对应的数据
    """
    # 读取JSON文件
    print(f"正在读取 {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总共有 {len(data)} 条记录")
    
    # 提取唯一的ID（从id字段中提取数字部分）
    unique_ids = OrderedDict()
    for item in data:
        # 从 "914-23" 格式中提取 "914"
        id_str = item['id'].split('-')[0]
        if id_str not in unique_ids:
            unique_ids[id_str] = []
        unique_ids[id_str].append(item)
    
    print(f"找到 {len(unique_ids)} 个唯一ID")
    
    # 取前n个ID
    selected_ids = list(unique_ids.keys())[:n]
    print(f"选择前 {len(selected_ids)} 个ID")
    
    # 创建测试目录
    test_dir = "test"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"创建目录: {test_dir}")
    
    # 收集要移动的数据
    test_data = []
    moved_folders = []
    
    for id_str in selected_ids:
        # 添加该ID的所有记录到测试数据
        test_data.extend(unique_ids[id_str])
        
        # 检查并移动对应的文件夹
        source_folder = os.path.join(train_dir, id_str)
        dest_folder = os.path.join(test_dir, id_str)
        
        if os.path.exists(source_folder):
            if os.path.exists(dest_folder):
                print(f"目标文件夹已存在，跳过: {dest_folder}")
            else:
                print(f"移动文件夹: {source_folder} -> {dest_folder}")
                shutil.move(source_folder, dest_folder)
                moved_folders.append(id_str)
        else:
            print(f"警告: 源文件夹不存在: {source_folder}")
    
    # 保存测试数据到test.json
    test_json_file = "test.json"
    print(f"保存测试数据到 {test_json_file}...")
    with open(test_json_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # 从原始数据中移除已提取的记录
    remaining_data = []
    for item in data:
        id_str = item['id'].split('-')[0]
        if id_str not in selected_ids:
            remaining_data.append(item)
    
    # 更新原始JSON文件
    print(f"更新原始文件 {json_file}...")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(remaining_data, f, ensure_ascii=False, indent=2)
    
    print("\n=== 操作完成 ===")
    print(f"提取的ID数量: {len(selected_ids)}")
    print(f"提取的记录数量: {len(test_data)}")
    print(f"移动的文件夹数量: {len(moved_folders)}")
    print(f"剩余记录数量: {len(remaining_data)}")
    print(f"测试数据保存到: {test_json_file}")
    print(f"测试文件夹: {test_dir}")
    
    # 显示前10个提取的ID作为示例
    print(f"\n前10个提取的ID: {selected_ids[:10]}")
    
    return selected_ids, test_data, moved_folders

if __name__ == "__main__":
    # 设置文件路径
    json_file = "/home/liu/dataset/RxR/annotations_llava_rxr.json"
    train_dir = "/home/liu/dataset/RxR/train"
    

    # 检查文件是否存在
    if not os.path.exists(json_file):
        print(f"错误: 找不到文件 {json_file}")
        exit(1)
    
    if not os.path.exists(train_dir):
        print(f"错误: 找不到目录 {train_dir}")
        exit(1)
    
    # 执行提取操作
    try:
        selected_ids, test_data, moved_folders = extract_first_n_ids(json_file, train_dir, 433)
        print("脚本执行成功！")
    except Exception as e:
        print(f"错误: {str(e)}")
        exit(1)