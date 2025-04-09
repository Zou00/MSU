import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

# 读取数据
print("正在读取标签数据...")
df = pd.read_csv('data/labelResultAll.txt', sep='\t')
df[['text_label', 'image_label']] = df['text,image'].str.split(',', expand=True)
df = df.drop('text,image', axis=1)
df = df.rename(columns={'ID': 'guid'})

# 创建数据集目录
os.makedirs('data/consistent_labels', exist_ok=True)
os.makedirs('data/all_labels', exist_ok=True)

# 读取文本内容的函数
def read_text_content(guid):
    file_path = f'data/data/{guid}.txt'
    try:
        # 首先尝试使用 utf-8 编码
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
            # 如果内容为空，返回空字符串
            if not content:
                return ""
            # 分割成行并获取第二行（如果存在）
            lines = content.split('\n')
            if len(lines) >= 2:
                return lines[1].strip()
            return lines[0].strip()  # 如果只有一行，返回第一行
    except Exception as e:
        try:
            # 如果 utf-8 失败，尝试使用 gbk 编码
            with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
                content = f.read().strip()
                if not content:
                    return ""
                lines = content.split('\n')
                if len(lines) >= 2:
                    return lines[1].strip()
                return lines[0].strip()
        except Exception as e:
            print(f"警告：无法读取文件 {file_path}")
            return ""
    return ""

print("正在处理数据集划分...")

# 定义一致标签样本（文本和图像标签完全相同）
consistent_mask = (df['text_label'] == df['image_label'])
consistent_samples = df[consistent_mask].copy()
consistent_samples['label'] = consistent_samples['text_label']

# 定义弱标签样本
weak_label_conditions = [
    # Positive + Neutral = positive
    ((df['text_label'] == 'positive') & (df['image_label'] == 'neutral')) |
    ((df['text_label'] == 'neutral') & (df['image_label'] == 'positive')),
    # Negative + Neutral = negative
    ((df['text_label'] == 'negative') & (df['image_label'] == 'neutral')) |
    ((df['text_label'] == 'neutral') & (df['image_label'] == 'negative'))
]
weak_label_mask = pd.concat([cond for cond in weak_label_conditions], axis=1).any(axis=1)
weak_label_samples = df[weak_label_mask].copy()

def get_weak_label(row):
    if (row['text_label'] == 'positive' and row['image_label'] == 'neutral') or \
       (row['text_label'] == 'neutral' and row['image_label'] == 'positive'):
        return 'positive'
    else:  # negative + neutral 或 neutral + negative
        return 'negative'

weak_label_samples['label'] = weak_label_samples.apply(get_weak_label, axis=1)

# 定义冲突样本（positive 和 negative 之间的冲突）
conflict_mask = (
    ((df['text_label'] == 'positive') & (df['image_label'] == 'negative')) |
    ((df['text_label'] == 'negative') & (df['image_label'] == 'positive'))
)
conflict_samples = df[conflict_mask].copy()

# 所有非冲突样本（一致标签 + 弱标签）
all_samples = pd.concat([consistent_samples, weak_label_samples])

def stratified_split(data, test_size=0.2, random_state=42):
    """
    使用分层采样进行数据集划分
    """
    # 首先划分出测试集
    train_val, test = train_test_split(
        data, 
        test_size=test_size,
        stratify=data['label'],
        random_state=random_state
    )
    
    # 然后从剩余数据中划分验证集
    train, val = train_test_split(
        train_val,
        test_size=0.25,  # 0.25 x 0.8 = 0.2 of original data
        stratify=train_val['label'],
        random_state=random_state
    )
    
    return train, val, test

print("正在进行数据集划分...")

# 对一致标签数据集进行分层划分
train_cons, val_cons, test_cons = stratified_split(consistent_samples)

# 对所有标签数据集进行分层划分
train_all, val_all, test_all = stratified_split(all_samples)

# 保存函数
def save_datasets(train, val, test, prefix):
    print(f"\n正在处理 {prefix} 数据集...")
    
    # 保存 txt 格式
    for name, data in [('train', train), ('val', val), ('test', test)]:
        data[['guid', 'label']].to_csv(f'data/{prefix}/{name}.txt', index=False)
        print(f"已保存 {name}.txt")
    
    # 保存 json 格式，包含文本内容
    for name, data in [('train', train), ('val', val), ('test', test)]:
        print(f"\n正在处理 {name}.json...")
        json_data = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc=f"处理{name}集"):
            # 读取对应的文本内容
            text_content = read_text_content(row['guid'])
            json_data.append({
                'guid': str(row['guid']),
                'label': row['label'],
                'text': text_content
            })
        with open(f'data/{prefix}/{name}.json', 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        print(f"已保存 {name}.json，包含 {len(json_data)} 个样本")

# 保存数据集
save_datasets(train_cons, val_cons, test_cons, 'consistent_labels')
save_datasets(train_all, val_all, test_all, 'all_labels')

# 保存标签冲突的样本
print("\n正在保存冲突样本...")
conflict_samples_output = conflict_samples[['guid']].copy()
conflict_samples_output['tag'] = 'null'  # 添加 null 标签列
conflict_samples_output.to_csv('data/test_without_label.txt', index=False)
print("已保存冲突样本")

# 打印数据集统计信息
print("\n=== 总体数据集统计 ===")
print(f"一致标签样本数：{len(consistent_samples)}")
print(f"弱标签样本数：{len(weak_label_samples)}")
print(f"冲突样本数（positive-negative 冲突）：{len(conflict_samples)}")
print(f"总样本数：{len(df)}")

def print_split_stats(name, train, val, test):
    print(f"\n=== {name} ===")
    print("\n训练集分布：")
    print(train['label'].value_counts())
    print(f"训练集总数：{len(train)}")
    print("\n验证集分布：")
    print(val['label'].value_counts())
    print(f"验证集总数：{len(val)}")
    print("\n测试集分布：")
    print(test['label'].value_counts())
    print(f"测试集总数：{len(test)}")
    
    # 打印每个类别的比例
    print("\n各集合中类别比例：")
    for label in train['label'].unique():
        train_ratio = len(train[train['label'] == label]) / len(train)
        val_ratio = len(val[val['label'] == label]) / len(val)
        test_ratio = len(test[test['label'] == label]) / len(test)
        print(f"\n{label}类别比例：")
        print(f"训练集: {train_ratio:.2%}")
        print(f"验证集: {val_ratio:.2%}")
        print(f"测试集: {test_ratio:.2%}")

print_split_stats("一致标签数据集", train_cons, val_cons, test_cons)
print_split_stats("全部数据集（包含弱标签）", train_all, val_all, test_all)

print("\n=== 标签组合详细分布 ===")
combination_dist = df.groupby(['text_label', 'image_label']).size()
print("文本标签-图像标签对应关系：")
print(combination_dist)