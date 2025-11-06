import json
from collections import Counter
import os
import sys

def count_label_attributes(input_filepath):
    """
    从 JSONL 文件中读取数据，并统计 'label' 字典中
    每个属性 (key) 出现的总次数。
    """
    
    # 1. 初始化一个 Counter 对象来存储标签的计数
    # Counter 是一个特殊的字典，用于计数
    label_counter = Counter()
    
    total_line_count = 0
    lines_with_label_count = 0
    
    print(f"--- 开始统计 '{input_filepath}' 中的标签 ---")

    # 2. 检查文件是否存在
    if not os.path.exists(input_filepath):
        print(f"[严重错误] 输入文件 '{input_filepath}' 未找到。")
        print("请确保 'merged_output.jsonl' 与此脚本在同一目录中。")
        return

    try:
        # 3. 打开并逐行读取 .jsonl 文件
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            for line in infile:
                total_line_count += 1
                try:
                    # 4. 解析该行 (移除末尾的换行符)
                    item = json.loads(line.strip())
                    
                    # 5. 安全地获取 'label' 字段
                    label = item.get("label")
                    
                    # 6. 核心逻辑：
                    # 检查 'label' 是否是一个字典 (排除了 'null' 或不存在的情况)
                    if isinstance(label, dict):
                        lines_with_label_count += 1
                        
                        # 7. 更新 Counter
                        # .keys() 会获取该条目所有的标签, 
                        # .update() 会为每个标签增加计数
                        label_counter.update(label.keys())
                        
                except json.JSONDecodeError:
                    print(f"  [警告] 跳过一行无法解析的JSON: {line.strip()}")
                except Exception as e:
                    print(f"  [错误] 处理行时发生意外: {line.strip()}: {e}")

    except IOError as e:
        print(f"\n[严重错误] 无法读取文件 '{input_filepath}': {e}")
        return
    except Exception as e:
        print(f"发生未知错误: {e}")
        return

    # --- 8. 打印最终报告 ---
    print("\n--- 统计完成 ---")
    print(f"总共读取行数: {total_line_count}")
    print(f"包含 'label' 字典的行数: {lines_with_label_count}")
    
    if not label_counter:
        print("未找到任何标签。")
        return

    print("\n--- 各个标签出现的次数 (降序排列) ---")
    
    # 9. 打印排序后的结果
    # .most_common() 会返回一个 (key, count) 元组的列表，按 count 降序排列
    for attribute, count in label_counter.most_common():
        # 使用 f-string 格式化输出，使其对齐
        # '{attribute:<25}' 表示该字符串左对齐，总共占25个字符宽度
        print(f"{attribute:<25} : {count} 次")

def main():
    input_file = '/home/sowingg/coding/LLM/reddit-self-disclosure/merged_data.jsonl'
    count_label_attributes(input_file)

if __name__ == "__main__":
    main()