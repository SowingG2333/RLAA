import json
import sys

def filter_null_labels(input_filepath, output_filepath):
    """
    加载一个 JSON 文件，移除 "label" 字段为 null 的条目，
    并保存到新的文件中。
    """
    
    # 1. 读取源 JSON 文件
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"错误：'{input_filepath}' 中的内容不是一个 JSON 列表。")
            return
            
        original_count = len(data)
        print(f"已从 '{input_filepath}' 加载 {original_count} 条数据。")

    except FileNotFoundError:
        print(f"错误：输入文件 '{input_filepath}' 未找到。")
        print("请先运行上一个脚本生成 'output.json'。")
        return
    except json.JSONDecodeError:
        print(f"错误：无法解析 '{input_filepath}'。文件可能已损坏或格式不正确。")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 2. 核心过滤逻辑
    # (在JSON中, null 会被 Python 读作 None)
    # 我们使用列表推导式来创建一个新列表，
    # 只包含 item["label"] 不是 None 的那些条目。
    
    # 使用 .get("label") 是一种更安全的方式，
    # 即使 "label" 键不存在也不会引发错误。
    filtered_data = [item for item in data if item.get("label") is not None]

    # 3. 报告结果
    filtered_count = len(filtered_data)
    removed_count = original_count - filtered_count
    
    print(f"过滤完成。")
    print(f" - 保留数据: {filtered_count} 条")
    print(f" - 移除 (label=null): {removed_count} 条")

    # 4. 写入到新文件
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            # 同样使用 indent=2 格式化输出
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
            
        print(f"\n成功！已将过滤后的数据保存到 '{output_filepath}'")
        
    except IOError as e:
        print(f"错误：无法写入到输出文件 '{output_filepath}': {e}")
    except Exception as e:
        print(f"转换并写入JSON时出错: {e}")

def main():
    input_file = '/root/autodl-tmp/RLAA/reddit-self-disclosure/dataset/output/relabel/original/health.jsonl'
    output_file = '/root/autodl-tmp/RLAA/reddit-self-disclosure/dataset/processed/health.jsonl'
    
    print(f"--- 开始清除 '{input_file}' 中的 'null' 标签 ---")
    filter_null_labels(input_file, output_file)

if __name__ == "__main__":
    main()