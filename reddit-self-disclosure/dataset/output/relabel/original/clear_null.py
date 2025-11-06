import json
import os
import sys

def filter_null_by_key(input_filepath, key_to_check):
    """
    过滤 JSONL 文件，移除 "personality" 字典中
    指定 'key_to_check' 的值为 null 的行。
    """
    
    # 1. 动态创建输出文件名
    # 例如: relabeled_output.jsonl -> filtered_relabeled_output_no_null_age.jsonl
    base_name, ext = os.path.splitext(input_filepath)
    output_filepath = f"filtered_{base_name}_no_null_{key_to_check}{ext}"
    
    total_count = 0
    kept_count = 0
    removed_count = 0
    
    print(f"\n--- 开始过滤 '{input_filepath}' ---")
    print(f"将移除 'personality' 中 '{key_to_check}' 为 null 的行...")

    try:
        # 2. 同时打开输入和输出文件
        with open(input_filepath, 'r', encoding='utf-8') as infile, \
             open(output_filepath, 'w', encoding='utf-8') as outfile:
            
            # 3. 逐行读取 .jsonl
            for line in infile:
                total_count += 1
                remove_line = False
                
                try:
                    # 4. 解析该行
                    item = json.loads(line.strip())
                    
                    # 5. 安全地获取 'personality' 字典
                    personality = item.get("personality")
                    
                    # 6. 核心过滤逻辑：
                    # 检查 'personality' 是否是一个字典
                    if isinstance(personality, dict):
                        # 检查 'key_to_check' 是否存在于字典中
                        # 并且 (and) 它的值是否为 None (JSON null)
                        if key_to_check in personality and personality.get(key_to_check) is None:
                            remove_line = True
                    
                    # 7. 根据 'remove_line' 标志决定是写入还是丢弃
                    if remove_line:
                        removed_count += 1
                    else:
                        # 写入原始行 (保留换行符)
                        # 注意：我们写入原始的 'line' 字符串，而不是重新转储 'item'
                        # 这样可以保留原始格式并提高效率
                        outfile.write(line)
                        kept_count += 1
                        
                except json.JSONDecodeError:
                    print(f"  [警告] 跳过一行无法解析的JSON: {line.strip()}")
                    # 决定保留格式错误的行
                    outfile.write(line)
                    kept_count += 1
                except Exception as e:
                    print(f"  [错误] 处理行时发生意外: {line.strip()}: {e}")
                    # 决定保留出错的行
                    outfile.write(line)
                    kept_count += 1

    except FileNotFoundError:
        print(f"\n[严重错误] 输入文件 '{input_filepath}' 未找到。")
        print("请确保文件名正确且文件在同一目录中。")
        return
    except IOError as e:
        print(f"\n[严重错误] 无法写入到输出文件 '{output_filepath}': {e}")
        return
    except Exception as e:
        print(f"发生未知错误: {e}")
        return

    # --- 打印最终报告 ---
    print("\n--- 过滤完成 ---")
    print(f"总共读取行数: {total_count}")
    print(f"保留的行数:   {kept_count}")
    print(f"移除的行数:   {removed_count} (因为 'personality.{key_to_check}' 为 null)")
    print(f"已保存到: {output_filepath}")

def main():
    # 1. 询问输入文件
    input_file = input("请输入你要过滤的 .jsonl 文件名 (例如: relabeled_output.jsonl): ").strip()
    
    # 2. 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：文件 '{input_file}' 未找到。程序退出。")
        return

    # 3. 询问要检查哪个属性
    key_to_check = input("请输入 'personality' 中要检查 'null' 值的属性名 (例如: age, sex, occupation): ").strip()
    
    if not key_to_check:
        print("错误：未提供属性名称。程序退出。")
        return

    # 4. 运行过滤函数
    filter_null_by_key(input_file, key_to_check)

if __name__ == "__main__":
    main()