import json
import os

def merge_json_to_jsonl(input_pattern, num_files, output_filepath):
    """
    合并多个 JSON 文件 (每个文件包含一个对象列表) 
    到一个 JSONL 文件中 (每个对象占一行)。

    :param input_pattern: 带有 {i} 占位符的输入文件名字典, e.g., 'batch-{i}-filter.json'
    :param num_files: 文件的总数 (e.g., 10)
    :param output_filepath: 最终的 .jsonl 输出文件
    """
    
    total_records_written = 0
    files_processed = 0
    files_missing = 0

    print(f"--- 开始合并文件到 {output_filepath} ---")

    # 使用 'w' (写入) 模式打开输出文件
    # 如果文件已存在，它将被覆盖
    try:
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            
            # 循环遍历文件编号 1 到 num_files
            for i in range(1, num_files + 1):
                # 构建当前要查找的文件名
                current_input_file = input_pattern.format(i=i)
                
                try:
                    # 尝试打开并读取当前的 JSON 文件
                    with open(current_input_file, 'r', encoding='utf-8') as infile:
                        # 加载整个 JSON 文件 (应该是一个列表)
                        data = json.load(infile)
                        
                        # 确保数据是一个列表
                        if not isinstance(data, list):
                            print(f"警告：文件 '{current_input_file}' 不包含 JSON 列表，已跳过。")
                            continue
                        
                        # 遍历列表中的每个对象 (item)
                        for item in data:
                            # 将该对象转换为 JSON 字符串
                            # ensure_ascii=False 以正确处理中文字符
                            json_line = json.dumps(item, ensure_ascii=False)
                            
                            # 将该字符串写入输出文件，并添加一个换行符
                            outfile.write(json_line + '\n')
                            total_records_written += 1
                        
                        print(f"  [成功] 已处理: {current_input_file} (包含 {len(data)} 条记录)")
                        files_processed += 1

                except FileNotFoundError:
                    # 如果文件未找到 (例如你只有 batch-1 和 batch-2)，则打印信息并继续
                    print(f"  [跳过] 未找到文件: '{current_input_file}'")
                    files_missing += 1
                except json.JSONDecodeError:
                    print(f"  [错误] 无法解析 '{current_input_file}'，文件可能已损坏。已跳过。")
                except Exception as e:
                    print(f"  [错误] 打开 '{current_input_file}' 时出错: {e}。已跳过。")

    except IOError as e:
        print(f"严重错误：无法写入到输出文件 '{output_filepath}': {e}")
        return
    except Exception as e:
        print(f"发生未知错误: {e}")
        return

    # --- 打印最终报告 ---
    print("\n--- 合并完成 ---")
    print(f"成功处理文件数: {files_processed}")
    print(f"未找到/跳过文件数: {files_missing}")
    print(f"总计写入记录 (行) 数: {total_records_written}")
    print(f"已保存到: {output_filepath}")

def main():
    # 1. 定义文件名模式。{i} 将被替换为 1, 2, 3...
    input_file_pattern = '/home/sowingg/coding/LLM/reddit-self-disclosure/output/batch-{i}-filter.json'
    
    # 2. 你提到了 10 个文件
    total_files_to_check = 10
    
    # 3. 定义输出的 JSONL 文件名
    output_file = '/home/sowingg/coding/LLM/reddit-self-disclosure/output/merged_data.jsonl'
    
    merge_json_to_jsonl(input_file_pattern, total_files_to_check, output_file)

if __name__ == "__main__":
    main()