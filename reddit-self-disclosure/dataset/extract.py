import json
import sys
import os

def extract_by_label_attribute(input_filepath, attribute_key):
    """
    从 JSONL 文件中读取数据，并提取所有 'label' 字典中
    包含指定 'attribute_key' 的条目。
    """
    
    # 1. 基于你输入的属性，动态创建一个输出文件名
    # 例如，输入 "Health" 将会创建 "extracted_Health.jsonl"
    output_filepath = f"extracted_{attribute_key}.jsonl"
    
    total_count = 0
    extracted_count = 0
    
    print(f"\n--- 开始从 '{input_filepath}' 提取 ---")
    print(f"查找 'label' 中包含 '{attribute_key}' 的条目...")

    try:
        # 2. 同时打开输入文件 (读取 'r') 和输出文件 (写入 'w')
        with open(input_filepath, 'r', encoding='utf-8') as infile, \
             open(output_filepath, 'w', encoding='utf-8') as outfile:
            
            # 3. 逐行读取 .jsonl 文件
            for line in infile:
                total_count += 1
                try:
                    # 4. 解析该行
                    # .strip() 移除末尾的换行符
                    item = json.loads(line.strip())
                    
                    # 5. 安全地获取 'label' 字段
                    # .get("label") 如果 "label" 不存在或为 null，会返回 None
                    label = item.get("label")
                    
                    # 6. 核心逻辑：
                    # 检查 'label' 是否是一个字典 (排除了 'null' 或不存在的情况)
                    # 并且 检查 'attribute_key' 是否是该字典的一个键
                    if isinstance(label, dict) and attribute_key in label:
                        
                        # 7. 如果是，将该条目写回新的 .jsonl 文件
                        # 我们重新转储 (dump) 这一行以保持 .jsonl 格式
                        json.dump(item, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        extracted_count += 1
                        
                except json.JSONDecodeError:
                    print(f"  [警告] 跳过一行无法解析的JSON: {line.strip()}")
                except Exception as e:
                    print(f"  [错误] 处理行时发生意外: {line.strip()}: {e}")

    except FileNotFoundError:
        print(f"\n[严重错误] 输入文件 '{input_filepath}' 未找到。")
        print("请确保 'merged_output.jsonl' 与此脚本在同一目录中。")
        return
    except IOError as e:
        print(f"\n[严重错误] 无法写入到输出文件 '{output_filepath}': {e}")
        return
    except Exception as e:
        print(f"发生未知错误: {e}")
        return

    # --- 打印最终报告 ---
    print("\n--- 提取完成 ---")
    print(f"总共读取行数: {total_count}")
    print(f"成功提取行数: {extracted_count}")
    print(f"已保存到: {output_filepath}")

def main():
    # 1. 定义你的输入文件 (上一步生成的)
    input_file = '/home/sowingg/coding/LLM/reddit-self-disclosure/merged_data.jsonl'
    
    # 2. 交互式地询问用户要查找哪个属性
    # .strip() 用于删除用户可能不小心输入的
    attribute_to_find = input("请输入你想要提取的标签属性 (例如: Health, Finance, Pet, Age): ").strip()
    
    # 3. 确保用户输入了内容
    if not attribute_to_find:
        print("错误：未提供属性名称。程序退出。")
        return

    # 4. 运行提取函数
    extract_by_label_attribute(input_file, attribute_to_find)

if __name__ == "__main__":
    main()