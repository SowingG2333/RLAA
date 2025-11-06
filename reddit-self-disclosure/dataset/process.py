import json
from collections import defaultdict
import sys

def parse_conll_to_json(filepath):
    """
    解析 CoNLL IOB2 格式的文件，并将其转换为 JSON 列表。

    每个 JSON 对象包含一个 'response' (完整句子) 和一个 'label' (实体字典)。
    """
    
    # 最终结果列表
    results = []
    
    # 用于构建当前句子的临时变量
    current_sentence_tokens = []
    current_labels = defaultdict(list)
    
    # 用于追踪当前正在构建的实体的临时变量
    active_entity_tokens = []
    active_entity_type = None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误：输入文件 '{filepath}' 未找到。")
        print("请确保 'batch-3-conll-format.txt' 与脚本在同一目录中。")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

    def process_sentence_data():
        """
        辅助函数，用于处理和保存已完成的句子。
        在遇到 [SEP] 或文件末尾时调用。
        """
        nonlocal active_entity_type, active_entity_tokens, current_sentence_tokens, current_labels, results

        # 1. 关闭任何仍在“活跃”状态的实体
        # (例如，句子的最后一个词是一个I-TAG)
        if active_entity_type:
            entity_string = " ".join(active_entity_tokens)
            current_labels[active_entity_type].append(entity_string)

        # 2. 检查当前句子是否有内容
        if current_sentence_tokens:
            # 3. 构建 JSON 对象
            response_string = " ".join(current_sentence_tokens)
            
            # 如果 current_labels 为空，则将 label 设为 None (在JSON中为 null)
            label_data = dict(current_labels) if current_labels else None
            
            json_obj = {
                "response": response_string,
                "label": label_data
            }
            results.append(json_obj)

        # 4. 重置所有临时变量以准备下一句
        current_sentence_tokens.clear()
        current_labels.clear()
        active_entity_tokens.clear()
        active_entity_type = None

    # --- 主循环：逐行解析文件 ---
    for line_number, line in enumerate(lines):
        line = line.strip()

        # 忽略源注释行和空行
        # --- 这是修复的部分 ---
        if line.startswith('#') or not line:
            continue

        # [SEP] 标记表示句子结束
        if line == '[SEP]':
        # --- 修复结束 ---
            process_sentence_data()
            continue

        # --- 处理 "TOKEN TAG" 行 ---
        parts = line.split()

        # 处理格式错误的行 (例如，只有单词没有标签)
        if len(parts) < 2:
            if parts:
                # 将这个孤立的词元添加到句子中
                current_sentence_tokens.append(parts[0])
            # 任何格式错误的行都会打断一个实体的连续性
            if active_entity_type:
                # 如果有实体正在追踪中，立即处理之前的句子部分
                entity_string = " ".join(active_entity_tokens)
                current_labels[active_entity_type].append(entity_string)
                active_entity_tokens.clear()
                active_entity_type = None
            continue
        
        if len(parts) > 2:
             # 有些词元可能包含空格，但标签是最后一个元素
             token = " ".join(parts[:-1])
             tag = parts[-1]
        else:
            token, tag = parts

        # 将词元添加到当前句子
        current_sentence_tokens.append(token)

        # --- IOB2 标签逻辑 ---
        if tag == 'O':
            # "Outside" 标签
            # 如果我们刚刚还在追踪一个实体，那么这个 'O' 标志着该实体的结束
            if active_entity_type:
                entity_string = " ".join(active_entity_tokens)
                current_labels[active_entity_type].append(entity_string)
                active_entity_tokens.clear()
                active_entity_type = None
        
        elif tag.startswith('B-'):
            # "Beginning" 标签
            # 1. 首先，关闭可能正在追踪的上一个实体
            if active_entity_type:
                entity_string = " ".join(active_entity_tokens)
                current_labels[active_entity_type].append(entity_string)
            
            # 2. 开始追踪新实体
            active_entity_type = tag.split('-', 1)[1]
            active_entity_tokens = [token]

        elif tag.startswith('I-'):
            # "Inside" 标签
            current_tag_type = tag.split('-', 1)[1]
            
            if current_tag_type == active_entity_type:
                # 如果类型匹配 (例如 I-Mental_Health 跟在 B-Mental_Health 后面)
                # 则继续添加到当前实体
                active_entity_tokens.append(token)
            else:
                # 这是一个不匹配的 'I' 标签 (例如，I-TagB 跟在 B-TagA 后面, 
                # 或者 'O' 后面直接跟 'I-Tag')
                
                # 1. 关闭上一个实体
                if active_entity_type:
                    entity_string = " ".join(active_entity_tokens)
                    current_labels[active_entity_type].append(entity_string)
                
                # 2. 按规范，'I' 不应开始一个新实体，但在此我们将其视为 'B' 来处理
                active_entity_type = current_tag_type
                active_entity_tokens = [token]

    # --- 文件末尾 ---
    # 处理文件中的最后一个句子（如果它没有以 [SEP] 结尾）
    process_sentence_data()

    return results

def main():
    input_file = '/home/sowingg/coding/LLM/reddit-self-disclosure/data/batch-10-adjudicated-conll-format.txt'
    output_file = '/home/sowingg/coding/LLM/reddit-self-disclosure/output/batch-10.json'
    
    print(f"开始解析 {input_file}...")
    
    data = parse_conll_to_json(input_file)
    
    if data is not None:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # 使用 indent=2 格式化输出，使其易于阅读
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            print(f"处理完成！")
            print(f"已成功将 {len(data)} 个句子对象保存到 {output_file}")
            
        except IOError as e:
            print(f"错误：无法写入到输出文件 '{output_file}': {e}")
        except Exception as e:
            print(f"转换JSON时出错: {e}")

if __name__ == "__main__":
    main()