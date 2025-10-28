import re
import json
import ast
import sys
from typing import Dict, Any, Optional, List

def parse_attacker_guess(attack_output: str, attribute_name: str) -> Any:
    """
    从 Attacker FULL OUTPUT 块中解析 'Guess:' 行的第一个猜测。
    """
    # 匹配 'Guess:' 行，并提取分号前的第一个值
    guess_match = re.search(r"Guess:\s*([^;\n]+)", attack_output, re.IGNORECASE)
    if not guess_match:
        return None

    guess_str = guess_match.group(1).strip()

    if guess_str.lower() == "unknown":
        return "unknown"

    # 根据属性名称尝试转换类型，模拟原始脚本的行为
    if attribute_name == 'age':
        try:
            # 修复：处理 '25; 28; unknown' 这种情况，只取 '25'
            return int(guess_str.split(';')[0].strip())
        except ValueError:
            return guess_str # 可能是 "unknown" 或其他非整数
    elif attribute_name in ['sex', 'relationship_status', 'income_level']:
        return guess_str.lower()
    else:
        return guess_str

def parse_record_chunk(chunk_content: str, record_id_str: str) -> Optional[Dict[str, Any]]:
    """
    解析单个记录的日志块并提取所有必需的字段。
    """
    
    result = {
        "record_id_debug": record_id_str, # 临时添加，用于调试
        "personality": None,
        "response": None,
        "target_persona": None,
        "anonymized_response": None,
        "anonymization_meta": {
            "status": "unknown",
            "iterations_used": 0,
            "final_leaked_attributes": [],
            "final_attacker_guesses": {},
            "target_attributes": {},
            "attributes_processed": []
        }
    }

    # --- 1. 提取 `personality`, `response`, `target_persona` ---
    # 这些信息在第一个 "Anonymizer PROMPT" 中
    first_anon_prompt_match = re.search(
        r"Anonymizing... --- Anonymizer PROMPT ---(.*?)--- End ---", 
        chunk_content, 
        re.DOTALL
    )
    if not first_anon_prompt_match:
        # 这是一个预期的失败情况（例如，记录被跳过）
        if "Skipped:" in chunk_content or "Skipping record" in chunk_content:
            print(f"[{record_id_str}] 状态：已跳过 (Skipped)。")
            return None
        print(f"[{record_id_str}] 错误：未找到第一个 Anonymizer PROMPT。", file=sys.stderr)
        return None

    prompt_content = first_anon_prompt_match.group(1)

    try:
        # 提取 Target Persona
        target_persona_match = re.search(r"\[Target Persona\]:\s*(\{.*?\})", prompt_content, re.DOTALL)
        result["target_persona"] = json.loads(target_persona_match.group(1))
        
        # 提取 Real Profile (personality)
        real_profile_match = re.search(r"\[Real Profile\]: \(Attributes to Hide\)\s*(\{.*?\})", prompt_content, re.DOTALL)
        result["personality"] = json.loads(real_profile_match.group(1))

        # 提取 Original Comments (response)
        response_match = re.search(r"\[Original Comments\]:\s*(.*?)\s*\[Real Profile\]:", prompt_content, re.DOTALL)
        result["response"] = response_match.group(1).strip()

        # 填充 meta
        attributes_list = list(result["target_persona"].keys())
        result["anonymization_meta"]["target_attributes"] = result["target_persona"]
        result["anonymization_meta"]["attributes_processed"] = attributes_list

    except Exception as e:
        print(f"[{record_id_str}] 错误：解析初始 Prompt 失败: {e}", file=sys.stderr)
        return None

    # --- 2. 提取 `anonymized_response` ---
    # 这是 *最后* 一个 "Anonymizer FULL OUTPUT" 中的 "anonymized_comment"
    all_anon_outputs = re.findall(
        r"Anonymizing... --- Anonymizer FULL OUTPUT ---(.*?)--- End ---", 
        chunk_content, 
        re.DOTALL
    )
    
    if all_anon_outputs:
        last_output_text = all_anon_outputs[-1]
        try:
            # 尝试解析完整的 JSON
            last_anon_output_json = json.loads(last_output_text)
            result["anonymized_response"] = last_anon_output_json.get("anonymized_comment")
        except json.JSONDecodeError:
            # 如果 JSON 格式不规范（例如，日志截断），回退到正则
            print(f"[{record_id_str}] 警告：解析最后一个 Anonymizer JSON 输出失败。回退到 Regex。", file=sys.stderr)
            last_anon_output_match = re.search(
                r'"anonymized_comment":\s*"(.*?)"', 
                last_output_text, 
                re.DOTALL
            )
            if last_anon_output_match:
                 # 需要 un-escape 字符串中的换行符等
                 result["anonymized_response"] = last_anon_output_match.group(1).encode('utf-8').decode('unicode_escape')
            else:
                print(f"[{record_id_str}] 错误：无法从最后一个 Anonymizer 输出中提取 'anonymized_comment'。", file=sys.stderr)
                result["anonymized_response"] = "[Extraction Failed]"
    else:
        # 如果没有 Anonymizer 输出（例如，在 Round 1 之前就失败了）
        print(f"[{record_id_str}] 警告：未找到 'Anonymizer FULL OUTPUT'。anonymized_response 将为 null。", file=sys.stderr)
        result["anonymized_response"] = None # 或 result["response"]

    # --- 3. 提取 `anonymization_meta` ---
    
    # 状态 (Status)
    if re.search(r"Max iterations reached", chunk_content):
        result["anonymization_meta"]["status"] = "max_iterations_reached"
    elif re.search(r"Early Stopping", chunk_content):
        result["anonymization_meta"]["status"] = "success"
    elif re.search(r"Attack failed", chunk_content):
         result["anonymization_meta"]["status"] = "model_error"
    elif re.search(r"Skipped:", chunk_content):
        result["anonymization_meta"]["status"] = "skipped" # 
# `parse_record_chunk` 早期会退出，但以防万一

    # 迭代次数 (Iterations Used)
    rounds = re.findall(r"\[Round (\d+)/\d+\]", chunk_content)
    if rounds:
        result["anonymization_meta"]["iterations_used"] = int(rounds[-1])
    elif "Anonymizing..." in chunk_content: # 至少运行了
        result["anonymization_meta"]["iterations_used"] = 1 # 
# 
# 
        pass # 
# 
# 

    # 最终泄露属性 (Final Leaked Attributes)
    # 查找 *最后* 一个 "Rule-based comparison"
    leaked_matches = re.findall(r"Rule-based comparison complete\. Leaked: (\[.*?\])", chunk_content)
    if leaked_matches:
        try:
            # 使用 ast.literal_eval 来安全地解析 Python 列表字符串
            leaked_list_str = leaked_matches[-1]
            result["anonymization_meta"]["final_leaked_attributes"] = ast.literal_eval(leaked_list_str)
        except Exception as e:
            print(f"[{record_id_str}] 警告：解析 Leaked attributes 失败: {e}", file=sys.stderr)
            result["anonymization_meta"]["final_leaked_attributes"] = [f"Parsing failed: {leaked_list_str}"]

    # 最终攻击者猜测 (Final Attacker Guesses)
    attributes_to_find = result["anonymization_meta"]["attributes_processed"]
    for attr in attributes_to_find:
        # 查找特定属性的所有 "Attacker FULL OUTPUT" 块
        attr_attack_outputs = re.findall(
            r"Attacking '" + re.escape(attr) + r"' --- Attacker FULL OUTPUT ---(.*?)--- End ---",
            chunk_content,
            re.DOTALL
        )
        if attr_attack_outputs:
            # 获取最后一次攻击的输出
            last_attack_output = attr_attack_outputs[-1]
            # 解析这个输出
            guess = parse_attacker_guess(last_attack_output, attr)
            result["anonymization_meta"]["final_attacker_guesses"][attr] = guess
        else:
            result["anonymization_meta"]["final_attacker_guesses"][attr] = None

    return result


def main():
    if len(sys.argv) != 3:
        print("用法: python parser_v3.py <log_file_path> <output_jsonl_path>")
        print("示例: python parser_v3.py incongni_llama_origin.log results.jsonl")
        sys.exit(1)

    log_file_path = sys.argv[1]
    output_jsonl_path = sys.argv[2]
    
    print(f"正在读取日志文件: {log_file_path}")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误：日志文件未找到: {log_file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误：读取日志文件失败: {e}", file=sys.stderr)
        sys.exit(1)

    # --- v3: 新的、正确的日志分组逻辑 ---
    all_records_content: Dict[str, List[str]] = {}
    current_record_id: Optional[str] = None
    record_id_regex = re.compile(r"\[Record-(\d+)\]")
    
    header_lines: List[str] = [] # 用于跳过日志开头的行

    print("正在按 Record ID 分组日志...")
    for line in lines:
        match = record_id_regex.search(line)
        
        if match:
            # 找到了一个新的 Record ID
            current_record_id = match.group(1) # '0', '1', etc.
            if current_record_id not in all_records_content:
                all_records_content[current_record_id] = []
            all_records_content[current_record_id].append(line)
        else:
            # 没有找到 Record ID，此行属于上一个ID
            if current_record_id is not None:
                # 将此行附加到当前活动的记录中
                all_records_content[current_record_id].append(line)
            else:
                # `current_record_id` 仍为 None，因此这是日志标题
                header_lines.append(line)

    print(f"日志分组完毕。检测到 {len(all_records_content)} 个独立的 Record ID。")
    if header_lines:
         print(f"跳过了 {len(header_lines)} 行日志标题。")

    final_results = []
    
    # 遍历 0 到 110
    for i in range(111):
        record_num_str = str(i)
        record_id_str = f"Record-{i}"
        
        if record_num_str in all_records_content:
            print(f"正在处理: [{record_id_str}]...")
            # 将该记录的所有行合并为一个大字符串块
            chunk_content = "".join(all_records_content[record_num_str])
            
            try:
                parsed_data = parse_record_chunk(chunk_content, record_id_str)
                if parsed_data:
                    # 构建最终的 JSON 对象
                    final_obj = {
                        "personality": parsed_data["personality"],
                        "response": parsed_data["response"],
                        "target_persona": parsed_data["target_persona"],
                        "anonymized_response": parsed_data["anonymized_response"],
                        "anonymization_meta": parsed_data["anonymization_meta"]
                    }
                    final_results.append(final_obj)
                # `else` 的情况（例如 "Skipped"）已经被 `parse_record_chunk` 打印
            except Exception as e:
                print(f"!! 处理 [{record_id_str}] 时发生严重错误: {e}", file=sys.stderr)
        else:
            # 这是正常的，例如日志中只有 0-109
            print(f"[{record_id_str}] 在日志文件中未找到。")

    # --- 写入 `jsonl` 文件 ---
    print(f"\n处理完成。正在将 {len(final_results)} 条有效结果写入: {output_jsonl_path}")
    try:
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for item in final_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"错误：写入输出文件失败: {e}", file=sys.stderr)
        sys.exit(1)

    print("脚本执行完毕。")

if __name__ == "__main__":
    main()