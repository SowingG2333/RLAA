import os
import json
import sys
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.exceptions import HttpResponseError

# --- 1. 配置 ---
# (在终端运行: export AZURE_LANGUAGE_ENDPOINT=你的终结点)
# (在终端运行: export AZURE_LANGUAGE_KEY=你的密钥)
try:
    AZURE_LANGUAGE_ENDPOINT = os.environ["AZURE_LANGUAGE_ENDPOINT"]
    AZURE_LANGUAGE_KEY = os.environ["AZURE_LANGUAGE_KEY"]
except KeyError:
    print("错误: 请设置 AZURE_LANGUAGE_ENDPOINT 和 AZURE_LANGUAGE_KEY 环境变量。")
    print("例如 (Linux/macOS): export AZURE_LANGUAGE_KEY='你的密钥'")
    sys.exit(1)

# 输入和输出文件
INPUT_FILE = '/root/autodl-tmp/RLAA/Eval/data/PR_resplit/test.jsonl'
OUTPUT_FILE = '/root/autodl-tmp/RLAA/Eval/data/PR_resplit/test_azure.jsonl' # 创建一个新文件

# Azure API 限制：
# - 单个文档最大 5,120 个字符。
# - 每次调用（批处理）最大 10 个文档 (对于 PII 编辑)。
BATCH_SIZE = 10 

# 新增的键名
OUTPUT_KEY_NAME = 'anonymized_response'


def get_azure_client():
    """创建并验证 Azure 客户端"""
    credential = AzureKeyCredential(AZURE_LANGUAGE_KEY)
    client = TextAnalyticsClient(endpoint=AZURE_LANGUAGE_ENDPOINT, credential=credential)
    return client

def write_batch_to_file(f_out, updated_json_objects):
    """将处理完的 JSON 对象写入输出文件"""
    for json_obj in updated_json_objects:
        json_string = json.dumps(json_obj, ensure_ascii=False)
        f_out.write(json_string + '\n')

def anonymize_batch(client, batch_originals):
    """
    调用 Azure API 匿名化一批文本, 并将其写回原始 JSON 对象。
    
    :param client: TextAnalyticsClient
    :param batch_originals: 原始 JSON 对象列表 (字典)
    :return: 更新后的 JSON 对象列表 (字典)
    """
    
    # 1. 准备要发送到 Azure 的文档
    # Azure SDK 需要一个字典列表: [{"id": "0", "text": "..."}, {"id": "1", "text": "..."}]
    documents_to_send = []
    # 存储原始JSON，以便稍后写回
    originals_map = {} 
    
    doc_id_counter = 0
    for original_json in batch_originals:
        text_to_anonymize = original_json.get('response') # 获取 'response' 键的值
        
        # 检查文本是否有效
        if not text_to_anonymize or not isinstance(text_to_anonymize, str) or text_to_anonymize.isspace():
            # 文本无效，跳过 API 调用，但仍需在最后写回
            original_json[OUTPUT_KEY_NAME] = None # 或 text_to_anonymize
            continue

        # 检查是否超过 Azure 字符限制
        if len(text_to_anonymize) > 5120:
            print(f"警告: 序号 {original_json.get('id', 'N/A')} 的 'response' 超过 5120 字符，已截断。")
            text_to_anonymize = text_to_anonymize[:5120]

        doc_id = str(doc_id_counter)
        documents_to_send.append({"id": doc_id, "text": text_to_anonymize})
        originals_map[doc_id] = original_json
        doc_id_counter += 1

    # 如果所有文本都无效，则提前返回
    if not documents_to_send:
        return batch_originals

    # 2. 调用 Azure API
    try:
        # redact_text 是“匿名化”的正确方法。
        # 它返回包含 "****" 的完整文本。
        redacted_results = client.redact_text(
            documents=documents_to_send,
            # 可选：指定要编辑的 PII 类别
            # categories_filter=[PiiEntityCategory.PHONE_NUMBER, PiiEntityCategory.EMAIL]
        )

        # 3. 处理结果并写回
        for result in redacted_results:
            if not result.is_error:
                # 找到此结果对应的原始 JSON 对象
                original_json = originals_map[result.id]
                # 将匿名化的文本写回到 JSON 对象中
                original_json[OUTPUT_KEY_NAME] = result.redacted_text
            else:
                # 处理 API 返回的单个文档错误
                print(f"错误 (文档 ID: {result.id}): {result.error.message}")
                original_json = originals_map[result.id]
                original_json[OUTPUT_KEY_NAME] = f"ERROR: {result.error.message}"
        
        return batch_originals

    except HttpResponseError as azure_error:
        print(f"--- Azure API 错误 (批处理) ---")
        print(f"错误: {azure_error.message}")
        print(f"批处理中的 {len(batch_originals)} 个文档已跳过。")
        # 发生批处理错误时，将所有原始对象返回，不进行修改
        for original_json in batch_originals:
             original_json[OUTPUT_KEY_NAME] = f"ERROR: Azure API 批处理失败"
        return batch_originals
    except Exception as e:
        print(f"--- 未知错误 (批处理) ---")
        print(f"错误: {e}")
        for original_json in batch_originals:
             original_json[OUTPUT_KEY_NAME] = f"ERROR: {e}"
        return batch_originals


def main():
    """主函数：读取、批处理、写入 JSONL"""
    print(f"正在启动 Azure 客户端...")
    client = get_azure_client()
    print("客户端启动成功。")
    print(f"开始处理文件: {INPUT_FILE}")

    batch_originals = [] # 存储当前批次的原始 JSON 对象
    total_lines = 0

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
             open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                
                total_lines += 1
                try:
                    # 1. 读取并解析 JSON
                    json_obj = json.loads(line)
                    batch_originals.append(json_obj)
                    
                    # 2. 检查是否达到批处理大小
                    if len(batch_originals) >= BATCH_SIZE:
                        print(f"正在处理 {total_lines - BATCH_SIZE + 1} - {total_lines} 行...")
                        updated_batch = anonymize_batch(client, batch_originals)
                        write_batch_to_file(f_out, updated_batch)
                        batch_originals = [] # 清空批次

                except json.JSONDecodeError:
                    print(f"警告: 第 {total_lines} 行JSON格式错误，已跳过。")
                    continue
            
            # 3. 处理最后一批 (如果
            if batch_originals:
                print(f"正在处理最后 {len(batch_originals)} 行...")
                updated_batch = anonymize_batch(client, batch_originals)
                write_batch_to_file(f_out, updated_batch)

        print("\n--- 处理完成 ---")
        print(f"总共处理了 {total_lines} 行。")
        print(f"已创建新的匿名化文件: {OUTPUT_FILE}")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到: {INPUT_FILE}")
        sys.exit(1)
    except Exception as e:
        print(f"\n发生严重错误: {e}")

if __name__ == "__main__":
    main()