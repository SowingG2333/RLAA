"""
Utility functions for the RLAA project (health self-disclosure variant).

This module provides common utilities for:
- Logging configuration
- JSONL file I/O operations
- JSON extraction from LLM outputs
- API calls to DeepSeek/OpenAI-style endpoints
- Privacy leakage comparison (health_issue specific)
- Logging helpers for debugging
- Text metrics calculation (ROUGE, BLEU)
"""

import os
import json
import logging
import sys
import re
import time
import requests
from typing import List, Dict, Any, Optional, Union

# =============================================================================
# 1. Logging Configuration
# =============================================================================

def setup_logging(log_level_str: str = "INFO", log_file: Optional[str] = None):
    """
    Configure the root logger with console and optional file handlers.
    
    Args:
        log_level_str: Logging level as string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to a log file.
    """
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, "w", "utf-8"))

    # Reset root logger to avoid duplicate handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    for handler in handlers:
        handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(handler)

    # Silence noisy third-party loggers
    for lib in ["urllib3", "requests", "openai", "httpx", "transformers"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


# =============================================================================
# 2. I/O Helpers
# =============================================================================

def load_jsonl(filepath: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load records from a JSONL file.
    
    Args:
        filepath: Path to the JSONL file.
        limit: Optional maximum number of records to load.
        
    Returns:
        List of parsed JSON objects.
    """
    data: List[Dict[str, Any]] = []
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        sys.exit(1)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and len(data) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed JSON on line {i + 1}")
        logging.info(f"Loaded {len(data)} records from {filepath}")
        return data
    except Exception as e:
        logging.error(f"Error loading {filepath}: {e}")
        sys.exit(1)


def save_jsonl(data: List[Dict[str, Any]], filepath: str):
    """
    Save records to a JSONL file.
    
    Args:
        data: List of dictionaries to save.
        filepath: Output file path.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logging.info(f"Saved {len(data)} records to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save to {filepath}: {e}")


# =============================================================================
# 3. JSON Extraction Helpers
# =============================================================================

def extract_first_json(text: str) -> Optional[Union[Dict, List]]:
    """
    Extract the first valid JSON object or list from an LLM output string.
    
    LLM outputs may include bracketed tags like "[Comment]" before the real JSON.
    This implementation scans all possible '{' / '[' start positions in order and
    returns the first parseable JSON segment.
    
    Args:
        text: Raw text potentially containing JSON.
        
    Returns:
        Parsed JSON object/list, or None if not found.
    """
    if not text:
        return None

    stripped = text.strip()
    # Handle ```json ...``` style fences
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()

    def _try_parse_from(s: str, start_idx: int, start_char: str, end_char: str):
        stack = 0
        for i in range(start_idx, len(s)):
            ch = s[i]
            if ch == start_char:
                stack += 1
            elif ch == end_char:
                stack -= 1
                if stack == 0:
                    candidate = s[start_idx : i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        return None
        return None

    for idx, ch in enumerate(stripped):
        if ch == "{":
            parsed = _try_parse_from(stripped, idx, "{", "}")
            if parsed is not None:
                return parsed
        elif ch == "[":
            parsed = _try_parse_from(stripped, idx, "[", "]")
            if parsed is not None:
                return parsed

    return None


def parse_attacker_output(response_text: str) -> Dict[str, Any]:
    """
    Parse attacker output to extract the inference block and the guess JSON.
    
    Args:
        response_text: Raw response from the attacker model.
        
    Returns:
        Dictionary with 'inference' (str) and 'guess_json' (dict) keys.
    """
    inference = ""
    guess_json = {}

    # Extract inference block
    match = re.search(r"Inference:(.*?)(Guess:|$)", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        inference = match.group(1).strip()

    # Extract JSON
    extracted = extract_first_json(response_text)
    if isinstance(extracted, dict):
        guess_json = extracted
    else:
        guess_json = {"error": "No valid JSON object found"}

    return {"inference": inference, "guess_json": guess_json}


# =============================================================================
# 4. API Calls
# =============================================================================

def call_deepseek_api(
    messages: List[Dict[str, str]],
    model_name: str,
    api_key: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> Optional[str]:
    """
    Call DeepSeek/OpenAI-style chat completions API.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'.
        model_name: Model identifier (e.g., 'deepseek-chat').
        api_key: API key (defaults to API_KEY environment variable).
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.
        max_retries: Number of retry attempts on failure.
        
    Returns:
        Response content string, or None on failure.
    """
    api_key = api_key or os.getenv("API_KEY")
    if not api_key:
        logging.error("API_KEY not found.")
        return None

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    url = "https://api.deepseek.com/chat/completions"

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logging.warning(f"API call failed (Attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2 * (attempt + 1))

    logging.error("API call failed after max retries.")
    return None


# =============================================================================
# 5. Privacy Leakage Comparison (health_issue specific)
# =============================================================================

def compare_profiles_rules(true_profile: Dict[str, Any], guessed_profile: Dict[str, Any]) -> List[str]:
    """
    Rule-based comparison for privacy leakage (health self-disclosure).
    
    For reddit-self-disclosure, true_profile typically contains:
      - health_issue: List[str]
    guessed_profile typically contains:
      - health_issue: str
    
    Args:
        true_profile: Ground truth profile dictionary.
        guessed_profile: Attacker's guessed profile dictionary.
        
    Returns:
        List of leaked attribute names.
    """
    leaks: List[str] = []
    if not true_profile or not guessed_profile or "error" in guessed_profile:
        return leaks

    # Handle health_issue (list vs string comparison)
    if "health_issue" in true_profile:
        true_val = true_profile.get("health_issue")
        guess_val = guessed_profile.get("health_issue")

        if isinstance(guess_val, str):
            guess_s = guess_val.strip().lower()
        else:
            guess_s = ""

        if guess_s and guess_s != "unknown" and len(guess_s) > 2:
            if isinstance(true_val, list):
                true_list = [str(x).strip().lower() for x in true_val if str(x).strip()]
                for t in true_list:
                    if len(t) <= 2:
                        continue
                    if guess_s in t or t in guess_s:
                        leaks.append("health_issue")
                        break
            elif isinstance(true_val, str):
                t = true_val.strip().lower()
                if t and (guess_s in t or t in guess_s):
                    leaks.append("health_issue")

    return leaks


# =============================================================================
# 6. Logging Helpers for Debugging
# =============================================================================

def truncate_text(text: Optional[str], max_chars: int = 400) -> str:
    """
    Truncate text for display purposes.
    
    Args:
        text: Input text.
        max_chars: Maximum number of characters to keep.
        
    Returns:
        Truncated text with length indicator if truncated.
    """
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [truncated, len={len(text)}]"


def make_context_tag(record_index: Optional[int] = None, iteration: Optional[int] = None) -> str:
    """
    Create a context tag string for logging.
    
    Args:
        record_index: Current record index.
        iteration: Current iteration number.
        
    Returns:
        Formatted context tag like '[record 1][iter 2]'.
    """
    rid = "?" if record_index is None else str(record_index)
    it = "?" if iteration is None else str(iteration)
    return f"[record {rid}][iter {it}]"


def log_full_messages(tag: str, messages: Optional[List[Dict[str, str]]]):
    """
    Log full message list for debugging.
    
    Args:
        tag: Context tag prefix.
        messages: List of message dictionaries.
    """
    logger = logging.getLogger(__name__)
    logger.info("%s ===== MESSAGES (full) =====", tag)
    for i, m in enumerate(messages or []):
        role = (m or {}).get("role", "unknown")
        content = (m or {}).get("content", "")
        logger.info("%s --- message[%d] role=%s ---\n%s", tag, i, role, content)
    logger.info("%s ===== END MESSAGES =====", tag)


def log_io_block(tag: str, kind: str, content: Optional[str]):
    """
    Log input/output block for debugging.
    
    Args:
        tag: Context tag prefix.
        kind: Type of block (e.g., 'INPUT_TEXT', 'OUTPUT_RAW').
        content: Content to log.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        "%s ===== %s (full) =====\n%s\n%s ===== END %s =====",
        tag,
        kind,
        (content or ""),
        tag,
        kind,
    )


# =============================================================================
# 7. RLAA Processing Helpers
# =============================================================================

def is_sanity_check_failed(new_comment: str, old_comment: str, min_len: int = 20) -> Optional[str]:
    """
    Check if anonymizer output failed sanity checks.
    
    Args:
        new_comment: New anonymized text.
        old_comment: Previous text.
        min_len: Minimum acceptable length.
        
    Returns:
        Error message string if failed, None if passed.
    """
    if not new_comment or len(new_comment.strip()) < min_len:
        return f"catastrophic_anonymizer_failure: output too short (len={len((new_comment or '').strip())})"
    if new_comment.strip() == old_comment.strip():
        return "anonymizer_stuck: output identical to previous iteration"
    return None


def filter_actionable_leaks(
    validated_leaks: Union[Dict, List, None],
    validity_levels: tuple = ("high",),
) -> List[Dict[str, Any]]:
    """
    Filter validated leaks to keep only actionable ones.
    
    Args:
        validated_leaks: List of leak dictionaries from arbitrator.
        validity_levels: Tuple of validity levels to consider actionable.
        
    Returns:
        List of actionable leak dictionaries.
    """
    if not validated_leaks:
        return []
    if isinstance(validated_leaks, dict):
        validated_leaks = [validated_leaks]
    if not isinstance(validated_leaks, list):
        return []

    actionable = []
    for item in validated_leaks:
        if not isinstance(item, dict):
            continue
        validity = str(item.get("validity_level", "")).lower()
        if validity in validity_levels:
            actionable.append(item)
    return actionable


def format_feedback_for_anonymizer(actionable_leaks: Optional[List[Dict[str, Any]]]) -> str:
    """
    Format leaked concepts as JSON string for anonymizer prompt.
    
    Args:
        actionable_leaks: List of actionable leak dictionaries.
        
    Returns:
        JSON string containing leak information.
    """
    normalized = []
    for item in actionable_leaks or []:
        evidence = item.get("reasoning_evidence", [])
        if evidence is None:
            evidence = []
        if isinstance(evidence, str):
            evidence = [evidence]

        normalized.append({
            "attribute": item.get("attribute"),
            "validity_level": item.get("validity_level"),
            "leaked_concept": item.get("leaked_concept"),
            "evidence": evidence,
            "reasoning_evidence": evidence,
            "validation_notes": item.get("validation_notes"),
        })
    return json.dumps(normalized, ensure_ascii=False, indent=2)


def count_validity_levels(leaks: Union[Dict, List, None]) -> Dict[str, int]:
    """
    Count occurrences of each validity level in leak list.
    
    Args:
        leaks: List of leak dictionaries.
        
    Returns:
        Dictionary with counts for each validity level.
    """
    counts = {"high": 0, "medium": 0, "low": 0, "invalid": 0, "other": 0}
    if not leaks:
        return counts
    if isinstance(leaks, dict):
        leaks = [leaks]
    if not isinstance(leaks, list):
        counts["other"] = 1
        return counts

    for item in leaks:
        if not isinstance(item, dict):
            counts["other"] += 1
            continue
        v = str(item.get("validity_level", "")).lower()
        if v in counts:
            counts[v] += 1
        else:
            counts["other"] += 1
    return counts


# =============================================================================
# 8. Text Metrics (for evaluation)
# =============================================================================

def calculate_text_metrics(original: str, anonymized: str) -> Dict[str, float]:
    """
    Compute ROUGE-L and BLEU metrics between original and anonymized text.
    
    Args:
        original: Original text.
        anonymized: Anonymized text.
        
    Returns:
        Dictionary with 'rouge_l_f1' and 'bleu' scores.
    """
    metrics: Dict[str, float] = {}

    # ROUGE-L
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(original, anonymized)
        metrics["rouge_l_f1"] = scores["rougeL"].fmeasure
    except Exception:
        metrics["rouge_l_f1"] = 0.0

    # BLEU
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref = original.split()
        cand = anonymized.split()
        if not ref or not cand:
            metrics["bleu"] = 0.0
        else:
            chencherry = SmoothingFunction().method7
            metrics["bleu"] = sentence_bleu([ref], cand, smoothing_function=chencherry)
    except Exception:
        metrics["bleu"] = 0.0

    return metrics
