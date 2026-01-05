"""
Local LLM handler for chat-style inference.

Supports models like Llama-3, Qwen, and other HuggingFace models
with chat template capabilities.
"""

import torch
import logging
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalModelHandler:
    """
    Local model loader that supports chat-style inference.
    
    This handler loads a HuggingFace model and provides a simple
    interface for generating responses from chat messages.
    """

    def __init__(self, model_path: str, device_map: str = "auto", load_in_4bit: bool = False):
        """
        Initialize the local model handler.
        
        Args:
            model_path: Path to the model directory or HuggingFace model ID.
            device_map: Device mapping strategy (default: 'auto').
            load_in_4bit: Whether to use 4-bit quantization.
        """
        logging.info(f"Loading local model from: {model_path} ...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Configure quantization if requested
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            else:
                quantization_config = None

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                trust_remote_code=True,
            )
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.critical(f"Failed to load model: {e}")
            raise

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """
        Generate model output for a list of chat messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0 for greedy decoding).
            
        Returns:
            Generated response text.
        """
        try:
            # Apply chat template to format messages
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0),
                    top_p=0.9 if temperature > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Extract only the newly generated tokens
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, outputs)
            ]
            response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response_text
        except Exception as e:
            logging.error(f"Error during generation: {e}")
            return ""
