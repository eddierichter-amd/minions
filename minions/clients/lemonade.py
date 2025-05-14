import logging
from typing import Any, Dict, List, Tuple
from minions.usage import Usage

import sys
import os

conda_env_path = os.environ.get("CONDA_ENV_PATH", "")
if conda_env_path:
    site_packages = os.path.join(conda_env_path, "Lib", "site-packages")  # Windows
    if not os.path.exists(site_packages):
        site_packages = os.path.join(conda_env_path, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")  # Linux/Mac
    if os.path.exists(site_packages):
        sys.path.insert(0, site_packages)
    else:
        raise FileNotFoundError(f"Invalid CONDA_ENV_PATH: {conda_env_path}")
else:
    raise EnvironmentError("Set CONDA_ENV_PATH to your Conda environment directory")

from lemonade.api import from_pretrained

class LemonadeClient:
    def __init__(
        self,
        model_name: str = "amd/Llama-3.2-1B-Instruct-awq-g128-int4-asym-fp16-onnx-hybrid",
        recipe: str = "oga-hybrid",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.recipe = recipe
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger("LemonadeClient")
        self.logger.setLevel(logging.INFO)
        
        # Load model and tokenizer during initialization
        self.model, self.tokenizer = from_pretrained(
            self.model_name,
            recipe=self.recipe
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """Uses local model for generation with chat formatting"""
        try:
            if not messages or not isinstance(messages, list):
                raise ValueError("Invalid messages format")
            
            # Convert chat messages to model's prompt format
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Generate response
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode only the new tokens
            generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
            response_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            
            # Calculate token usage
            usage = Usage(
                prompt_tokens=inputs.input_ids.shape[-1],
                completion_tokens=generated_tokens.shape[-1]
            )
            
            return ([response_text], usage, ["stop"])  # Single response

        except Exception as e:
            self.logger.error(f"Generation Error: {str(e)}")
            return ([""], Usage(), ["error"])
