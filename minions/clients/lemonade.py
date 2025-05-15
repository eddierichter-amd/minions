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
from lemonade.tools.ort_genai.oga import OrtGenaiStreamer


class LemonadeClient:
    def __init__(
        self,
        model_name: str = "amd/Llama-3.2-1B-Instruct-awq-g128-int4-asym-fp16-onnx-hybrid",
        recipe: str = "oga-hybrid",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        streaming: bool = False  # <-- Add streaming option
    ):
        self.model_name = model_name
        self.recipe = recipe
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming  # <-- Store the streaming flag
        self.logger = logging.getLogger("LemonadeClient")
        self.logger.setLevel(logging.INFO)
        try:
            self.model, self.tokenizer = from_pretrained(
                self.model_name,
                recipe=self.recipe
            )
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise

    def _prepare_prompt(self, messages: List[Dict[str, Any]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        streaming: Optional[bool] = None,  # Allow per-call override
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Unified chat interface.
        If streaming is True, uses streaming; else blocking.
        """
        use_streaming = self.streaming if streaming is None else streaming
        if use_streaming:
            return self.schat(messages, **kwargs)
        else:
            return self.bchat(messages, **kwargs)

    def bchat(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """Blocking generation: returns the full response at once."""
        try:
            if not messages or not isinstance(messages, list):
                raise ValueError("Invalid messages format")

            prompt = self._prepare_prompt(messages)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
            response_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            usage = Usage(
                prompt_tokens=inputs.input_ids.shape[-1],
                completion_tokens=generated_tokens.shape[-1]
            )
            return ([response_text], usage, ["stop"])
        except Exception as e:
            self.logger.error(f"Lemonade blocking generation error: {e}")
            return ([""], Usage(), ["error"])

    def schat(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """Streaming generation: yields tokens as they are generated."""
        try:
            if not messages or not isinstance(messages, list):
                raise ValueError("Invalid messages format")

            prompt = self._prepare_prompt(messages)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            streamer = OrtGenaiStreamer(self.tokenizer)
            generation_kwargs = {
                "input_ids": inputs.input_ids,
                "streamer": streamer,
                "max_new_tokens": self.max_tokens,
                "temperature": self.temperature,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            streamed_text = ""
            for new_text in streamer:
                streamed_text += new_text

            thread.join()
            usage = Usage(
                prompt_tokens=inputs.input_ids.shape[-1],
                completion_tokens=len(self.tokenizer(streamed_text, return_tensors="pt").input_ids[0])
            )
            return ([streamed_text], usage, ["stop"])
        except Exception as e:
            self.logger.error(f"Lemonade streaming generation error: {e}")
            return ([""], Usage(), ["error"])