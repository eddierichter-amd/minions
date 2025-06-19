import os
import logging
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from minions.clients.openai import OpenAIClient
from minions.usage import Usage

# Try to import llama_cpp for GGUF model support
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None
    LLAMA_CPP_AVAILABLE = False

class LemonadeClient(OpenAIClient):
    """Client for interacting with a local Lemonade inference server."""

    def __init__(
        self,
        model_name: str = "Llama-3.2-3B-Instruct-Hybrid",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        base_url = base_url or os.getenv("LEMONADE_BASE_URL", "http://localhost:8000/api/v1")
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            **kwargs,
        )
        self.session = requests.Session()
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.llamacpp_missing_warning = None
        self.run_gguf = "GGUF" in model_name.upper()
        # Only set warning if user will use chat_gguf and llama_cpp is missing
        if self.run_gguf and not LLAMA_CPP_AVAILABLE:
            self.llamacpp_missing_warning = (
                "You have selected a GGUF model, but the 'llama-cpp-python' package is not installed. "
                "Please install it with `pip install llama-cpp-python` to enable GGUF model features."
            )
        self._ensure_model_available()

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using direct HTTP requests to the lemonade service.
        """
        assert len(messages) > 0, "Messages cannot be empty."

        # send to GGUF chat function if it is a GGUF model
        if self.run_gguf:
            return self.streaming_chat_gguf(messages, **kwargs)

        try:
            # Prepare the request payload for lemonade's OpenAI-compatible endpoint
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs,
            }

            # Make direct HTTP request to lemonade's chat completions endpoint
            response = self.session.post(
                f"{self.base_url.rstrip('/api/v1')}/api/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            response_data = response.json()
        except Exception as e:
            self.logger.error(f"Error during Lemonade API call: {e}")
            raise

        # Extract responses from the lemonade response
        choices = response_data.get("choices", [])
        responses = [choice["message"]["content"] for choice in choices if "message" in choice]


        usage = Usage()

        usage += Usage(
            prompt_tokens=response_data.get('usage', 0)['prompt_tokens'],
            completion_tokens=response_data.get('usage', 0)['completion_tokens'],
        )

        done_reason = [choice.get("finish_reason", "stop") for choice in choices]

        return responses, usage, done_reason

    # ----------------- GGUF SUPPORT BELOW -----------------

    @staticmethod
    def remove_thinking_sections(text: str) -> str:
        """
        Removes all <think>...</think> sections from text.
        """
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return cleaned.strip()

    def get_gguf_model_path(self, model_name: str) -> str:
        """
        Retrieves the GGUF model path from Lemonade Server's configuration.
        Returns None if not found.
        """
        # Determine Lemonade Server configuration path
        if os.name == 'nt':  # Windows
            appdata = os.getenv('LOCALAPPDATA', '')
            config_path = os.path.join(appdata, 'lemonade_server', 'src', 'lemonade_server', 'server_models.json')
        else:  # Linux/macOS
            home = os.path.expanduser('~')
            config_path = os.path.join(home, '.config', 'lemonade_server', 'server_models.json')
        
        if not os.path.exists(config_path):
            print(f"Lemonade config not found at: {config_path}")
            return None
        
        try:
            with open(config_path, 'r') as f:
                models_config = json.load(f)
            
            # Find the model in the configuration
            if model_name in models_config:
                checkpoint = models_config[model_name].get("checkpoint", "")
                
                # Extract repository and filename from checkpoint format: "org/repo:filename"
                if ':' in checkpoint:
                    repo_part, file_part = checkpoint.split(':', 1)
                    repo_name = repo_part.replace('/', '--')
                    
                    # Construct Hugging Face cache path
                    hf_cache = os.getenv('HF_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub'))
                    repo_path = os.path.join(hf_cache, f'models--{repo_name}')
                    
                    # Find the latest snapshot
                    snapshots_dir = os.path.join(repo_path, 'snapshots')
                    if os.path.exists(snapshots_dir):
                        snapshots = sorted(os.listdir(snapshots_dir), reverse=True)
                        for snapshot in snapshots:
                            snapshot_path = os.path.join(snapshots_dir, snapshot)
                            # Look for the full GGUF file, e.g., Qwen3-8B-Q4_1.gguf
                            gguf_files = [f for f in os.listdir(snapshot_path) if f.lower().endswith('.gguf')]
                            for fname in gguf_files:
                                # Check if the file matches the expected quantization and model name
                                if file_part.lower() in fname.lower():
                                    model_path = os.path.join(snapshot_path, fname)
                                    if os.path.exists(model_path):
                                        return model_path
                        print(f"GGUF file not found for {file_part} in {snapshots_dir}")
                else:
                    print(f"Invalid checkpoint format for {model_name}: {checkpoint}")
            else:
                print(f"Model {model_name} not found in Lemonade configuration")
                
        except Exception as e:
            print(f"Error reading Lemonade config: {str(e)}")
        
        return None

    def streaming_chat_gguf(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """
        GGUF-specialized chat: removes <think> sections, uses llama-cpp for token counting if available.
        Handles streaming responses for GGUF/llama.cpp models and accumulates usage properly.
        """
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError(
                "llama-cpp-python is required for GGUF model support. "
                "Please install it with `pip install llama-cpp-python`."
            )
        assert len(messages) > 0, "Messages cannot be empty."

        # Compose the prompt text for tokenization
        prompt_text = "\n".join([msg["content"] for msg in messages])

        # Get GGUF model path and tokenizer
        gguf_path = self.get_gguf_model_path(self.model_name)
        tokenizer = None
        if gguf_path:
            try:
                tokenizer = Llama(model_path=gguf_path, n_ctx=2048, embedding=False, logits_all=False)
                self.logger.info(f"Using tokenizer for: {gguf_path}")
            except Exception as e:
                self.logger.error(f"Tokenizer init failed: {str(e)}")

        # Prepare the request payload for Lemonade's GGUF endpoint
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,  # Ensure streaming is enabled
            **kwargs,
        }
        url = f"{self.base_url.rstrip('/api/v1')}/api/v0/chat/completions"
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url, headers=headers, json=payload, stream=True)
            response.raise_for_status()

            content = ""
            done_reasons = []

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    json_str = line[len("data: "):]
                    if json_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(json_str)
                        if data.get("choices"):
                            choice = data["choices"][0]
                            delta = choice.get("delta", {})
                            chunk = delta.get("content")
                            if chunk:
                                content += chunk
                            finish_reason = choice.get("finish_reason")
                            if finish_reason:
                                done_reasons.append(finish_reason)
                    except json.JSONDecodeError:
                        continue

            clean_content = self.remove_thinking_sections(content) if content else ""
            responses = [clean_content]

            # Token counting (count ONCE for prompt, ONCE for full completion)
            if tokenizer:
                prompt_tokens = len(tokenizer.tokenize(prompt_text.encode("utf-8"), add_bos=False))
                completion_tokens = len(tokenizer.tokenize(clean_content.encode("utf-8"), add_bos=False))
            else:
                prompt_tokens = 0
                completion_tokens = 0

            usage_total = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            if not done_reasons:
                done_reasons = ["stop"]

        except Exception as e:
            self.logger.error(f"Error during Lemonade GGUF API call: {e}")
            raise

        return responses, usage_total, done_reasons


    # ------------------------------------------------------------------
    # Lemonade specific helper APIs
    # ------------------------------------------------------------------
    def get_models(self) -> Dict[str, Any]:
        """Return models available on the server."""
        resp = self.session.get(f"{self.base_url}/models")
        resp.raise_for_status()
        return resp.json()

    def get_available_models(self) -> List[str]:
        """Return a list of model names available on the server."""
        models = self.get_models().get("data", [])
        return [model["id"] for model in models]
    
    def _ensure_model_available(self):
        """Ensure the specified model is available on the Lemonade server."""

        # Catch any connection issues when fetching available models
        # as that typically means the Lemonade server is not running.
        try:
            available_models = self.get_available_models()
        except requests.RequestException as e:
            msg = (f"Failed to fetch available models from Lemonade server."
                   f"Check if the Lemonade server is running")
            self.logger.error(msg)
            raise RuntimeError(msg)

        if self.model_name not in available_models:
            self.logger.info("Pulling model: %s", self.model_name)
            try:
                self.pull_model(self.model_name)
            except:
                msg = (f"Model '{self.model_name}' not found on Lemonade server and unable to pull.\n"
                    f"Available models: {available_models}")
                self.logger.error(msg)
                raise RuntimeError(msg)

    def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Download and register a model on the server."""
        resp = self.session.post(f"{self.base_url}/pull", json={"model_name": model_name})
        resp.raise_for_status()
        return resp.json()

    def load_model(
        self,
        *,
        model_name: Optional[str] = None,
        checkpoint: Optional[str] = None,
        recipe: Optional[str] = None,
        reasoning: Optional[bool] = None,
        mmproj: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Explicitly load a model into memory."""
        payload: Dict[str, Any] = {}
        if model_name:
            payload["model_name"] = model_name
        if checkpoint:
            payload["checkpoint"] = checkpoint
        if recipe:
            payload["recipe"] = recipe
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if mmproj:
            payload["mmproj"] = mmproj
        resp = self.session.post(f"{self.base_url}/load", json=payload)
        resp.raise_for_status()
        return resp.json()

    def unload_model(self) -> Dict[str, Any]:
        """Unload the currently loaded model."""
        resp = self.session.post(f"{self.base_url}/unload")
        resp.raise_for_status()
        return resp.json()

    def set_params(self, **params: Any) -> Dict[str, Any]:
        """Set generation parameters that persist across requests."""
        resp = self.session.post(f"{self.base_url}/params", json=params)
        resp.raise_for_status()
        return resp.json()

    def get_health(self) -> Dict[str, Any]:
        """Check the health of the server."""
        resp = self.session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def get_stats(self) -> Dict[str, Any]:
        """Return performance statistics from the last request."""
        resp = self.session.get(f"{self.base_url}/stats")
        resp.raise_for_status()
        return resp.json()