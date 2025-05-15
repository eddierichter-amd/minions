# lemonade_client.py
import requests
import logging
import json
from typing import Any, Dict, List, Tuple, Optional, Generator
from threading import Thread
from minions.usage import Usage

class LemonadeClient:
    def __init__(
        self,
        model_name: str = "Llama-3.2-1B-Instruct-Hybrid",
        base_url: str = "http://localhost:8000",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        streaming: bool = False,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming
        self.logger = logging.getLogger("LemonadeClient")
        self.logger.setLevel(logging.INFO)
        
        # Validate server connection and model availability
        self._ensure_model_available()

    @staticmethod
    def get_available_models(base_url: str = "http://localhost:8000") -> List[str]:
        try:
            response = requests.get(f"{base_url}/api/v0/models", timeout=10)
            response.raise_for_status()
            return [model["id"] for model in response.json().get("data", [])]
        except Exception as e:
            logging.error(f"Failed to get Lemonade model list: {e}")
            return []

    def _ensure_model_available(self):
        available_models = self.get_available_models(self.base_url)
        if self.model_name not in available_models:
            msg = (f"Model '{self.model_name}' not found on Lemonade server.\n"
                   f"Available models: {available_models}")
            self.logger.error(msg)
            raise RuntimeError(msg)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        streaming: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        use_streaming = self.streaming if streaming is None else streaming
        return self.schat(messages, **kwargs) if use_streaming else self.bchat(messages, **kwargs)

    def bchat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        try:
            response = requests.post(
                f"{self.base_url}/api/v0/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            return self._parse_response(response.json())
        except Exception as e:
            self.logger.error(f"Blocking API Error: {str(e)}")
            return ([""], Usage(), ["error"])

    def schat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        try:
            response = requests.post(
                f"{self.base_url}/api/v0/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": True
                },
                stream=True,
                timeout=30
            )
            response.raise_for_status()
            return self._parse_stream(response)
        except Exception as e:
            self.logger.error(f"Streaming API Error: {str(e)}")
            return ([""], Usage(), ["error"])

    def _parse_response(self, data: dict) -> Tuple[List[str], Usage, List[str]]:
        choices = data.get("choices", [{}])
        usage_data = data.get("usage", {})
        return (
            [choice.get("message", {}).get("content", "") for choice in choices],
            Usage(usage_data.get("prompt_tokens", 0), usage_data.get("completion_tokens", 0)),
            [choice.get("finish_reason", "stop") for choice in choices]
        )

    def _parse_stream(self, response: requests.Response) -> Tuple[List[str], Usage, List[str]]:
        full_response = ""
        finish_reasons = []
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode("utf-8").lstrip("data: "))
                if content := chunk["choices"][0]["delta"].get("content"):
                    full_response += content
                if finish_reason := chunk["choices"][0].get("finish_reason"):
                    finish_reasons.append(finish_reason)
        return (
            [full_response],
            Usage(0, len(full_response.split())),  # Estimate completion tokens
            finish_reasons or ["stop"]
        )
