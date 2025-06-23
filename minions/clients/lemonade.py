import os
import logging
from typing import Any, Dict, List, Optional, Tuple
import requests

from minions.clients.openai import OpenAIClient
from minions.usage import Usage
from pydantic import BaseModel


try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

import asyncio

class LemonadeClient(OpenAIClient):
    """
    Uses Lemonade server API to run local clients.
    """

    def __init__(
        self,
        model_name: str = "Llama-3.2-3B-Instruct-Hybrid",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        base_url: Optional[str] = None,
        structured_output_schema: Optional[BaseModel] = None,
        use_async: bool = False,
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
        self.logger.setLevel(logging.INFO)
        self.structured_output_schema = structured_output_schema
        self.use_async = use_async

        # Lemonade only supports GGUF models for structured output schemas for now
        if self.structured_output_schema and not "GGUF" in self.model_name.upper():
            raise TypeError(f"The model used for Minions and Minions-CUA must be GGUF. A GGUF model was not used.")
        # Validate Lemonade server connection and model
        self._ensure_model_available()

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """
        Main chat method: dispatches to schat or achat depending on use_async.
        """
        if self.use_async:
            return self.achat(messages, **kwargs)
        else:
            return self.schat(messages, **kwargs)

    def schat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """
        Synchronous chat: used for Minion. This assumes structured_output_schema is not used here.
        """
        assert len(messages) > 0, "Messages cannot be empty."
        if self.structured_output_schema is not None:
            raise TypeError("Lemonade does not currently support this configuration. Forced output schema isn't available in synchronous chats.")
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **kwargs,
        }
        response = self.session.post(
            f"{self.base_url.rstrip('/api/v1')}/api/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        response_data = response.json()
        choices = response_data.get("choices", [])
        responses = [choice["message"]["content"] for choice in choices if "message" in choice]
        usage = Usage()
        usage += Usage(
            prompt_tokens=response_data.get('usage', {}).get('prompt_tokens', 0),
            completion_tokens=response_data.get('usage', {}).get('completion_tokens', 0),
        )
        done_reason = [choice.get("finish_reason", "stop") for choice in choices]
        return responses, usage, done_reason

    def achat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """
        Asynchronous chat: uses response_format if forced schema. This is used for Minions.
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required for async Lemonade client. Please install with: pip install aiohttp")
        async def run_async():
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs,
            }
            if self.structured_output_schema:
                payload["response_format"] = {
                    "type": "json_object",
                    "schema": self.structured_output_schema.model_json_schema()
                }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url.rstrip('/api/v1')}/api/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    response.raise_for_status()
                    response_data = await response.json()
            choices = response_data.get("choices", [])
            responses = [choice["message"]["content"] for choice in choices if "message" in choice]
            usage = Usage()
            usage += Usage(
                prompt_tokens=response_data.get('usage', {}).get('prompt_tokens', 0),
                completion_tokens=response_data.get('usage', {}).get('completion_tokens', 0),
            )
            done_reason = [choice.get("finish_reason", "stop") for choice in choices]
            return responses, usage, done_reason
        return asyncio.run(run_async())
        
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