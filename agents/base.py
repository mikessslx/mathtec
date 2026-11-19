import asyncio
import logging
from typing import Optional

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from config import ModelConfig

logger = logging.getLogger(__name__)


class LLMAgent:
    """Unified async LLM client supporting OpenAI, Anthropic, DeepSeek, and Google."""

    def __init__(self, config: ModelConfig, role: str, system_prompt: str = ""):
        self.config = config
        self.role = role
        self.system_prompt = system_prompt
        self._client = None

    def _get_openai_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    def _get_anthropic_client(self) -> AsyncAnthropic:
        return AsyncAnthropic(api_key=self.config.api_key)

    async def call(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        if self.config.provider in ("openai", "deepseek", "google"):
            return await self._call_openai_compatible(prompt, temp, tokens)
        elif self.config.provider == "anthropic":
            return await self._call_anthropic(prompt, temp, tokens)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    async def _call_openai_compatible(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        client = self._get_openai_client()
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    async def _call_anthropic(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        client = self._get_anthropic_client()
        kwargs = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.system_prompt:
            kwargs["system"] = self.system_prompt

        response = await client.messages.create(**kwargs)
        return response.content[0].text.strip()

    async def call_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        temperature: Optional[float] = None,
    ) -> str:
        last_error = None
        for attempt in range(max_retries):
            try:
                return await self.call(prompt, temperature=temperature)
            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    f"[{self.role}] Attempt {attempt+1}/{max_retries} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                await asyncio.sleep(wait)
        raise RuntimeError(
            f"[{self.role}] All {max_retries} attempts failed. Last error: {last_error}"
        )

    def __repr__(self) -> str:
        return f"LLMAgent(role={self.role}, model={self.config.model})"
