"""
Cliente LLM unificado para o pipeline agêntico.

Abstrai dois backends:
- Ollama (modelos locais: Llama 3.2 3B para SCB, Qwen2.5 7B para Planner)
- Azure OpenAI (GPT-4o como oracle e Synthesizer no regime API)

A escolha do backend é feita por configuração, não por lógica condicional
espalhada pelo código — facilitando a comparação dos regimes na dissertação.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx


class LLMBackend(str, Enum):
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"


@dataclass
class LLMConfig:
    backend: LLMBackend
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    # Azure OpenAI
    azure_endpoint: str = ""
    azure_api_key: str = ""
    azure_api_version: str = "2024-08-01-preview"
    azure_deployment: str = ""


class LLMClient:
    """
    Cliente unificado. Cada componente do pipeline (SCB, Planner,
    Query Generator, Synthesizer) instancia um LLMClient com sua
    própria config — permitindo regimes mistos (SCB local + oracle API).
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._http = httpx.Client(timeout=120.0)

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def chat(
        self,
        system: str,
        user: str,
        json_schema: dict | None = None,
    ) -> str:
        """
        Envia uma mensagem e retorna o texto da resposta.

        Args:
            system: system prompt
            user: mensagem do usuário
            json_schema: se fornecido, ativa structured output (Ollama)
                         ou json_object mode (Azure). Garante JSON válido.

        Returns:
            Texto bruto da resposta (JSON string quando json_schema fornecido)
        """
        if self.config.backend == LLMBackend.OLLAMA:
            return self._chat_ollama(system, user, json_schema)
        return self._chat_azure(system, user, json_schema)

    def chat_json(
        self,
        system: str,
        user: str,
        json_schema: dict | None = None,
    ) -> dict:
        """
        Versão tipada: chama chat() e parseia o JSON retornado.
        Levanta JSONDecodeError se o modelo não retornar JSON válido.
        """
        raw = self.chat(system, user, json_schema)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Tenta extrair JSON de resposta com texto ao redor
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _chat_ollama(
        self,
        system: str,
        user: str,
        json_schema: dict | None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        # Structured output: Ollama aceita JSON schema via "format"
        if json_schema:
            payload["format"] = json_schema

        resp = self._http.post(
            f"{self.config.ollama_base_url}/api/chat",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _chat_azure(
        self,
        system: str,
        user: str,
        json_schema: dict | None,
    ) -> str:
        endpoint = (
            f"{self.config.azure_endpoint}/openai/deployments/"
            f"{self.config.azure_deployment}/chat/completions"
            f"?api-version={self.config.azure_api_version}"
        )
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if json_schema:
            payload["response_format"] = {"type": "json_object"}

        resp = self._http.post(
            endpoint,
            json=payload,
            headers={
                "api-key": self.config.azure_api_key,
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Configurações pré-definidas para os regimes da dissertação
# ---------------------------------------------------------------------------

def make_local_light_configs() -> dict[str, LLMConfig]:
    """
    Regime local-light:
    - SCB: Llama 3.2 3B (configuração validada no artigo)
    - Planner + Query Generator + Synthesizer: Qwen2.5 7B
    """
    return {
        "scb": LLMConfig(
            backend=LLMBackend.OLLAMA,
            model="llama3.2:3b",
            temperature=0.0,
            max_tokens=512,
        ),
        "planner": LLMConfig(
            backend=LLMBackend.OLLAMA,
            model="qwen2.5:7b",
            temperature=0.0,
            max_tokens=1024,
        ),
        "query_generator": LLMConfig(
            backend=LLMBackend.OLLAMA,
            model="qwen2.5:7b",
            temperature=0.0,
            max_tokens=512,
        ),
        "synthesizer": LLMConfig(
            backend=LLMBackend.OLLAMA,
            model="qwen2.5:7b",
            temperature=0.0,
            max_tokens=1024,
        ),
    }


def make_oracle_configs() -> dict[str, LLMConfig]:
    """
    Regime oracle:
    - SCB: Llama 3.2 3B (mantém soberania na seleção)
    - Planner + Query Generator + Synthesizer: GPT-4o via Azure
    """
    azure_base = dict(
        backend=LLMBackend.AZURE_OPENAI,
        model="gpt-4o",
        temperature=0.0,
        max_tokens=1024,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        azure_api_key=os.getenv("AZURE_OPENAI_KEY", ""),
        azure_api_version="2024-08-01-preview",
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
    )
    return {
        "scb": LLMConfig(
            backend=LLMBackend.OLLAMA,
            model="llama3.2:3b",
            temperature=0.0,
            max_tokens=512,
        ),
        "planner": LLMConfig(**azure_base),
        "query_generator": LLMConfig(**azure_base),
        "synthesizer": LLMConfig(**azure_base),
    }
