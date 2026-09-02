"""
Base agent class. All agents in GovBench-Med inherit from this.
Uses the Ollama local inference API so everything runs on your machine
with no API keys and no cost.
"""
import time
import json
import requests
from dataclasses import dataclass, field
from typing import Optional


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 600  # 10 minutes for CPU cold starts


@dataclass
class AgentResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency: float          # seconds
    model: str
    agent_role: str
    raw: dict = field(default_factory=dict)


class BaseAgent:
    def __init__(self, model: str, role: str, system_prompt: str, temperature: float = 0.0):
        self.model = model
        self.role = role
        self.system_prompt = system_prompt
        self.temperature = temperature

    def call(self, user_message: str, context: Optional[str] = None) -> AgentResponse:
        prompt = f"{self.system_prompt}\n\n"
        if context:
            prompt += f"Context:\n{context}\n\n"
        prompt += f"Input:\n{user_message}"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "seed": 0},
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Ollama call failed for {self.role}: {e}")
        latency = time.perf_counter() - t0

        return AgentResponse(
            text=data.get("response", ""),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency=latency,
            model=self.model,
            agent_role=self.role,
            raw=data,
        )
