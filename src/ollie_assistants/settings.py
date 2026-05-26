from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Ollive Assistants"
    openai_api_key: str | None = None
    frontier_model: str = "gpt-4.1-mini"
    openai_input_usd_per_1m_tokens: float = 0.40
    openai_output_usd_per_1m_tokens: float = 1.60
    oss_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    oss_served_model_name: str = "oss-assistant"
    oss_provider: str = "openai_compatible"
    oss_base_url: str | None = None
    oss_bearer_token: str | None = None
    oss_modal_secret_name: str = "ollie-oss-secrets"
    oss_modal_app_name: str = "ollie-oss-vllm"
    oss_modal_gpu: str = "L4"
    oss_modal_scaledown_seconds: int = 600
    oss_modal_max_inputs: int = 20
    oss_modal_max_model_len: int = 4096
    modal_l4_usd_per_second: float = 0.000222
    modal_l4_usd_per_hour: float = 0.80
    frontier_provider: str = "openai"
    chat_enabled: bool = True
    evals_enabled: bool = True
    analytics_enabled: bool = True
    traces_path: Path = Path(".runs/traces.jsonl")
    observability_db_path: Path = Path(".runs/observability.sqlite")
    api_base_url: str = "http://127.0.0.1:8000"
    memory_turns: int = 8
    max_prompt_chars: int = 4_000
    max_new_tokens: int = 256


@lru_cache
def get_settings() -> Settings:
    return Settings()
