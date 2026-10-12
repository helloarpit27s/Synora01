from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Everything configurable, nothing hardcoded.

    Model ids are settings rather than constants on purpose: the NIM catalogue
    changes faster than any release cycle, and swapping a model should never
    require a code change.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    nvidia_api_key: str = ""
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"

    nim_model_reasoning: str = "nvidia/llama-3.3-nemotron-super-49b-v1"
    nim_model_light: str = "meta/llama-3.1-8b-instruct"
    nim_model_vision: str = "meta/llama-3.2-90b-vision-instruct"
    nim_model_embed: str = "nvidia/nv-embedqa-e5-v5"
    nim_model_rerank: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    nim_model_guard: str = "nvidia/llama-3.1-nemoguard-8b-content-safety"

    nim_min_interval_s: float = 1.5
    nim_timeout_s: float = 45.0
    nim_max_retries: int = 3

    rag_top_k: int = 8
    rag_rerank_to: int = 4
    rag_min_score: float = 0.15

    allowed_origins: str = "http://localhost:5173"
    log_level: str = "INFO"

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def configured(self) -> bool:
        return bool(self.nvidia_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
