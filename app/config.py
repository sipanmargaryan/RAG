from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


class QdrantSettings(BaseSettings):
    """Qdrant settings."""

    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        env_file=".env",
        extra="ignore",
    )

    host: str = "localhost"
    port: int = Field(default=6333)
    scheme: Literal["http", "https"] = "http"
    api_key: SecretStr | None = Field(default=None, repr=False)
    timeout_seconds: int = Field(default=10, gt=0)
    max_connections: int = Field(default=100, gt=0)
    max_keepalive_connections: int = Field(default=20, ge=0)
    keepalive_expiry_seconds: float = Field(default=5, ge=0)
    collection_name: str = "pdf_chunks_minilm_l6_v2"
    batch_size: int = Field(default=64, gt=0)

    @property
    def qdrant_url(self) -> HttpUrl:
        return HttpUrl.build(scheme=self.scheme, host=self.host, port=self.port)


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_", env_file=".env", extra="ignore"
    )

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_size: int = Field(default=384, gt=0)
    batch_size: int = Field(default=32, gt=0)


class RerankingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RERANKING_", env_file=".env", extra="ignore"
    )

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = Field(default=16, gt=0)


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENAI_", env_file=".env", extra="ignore"
    )

    api_key: SecretStr | None = Field(default=None, repr=False)
    model: str = "gpt-4.1-mini"
    timeout_seconds: float = Field(default=30, gt=0)


class GeminiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GEMINI_", env_file=".env", extra="ignore"
    )

    api_key: SecretStr | None = Field(default=None, repr=False)
    model: str = "gemini-3.8-flash"
    timeout_seconds: float = Field(default=30, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)


class ClaudeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLAUDE_", env_file=".env", extra="ignore"
    )

    api_key: SecretStr | None = Field(default=None, repr=False)
    model: str = "claude-haiku-5-5"
    api_version: str = "2026-10-21"
    workspace_id: str | None = None
    timeout_seconds: float = Field(default=60, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)


class QueryExpansionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QUERY_EXPANSION_", env_file=".env", extra="ignore"
    )

    provider: Literal["openai", "gemini", "claude"]


class AnswerGenerationSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ANSWER_", env_file=".env", extra="ignore"
    )

    provider: Literal["openai", "gemini", "claude"]
    max_output_tokens: int = Field(default=2048, gt=0)


class MinioSettings(BaseSettings):
    """Object storage settings; default credentials are for local development."""

    model_config = SettingsConfigDict(
        env_prefix="MINIO_", env_file=".env", extra="ignore"
    )

    endpoint: str = "localhost:9000"
    access_key: SecretStr = SecretStr("minioadmin")
    secret_key: SecretStr = SecretStr("minioadmin")
    secure: bool = False
    bucket_name: str = "processed-pdfs"
    region: str = "us-east-1"
    timeout_seconds: float = Field(default=30, gt=0)


class DatabaseSettings(BaseSettings):
    """PostgreSQL settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_prefix="POSTGRES_",
    )

    host: str
    port: int = 5432
    user: str
    password: str
    database: str
    pool_size: int = Field(default=5, gt=0)
    max_overflow: int = Field(default=0, ge=0)
    pool_timeout: float = Field(default=30, gt=0)

    @property
    def database_url(self) -> URL:
        return URL.create(
            drivername="postgresql+asyncpg",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )


class AppSettings(BaseModel):
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    minio: MinioSettings = Field(default_factory=MinioSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranking: RerankingSettings = Field(default_factory=RerankingSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    claude: ClaudeSettings = Field(default_factory=ClaudeSettings)
    query_expansion: QueryExpansionSettings = Field(
        default_factory=QueryExpansionSettings
    )
    answer: AnswerGenerationSettings = Field(default_factory=AnswerGenerationSettings)


settings = AppSettings()
