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


settings = AppSettings()
