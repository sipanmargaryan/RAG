"""Async SQLAlchemy sessions shared by endpoints and tests."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

from fastapi import Depends
from sqlalchemy import DateTime, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings

engine = create_async_engine(
    settings.database.database_url,
    pool_pre_ping=True,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    pool_timeout=settings.database.pool_timeout,
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class BaseDBModel(Base):
    """Optional timestamp base for application models."""

    __abstract__ = True

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


@asynccontextmanager
async def database_lifespan() -> AsyncGenerator[None]:
    """Dispose the worker's shared engine when the application shuts down."""
    try:
        yield
    finally:
        await engine.dispose()


async def get_session() -> AsyncGenerator[AsyncSession]:
    """Provide a separate session; callers explicitly commit successful writes."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


AsyncSessionDep = Annotated[AsyncSession, Depends(get_session)]
