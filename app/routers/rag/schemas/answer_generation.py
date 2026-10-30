"""Structured model output for a grounded answer and its source references."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1)
    source_ids: list[Annotated[int, Field(strict=True, ge=1)]]
