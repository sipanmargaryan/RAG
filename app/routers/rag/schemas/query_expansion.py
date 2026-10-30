"""Structured query expansion output and provider response schemas."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


class ExpandedQueries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    ] = Field(min_length=4, max_length=4)

    @field_validator("queries")
    @classmethod
    def distinct_queries(cls, queries: list[str]) -> list[str]:
        if len({" ".join(query.casefold().split()) for query in queries}) != 4:
            raise ValueError("Expanded queries must be distinct")
        return queries
