"""External provider response envelopes shared by RAG services."""

from pydantic import BaseModel, Field


class OpenAIContent(BaseModel):
    type: str
    text: str = ""


class OpenAIOutput(BaseModel):
    type: str
    content: list[OpenAIContent] = Field(default_factory=list)


class OpenAIResponse(BaseModel):
    status: str
    output: list[OpenAIOutput]


class GeminiContent(BaseModel):
    type: str
    text: str = ""


class GeminiStep(BaseModel):
    type: str
    content: list[GeminiContent] = Field(default_factory=list)


class GeminiResponse(BaseModel):
    status: str
    steps: list[GeminiStep]


class ClaudeContent(BaseModel):
    type: str
    text: str = ""


class ClaudeStopDetails(BaseModel):
    type: str


class ClaudeResponse(BaseModel):
    stop_reason: str
    stop_details: ClaudeStopDetails | None = None
    content: list[ClaudeContent]
