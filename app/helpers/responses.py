from pydantic import BaseModel


class APIResponse[Data](BaseModel):
    message: str
    data: Data | None = None
