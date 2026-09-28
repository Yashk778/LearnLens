from pydantic import BaseModel,Field
from typing import Any


class Document(BaseModel):

    source_type:str
    title:str
    text:str
    url:str | None = None
    metadata: dict[str,Any] = Field(default_factory=dict)

