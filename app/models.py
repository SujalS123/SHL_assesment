# models.py — Request & Response schemas
# These must match the spec exactly — the automated evaluator checks them.

from pydantic import BaseModel
from typing import Optional


class Message(BaseModel):
    role: str        # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str   # e.g. "K" (knowledge), "P" (personality), "A" (ability)


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = []
    end_of_conversation: bool = False
