# main.py — FastAPI app with /health and /chat endpoints.
#
# /health  → simple readiness check (evaluator polls this on cold start)
# /chat    → stateless multi-turn conversation; returns reply + structured recs

import re
import json
from fastapi import FastAPI, HTTPException
from app.models import ChatRequest, ChatResponse, Recommendation
from app.agent import run_agent

app = FastAPI(title="SHL Assessment Recommender")


# ---------------------------------------------------------------------------
# Health check — evaluator waits up to 2 min for this to return 200
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages list cannot be empty")

    # Convert Pydantic objects to plain dicts for the agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Run the agent
    reply_text = run_agent(messages)

    # Parse structured recommendations from the agent's reply
    recommendations = _extract_recommendations(reply_text)

    # Detect end-of-conversation signal
    end_flag = "end_of_conversation: true" in reply_text.lower()

    # Clean the reply — strip the end_of_conversation marker if present
    clean_reply = re.sub(r"end_of_conversation:\s*(true|false)", "", reply_text, flags=re.IGNORECASE).strip()

    return ChatResponse(
        reply=clean_reply,
        recommendations=recommendations,
        end_of_conversation=end_flag,
    )


# ---------------------------------------------------------------------------
# Helper: extract structured recommendations from agent reply text
# ---------------------------------------------------------------------------

# Matches lines like: "Java 8 (New) | K | https://www.shl.com/..."
_REC_PATTERN = re.compile(
    r"(?P<name>[^\|]+)\s*\|\s*(?P<test_type>[^\|]+)\s*\|\s*(?P<url>https://[^\s]+)",
    re.IGNORECASE,
)

# Also try to parse JSON blocks the agent might emit
_JSON_BLOCK = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def _extract_recommendations(text: str) -> list[Recommendation]:
    """
    Try two parsing strategies:
    1. Pipe-delimited lines:  Name | TestType | URL
    2. JSON block embedded in the reply

    Returns an empty list when no recommendations are detected (clarifying turn).
    """
    recs: list[Recommendation] = []

    # Strategy 1: pipe-delimited
    for match in _REC_PATTERN.finditer(text):
        raw_type = match.group("test_type").strip().upper()
        # If model expanded 'K' to 'Knowledge Test', take just the first letter
        test_type = raw_type[0] if raw_type else ""
        
        recs.append(Recommendation(
            name=match.group("name").strip(),
            test_type=test_type,
            url=match.group("url").strip(),
        ))

    if recs:
        return recs[:10]   # cap at 10 per spec

    # Strategy 2: JSON block
    json_match = _JSON_BLOCK.search(text)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if isinstance(data, list):
                for item in data[:10]:
                    if all(k in item for k in ("name", "url", "test_type")):
                        recs.append(Recommendation(**item))
        except (json.JSONDecodeError, TypeError):
            pass

    return recs
