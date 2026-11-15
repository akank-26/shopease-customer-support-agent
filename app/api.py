
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.combined_agent import answer_customer


logger = logging.getLogger(__name__)

app = FastAPI(
    title="ShopEase AI Support API",
    version="0.1.0",
)

# LOCAL DEMO ONLY.
# Never use a shared customer identity in production.
DEMO_CUSTOMER_ID = "CUST-001"


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
    )


class ChatResponse(BaseModel):
    answer: str
    citations: dict
    tool_used: bool


@app.get("/health")
def health():
    """Check whether the API process is running."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Answer a support question using our agent."""

    if not request.question.strip():
        raise HTTPException(
            status_code=422,
            detail="Question cannot be blank",
        )

    try:
        result = answer_customer(
            question=request.question,
            authenticated_customer_id=DEMO_CUSTOMER_ID,
        )

        return ChatResponse(
            answer=result["answer"],
            citations=result.get("citations", {}),
            tool_used=result.get("tool_used", False),
        )

    except Exception:
        logger.exception("Support agent request failed")

        raise HTTPException(
            status_code=503,
            detail="Support assistant temporarily unavailable",
        )
