"""FastAPI service for the COVID-19 FAQ chatbot.

    uvicorn src.api:app --reload

Loads the knowledge base, the encoder named in ``artifacts/retriever_config.json``
and the precomputed embeddings **once at startup**, then answers from memory.

The service imports the same ``src/`` modules the notebook does, so
the behaviour measured during development is the behaviour served here — not a
reimplementation that drifts.

Every answer is a stored KB answer with its source, trust tier and a medical
disclaimer, or an explicit abstention. Nothing is generated.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.evaluate import DATA_DIR
from src.index import ARTIFACTS_DIR, load_or_build_index
from src.retriever import BiEncoderRetriever, load_kb

logger = logging.getLogger("covicare.api")

MAX_QUESTION_CHARS = 500

DISCLAIMER = (
    "This is general information, not medical advice. For personal or urgent concerns, "
    "consult a healthcare professional or official sources (WHO/CDC)."
)

# Fixed text for the abstention path. Deliberately offers no FAQ content and no
# source: a low-confidence match must not be dressed up as an attributed answer
# .
ABSTENTION_ANSWER = (
    "I don't have a vetted answer for that. This assistant only covers COVID-19 "
    "questions from a fixed set of public-health FAQs. For reliable guidance, see the "
    "World Health Organization (who.int) or the CDC (cdc.gov)."
)


class Settings(BaseSettings):
    """Runtime configuration, overridable by environment variable."""

    model_config = SettingsConfigDict(env_prefix="COVICARE_", env_file=".env", extra="ignore")

    # Explicit origins, never "*" with credentials. Defaults cover
    # local Flutter web and desktop development.
    cors_origins: list[str] = ["http://localhost:8080", "http://127.0.0.1:8080"]
    cors_allow_credentials: bool = False

    rate_limit: str = "30/minute"

    # Health questions are sensitive. At INFO the service records a salted hash
    # and the length, never the text; set this only on a machine where seeing
    # real queries is acceptable.
    log_queries: bool = False
    log_salt: str = "covicare"

    top_k: int = 5


settings = Settings()


# --------------------------------------------------------------------------
# Request / response models
# --------------------------------------------------------------------------
class PredictRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)

    @field_validator("question")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        """`min_length` alone accepts "   ", which is not a question."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must contain non-whitespace characters")
        return stripped


class PredictResponse(BaseModel):
    """The /predict response contract.

    `source`, `trust`, `url` and `matched_question` are null on abstention so a
    client cannot accidentally attribute the safe message to a real FAQ entry.
    """

    answer: str
    matched_question: str | None = None
    source: str | None = None
    trust: str | None = None
    url: str | None = None
    score: float
    abstained: bool
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    status: str
    encoder: str
    corpus_size: int
    threshold: float


# --------------------------------------------------------------------------
# The retrieval service
# --------------------------------------------------------------------------
@dataclass
class ChatbotService:
    """Holds the loaded model and answers questions.

    Kept free of FastAPI types so it can be tested directly, and so the notebook
    could import it unchanged.
    """

    retriever: BiEncoderRetriever
    kb: pd.DataFrame
    tau: float
    encoder_name: str

    @classmethod
    def load(cls, data_dir: Path = DATA_DIR, artifacts_dir: Path = ARTIFACTS_DIR) -> ChatbotService:
        """Load KB, config and embeddings. Called once, at startup."""
        import json

        config_path = artifacts_dir / "retriever_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_path} not found. Run `python -m src.evaluate --tune-threshold` "
                f"(after `src.download`, `src.prep` and `src.index`)."
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        kb = load_kb(data_dir)
        index = load_or_build_index(
            kb, model_name=config["encoder"], field=config.get("field", "question")
        )
        return cls(
            retriever=BiEncoderRetriever(index),
            kb=kb.set_index("id", drop=False),
            tau=float(config["tau"]),
            encoder_name=config["encoder"],
        )

    def answer(self, question: str, top_k: int = 5) -> PredictResponse:
        """Retrieve, threshold, and build the response.

        Below τ the bot abstains rather than returning its best guess: a
        confident wrong answer is the failure mode that matters here.
        """
        hits = self.retriever.search(question, top_k=top_k)
        if not hits:
            return PredictResponse(answer=ABSTENTION_ANSWER, score=0.0, abstained=True)

        kb_id, score = hits[0]
        if score < self.tau:
            return PredictResponse(answer=ABSTENTION_ANSWER, score=round(score, 4), abstained=True)

        row = self.kb.loc[kb_id]
        return PredictResponse(
            answer=row["answer"],
            matched_question=row["question"],
            source=row["source"],
            trust=row["trust"],
            url=row["url"] or None,
            score=round(score, 4),
            abstained=False,
        )


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> Iterator[None]:
    """Load the model once; keep it in memory for the process lifetime."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    started = time.perf_counter()
    app.state.service = ChatbotService.load()
    logger.info(
        "startup encoder=%s corpus=%d tau=%.4f load_seconds=%.1f",
        app.state.service.encoder_name,
        len(app.state.service.kb),
        app.state.service.tau,
        time.perf_counter() - started,
    )
    yield
    app.state.service = None


app = FastAPI(
    title="Covicare",
    version="0.1.0",
    description=(
        "Retrieval-only COVID-19 FAQ chatbot. Returns a stored, source-attributed answer "
        "or abstains. Never generates medical text."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Rate limiting. slowapi keys on client IP and returns 429 past the budget.
#
# The decorator has to sit *inside* the route decorator, below @app.post: FastAPI
# registers the function when @app.post runs, so wrapping the name afterwards
# changes nothing and the limit silently never applies. Only /predict is limited
# -- /health is what a load balancer polls.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def get_service(request: Request) -> ChatbotService:
    service = getattr(request.app.state, "service", None)
    if service is None:  # pragma: no cover - only reachable outside lifespan
        raise RuntimeError("service not loaded")
    return service


def _query_fingerprint(question: str) -> str:
    """A stable id for a question that is not the question.

    Lets us spot repeated queries and debug a specific report without writing
    someone's health question to disk.
    """
    digest = hashlib.sha256((settings.log_salt + question).encode("utf-8")).hexdigest()
    return digest[:12]


@app.get("/health", response_model=HealthResponse)
def health(service: ChatbotService = Depends(get_service)) -> HealthResponse:
    """Readiness check. Reports what is actually loaded, not just liveness."""
    return HealthResponse(
        status="ok",
        encoder=service.encoder_name,
        corpus_size=len(service.kb),
        threshold=service.tau,
    )


@app.get("/")
def root(service: ChatbotService = Depends(get_service)) -> dict:
    return {
        "service": "Covicare",
        "description": "Retrieval-only COVID-19 FAQ chatbot.",
        "endpoints": {"predict": "POST /predict", "health": "GET /health", "docs": "GET /docs"},
        "corpus_size": len(service.kb),
        "disclaimer": DISCLAIMER,
    }


# Defined with `def`, not `async def`: encoding a query is blocking CPU work, and
# an async handler would run it on the event loop and stall every other request.
# Starlette dispatches sync handlers to a threadpool instead.
@app.post("/predict", response_model=PredictResponse)
@limiter.limit(settings.rate_limit)
def predict(
    request: Request,
    payload: PredictRequest,
    service: ChatbotService = Depends(get_service),
) -> PredictResponse:
    started = time.perf_counter()
    response = service.answer(payload.question, top_k=settings.top_k)
    elapsed_ms = (time.perf_counter() - started) * 1000

    logger.info(
        "predict query=%s len=%d score=%.4f abstained=%s matched=%s ms=%.1f",
        _query_fingerprint(payload.question),
        len(payload.question),
        response.score,
        response.abstained,
        response.matched_question is not None,
        elapsed_ms,
    )
    if settings.log_queries:
        logger.debug("predict text=%r", payload.question)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a generic error.

    Returning `str(exc)` to the caller leaks filesystem paths and internals.
    The detail belongs in the log, not the response.
    """
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})
