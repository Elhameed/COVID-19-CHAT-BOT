"""Tests for the FastAPI service (PRD §19).

Two layers. Most tests run against a stub service so the contract, validation
and abstention paths are checked in milliseconds. A smaller set loads the real
encoder and corpus, because the guarantee that matters most — every answer is
verbatim KB text — can only be checked against the real knowledge base.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import (
    ABSTENTION_ANSWER,
    DISCLAIMER,
    MAX_QUESTION_CHARS,
    ChatbotService,
    app,
    get_service,
    limiter,
)
from src.evaluate import DATA_DIR


class _StubRetriever:
    """Returns a scripted (kb_id, score) so thresholding can be driven exactly."""

    name = "stub"

    def __init__(self, hits: list[tuple[int, float]]) -> None:
        self.hits = hits

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        return self.hits[:top_k]


def _stub_service(hits: list[tuple[int, float]], tau: float = 0.7) -> ChatbotService:
    kb = pd.DataFrame(
        {
            "id": [1, 2],
            "question": ["How does COVID-19 spread?", "Should I wear a mask?"],
            "answer": ["Mainly person to person, within 6 feet.", "Yes, in public settings."],
            "source": ["World Health Organization", "WikiHow"],
            "url": ["https://who.int/faq", ""],
            "trust": ["official", "community"],
        }
    ).set_index("id", drop=False)
    return ChatbotService(
        retriever=_StubRetriever(hits), kb=kb, tau=tau, encoder_name="stub-encoder"
    )


def _client(service: ChatbotService) -> TestClient:
    """A client whose service is stubbed.

    Deliberately NOT used as a context manager: that would run the lifespan and
    load the real encoder and embeddings (~6s), for a service this immediately
    overrides. Skipping startup is what keeps the API suite fast.
    """
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


@pytest.fixture(autouse=True)
def _no_rate_limit() -> Iterator[None]:
    """Disable the limiter except where a test opts in."""
    limiter.enabled = False
    limiter.reset()
    yield
    limiter.enabled = True


@pytest.fixture
def client_confident() -> Iterator[TestClient]:
    yield _client(_stub_service([(1, 0.91)]))
    app.dependency_overrides.clear()


@pytest.fixture
def client_unsure() -> Iterator[TestClient]:
    yield _client(_stub_service([(1, 0.42)]))
    app.dependency_overrides.clear()


class TestHealth:
    def test_reports_what_is_loaded_not_just_liveness(self, client_confident: TestClient) -> None:
        body = client_confident.get("/health").json()
        assert body["status"] == "ok"
        assert body["encoder"] == "stub-encoder"
        assert body["corpus_size"] == 2
        assert body["threshold"] == pytest.approx(0.7)

    def test_root_advertises_the_endpoints(self, client_confident: TestClient) -> None:
        body = client_confident.get("/").json()
        assert "predict" in body["endpoints"]
        assert body["disclaimer"] == DISCLAIMER


class TestPredictContract:
    """PRD §10.2 — the exact shape the Flutter client depends on."""

    def test_response_has_every_contract_field(self, client_confident: TestClient) -> None:
        body = client_confident.post("/predict", json={"question": "how does covid spread?"}).json()
        assert set(body) == {
            "answer",
            "matched_question",
            "source",
            "trust",
            "url",
            "score",
            "abstained",
            "disclaimer",
        }

    def test_returns_the_stored_answer_with_attribution(self, client_confident: TestClient) -> None:
        body = client_confident.post("/predict", json={"question": "spread?"}).json()
        assert body["answer"] == "Mainly person to person, within 6 feet."
        assert body["matched_question"] == "How does COVID-19 spread?"
        assert body["source"] == "World Health Organization"
        assert body["trust"] == "official"
        assert body["url"] == "https://who.int/faq"
        assert body["abstained"] is False
        assert body["score"] == pytest.approx(0.91)

    def test_disclaimer_is_always_present(self, client_confident: TestClient) -> None:
        """Hard constraint #3: no answer ships without it."""
        body = client_confident.post("/predict", json={"question": "spread?"}).json()
        assert body["disclaimer"] == DISCLAIMER
        assert "not medical advice" in body["disclaimer"]

    def test_preserves_numbers_in_the_served_answer(self, client_confident: TestClient) -> None:
        """Hard constraint #5 reaches all the way to the response body."""
        body = client_confident.post("/predict", json={"question": "spread?"}).json()
        assert "6 feet" in body["answer"]

    def test_empty_url_becomes_null_rather_than_empty_string(self) -> None:
        body = (
            _client(_stub_service([(2, 0.95)])).post("/predict", json={"question": "mask?"}).json()
        )
        app.dependency_overrides.clear()
        assert body["url"] is None
        assert body["trust"] == "community"


class TestAbstention:
    """PRD §7.5 — below τ the bot declines instead of guessing."""

    def test_abstains_below_the_threshold(self, client_unsure: TestClient) -> None:
        body = client_unsure.post("/predict", json={"question": "what's the weather?"}).json()
        assert body["abstained"] is True
        assert body["answer"] == ABSTENTION_ANSWER

    def test_abstention_carries_no_misleading_attribution(self, client_unsure: TestClient) -> None:
        """The dangerous failure would be a safe message wearing a WHO badge."""
        body = client_unsure.post("/predict", json={"question": "weather?"}).json()
        assert body["source"] is None
        assert body["trust"] is None
        assert body["url"] is None
        assert body["matched_question"] is None

    def test_abstention_still_carries_the_disclaimer(self, client_unsure: TestClient) -> None:
        body = client_unsure.post("/predict", json={"question": "weather?"}).json()
        assert body["disclaimer"] == DISCLAIMER

    def test_abstention_points_at_official_sources(self, client_unsure: TestClient) -> None:
        body = client_unsure.post("/predict", json={"question": "weather?"}).json()
        assert "who.int" in body["answer"] and "cdc.gov" in body["answer"]

    def test_reports_the_score_it_declined_on(self, client_unsure: TestClient) -> None:
        body = client_unsure.post("/predict", json={"question": "weather?"}).json()
        assert body["score"] == pytest.approx(0.42)

    def test_no_hits_at_all_abstains(self) -> None:
        body = _client(_stub_service([])).post("/predict", json={"question": "???"}).json()
        app.dependency_overrides.clear()
        assert body["abstained"] is True
        assert body["score"] == 0.0

    def test_score_exactly_at_tau_is_answered(self) -> None:
        """The threshold is inclusive; stated so the boundary can't drift silently."""
        body = (
            _client(_stub_service([(1, 0.7)], tau=0.7))
            .post("/predict", json={"question": "spread?"})
            .json()
        )
        app.dependency_overrides.clear()
        assert body["abstained"] is False


class TestValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            {"question": ""},
            {"question": "   "},
            {"question": "\n\t "},
            {"question": "x" * (MAX_QUESTION_CHARS + 1)},
            {},
            {"question": None},
            {"question": 42},
            {"wrong_field": "hello"},
        ],
    )
    def test_rejects_malformed_input(self, client_confident: TestClient, payload: dict) -> None:
        assert client_confident.post("/predict", json=payload).status_code == 422

    def test_accepts_a_question_at_the_length_cap(self, client_confident: TestClient) -> None:
        payload = {"question": "a" * MAX_QUESTION_CHARS}
        assert client_confident.post("/predict", json=payload).status_code == 200

    def test_surrounding_whitespace_is_trimmed(self, client_confident: TestClient) -> None:
        assert (
            client_confident.post(
                "/predict", json={"question": "  how does covid spread?  "}
            ).status_code
            == 200
        )


class TestErrorHandling:
    def test_internal_errors_do_not_leak_details(self) -> None:
        """The previous implementation returned str(exc) to the caller, exposing
        filesystem paths. The detail belongs in the log."""

        class _Exploding:
            name = "boom"

            def search(self, query: str, top_k: int = 10):
                raise RuntimeError("secret path C:/Users/DELL/private/model")

        service = _stub_service([])
        service.retriever = _Exploding()
        app.dependency_overrides[get_service] = lambda: service
        response = TestClient(app, raise_server_exceptions=False).post(
            "/predict", json={"question": "spread?"}
        )
        app.dependency_overrides.clear()

        assert response.status_code == 500
        body = response.text
        assert "secret path" not in body
        assert "C:/Users" not in body


class TestCors:
    def test_wildcard_origin_is_not_allowed(self, client_confident: TestClient) -> None:
        """PRD §10.3 forbids `*`; the old service used it with credentials."""
        response = client_confident.options(
            "/predict",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.headers.get("access-control-allow-origin") != "*"

    def test_configured_origin_is_allowed(self, client_confident: TestClient) -> None:
        response = client_confident.options(
            "/predict",
            headers={
                "Origin": "http://localhost:8080",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:8080"


class TestRateLimit:
    def test_excess_requests_are_rejected(self) -> None:
        """Regression guard: the limiter decorator must sit inside @app.post, or
        FastAPI registers the undecorated function and the limit never applies."""
        limiter.enabled = True
        limiter.reset()
        client = _client(_stub_service([(1, 0.91)]))
        try:
            codes = [
                client.post("/predict", json={"question": f"q{i}"}).status_code for i in range(40)
            ]
        finally:
            app.dependency_overrides.clear()
            limiter.reset()
            limiter.enabled = False

        assert 429 in codes, "rate limiting is not engaging"
        assert codes.count(200) == 30

    def test_health_is_not_rate_limited(self) -> None:
        """A load balancer polls /health; limiting it would cause false outages."""
        limiter.enabled = True
        limiter.reset()
        client = _client(_stub_service([(1, 0.91)]))
        try:
            codes = [client.get("/health").status_code for _ in range(50)]
        finally:
            app.dependency_overrides.clear()
            limiter.reset()
            limiter.enabled = False
        assert set(codes) == {200}


# --------------------------------------------------------------------------
# Against the real corpus
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_client() -> Iterator[TestClient]:
    """The real service: loads the encoder and corpus once for this module."""
    app.dependency_overrides.clear()
    with TestClient(app) as c:
        yield c


@pytest.mark.skipif(
    not (DATA_DIR / "kb.parquet").exists(),
    reason="run the data pipeline first",
)
class TestAgainstRealCorpus:
    def test_health_reports_the_real_corpus(self, real_client: TestClient) -> None:
        body = real_client.get("/health").json()
        assert body["corpus_size"] > 7000
        assert "MiniLM" in body["encoder"]

    def test_every_answer_is_verbatim_kb_text(self, real_client: TestClient) -> None:
        """Hard constraint #3, enforced end to end: the service must never
        return text that is not already in the knowledge base."""
        from src.retriever import load_kb

        kb_answers = set(load_kb()["answer"])
        questions = [
            "How does COVID-19 spread?",
            "How long should I isolate?",
            "Do masks work?",
            "What are the symptoms?",
        ]
        for question in questions:
            body = real_client.post("/predict", json={"question": question}).json()
            if not body["abstained"]:
                assert body["answer"] in kb_answers, f"invented text for {question!r}"

    def test_off_topic_question_abstains(self, real_client: TestClient) -> None:
        body = real_client.post("/predict", json={"question": "What's the weather like?"}).json()
        assert body["abstained"] is True
        assert body["source"] is None

    def test_warm_latency_is_within_budget(self, real_client: TestClient) -> None:
        """PRD §10.3 targets <~300 ms added latency after warm start."""
        import time

        real_client.post("/predict", json={"question": "warmup"})
        elapsed = []
        for q in ["how does covid spread", "isolation period", "vaccine side effects"]:
            start = time.perf_counter()
            real_client.post("/predict", json={"question": q})
            elapsed.append((time.perf_counter() - start) * 1000)
        assert max(elapsed) < 300, f"slowest request {max(elapsed):.0f} ms"
