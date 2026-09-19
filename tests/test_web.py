"""Web interface (MVP 7)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi", reason="FastAPI is an optional dependency")
from fastapi.testclient import TestClient  # noqa: E402

from app.config import Config  # noqa: E402
from app.web import create_app  # noqa: E402
from app.web.jobs import JobManager, JobStatus, stages  # noqa: E402


@pytest.fixture
def client(config: Config) -> TestClient:
    return TestClient(create_app(config, offline=True))


def _wait(client: TestClient, job_id: str, timeout: float = 60.0) -> dict:
    """Poll a job until it leaves the running state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/reviews/{job_id}").json()
        if job["status"] in {"done", "failed"}:
            return job
        time.sleep(0.2)
    raise AssertionError("job did not finish in time")


def test_index_page_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Reviewer2" in response.text
    assert "<form" in response.text


def test_health_reports_backends(client: TestClient) -> None:
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert "provider" in payload["llm"]
    assert set(payload["backends"]) >= {"asr", "embeddings", "pdf"}


def test_review_requires_a_reference(client: TestClient, tmp_path: Path) -> None:
    video = tmp_path / "aula.mp4"
    video.write_bytes(b"fake")
    response = client.post(
        "/api/reviews", files={"video": ("aula.mp4", video.read_bytes(), "video/mp4")}
    )
    assert response.status_code == 400
    assert "referência" in response.json()["detail"]


def test_unsupported_video_format_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/reviews",
        files={
            "video": ("aula.exe", b"x", "application/octet-stream"),
            "references": ("ref.md", b"# t\n\ntexto", "text/markdown"),
        },
    )
    assert response.status_code == 400
    assert "não suportado" in response.json()["detail"]


def test_unsupported_reference_format_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/reviews",
        files={
            "video": ("aula.mp4", b"x", "video/mp4"),
            "references": ("ref.bin", b"x", "application/octet-stream"),
        },
    )
    assert response.status_code == 400


def test_invalid_url_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/reviews",
        files={"video": ("aula.mp4", b"x", "video/mp4")},
        data={"reference_urls": "não-é-uma-url"},
    )
    assert response.status_code == 400
    assert "URL inválida" in response.json()["detail"]


def test_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/api/reviews/deadbeef").status_code == 404
    assert client.get("/api/reviews/deadbeef/report").status_code == 404


def test_full_review_through_the_api(
    client: TestClient, transcript_file: Path, reference_file: Path
) -> None:
    """Happy path: a transcript JSON is accepted as media, skipping the ASR stage."""
    response = client.post(
        "/api/reviews",
        files=[
            ("video", ("aula.json", transcript_file.read_bytes(), "application/json")),
            ("references", ("referencia.md", reference_file.read_bytes(), "text/markdown")),
        ],
    )
    assert response.status_code == 202
    job = response.json()
    assert job["status"] in {"queued", "running"}
    assert job["stage_count"] == 6

    finished = _wait(client, job["job_id"])
    assert finished["status"] == "done", finished["error"]
    assert finished["summary"]["claims"] > 0
    assert finished["summary"]["analysed"] > 0
    assert finished["log"]

    report = client.get(f"/api/reviews/{job['job_id']}/report")
    assert report.status_code == 200
    assert report.text.startswith("# Reviewer2")
    assert "## 5. Análise das Afirmações" in report.text

    listing = client.get("/api/reviews").json()
    assert any(item["job_id"] == job["job_id"] for item in listing["jobs"])


def test_media_that_cannot_be_transcribed_fails_actionably(
    client: TestClient, reference_file: Path
) -> None:
    """A corrupt upload must surface the actionable error, not crash the server."""
    response = client.post(
        "/api/reviews",
        files=[
            ("video", ("quebrado.mp4", b"not a real video", "video/mp4")),
            ("references", ("referencia.md", reference_file.read_bytes(), "text/markdown")),
        ],
    )
    assert response.status_code == 202
    finished = _wait(client, response.json()["job_id"])
    assert finished["status"] == "failed"
    assert "recommended action" in finished["error"]


def test_job_manager_records_progress_and_summary(
    config: Config, transcript_file: Path, reference_file: Path
) -> None:
    """Drive the manager directly with a real transcript."""
    manager = JobManager(config)
    job = manager.submit(str(transcript_file), [str(reference_file)], offline=True)

    deadline = time.time() + 60
    while job.status not in {JobStatus.DONE, JobStatus.FAILED} and time.time() < deadline:
        time.sleep(0.2)

    assert job.status is JobStatus.DONE, job.error
    assert job.report_path and Path(job.report_path).exists()
    assert job.summary["claims"] > 0
    assert job.stage_index == len(job.to_dict()["log"]) or job.stage_index > 0
    assert 0.0 <= job.summary["evidence_coverage"] <= 1.0

    payload = job.to_dict()
    assert payload["status"] == "done"
    assert payload["stage_count"] == 6


def test_interface_follows_the_report_language(config: Config) -> None:
    """An English report language serves an English page and English stages."""
    config.report.language = "en"
    client = TestClient(create_app(config, offline=True))

    page = client.get("/").text
    assert '<html lang="en">' in page
    assert "Reference material" in page
    assert "Review</button>" in page
    assert "Materiais de referência" not in page

    assert stages("en")[0] == "transcription"


def test_errors_follow_the_report_language(config: Config, tmp_path: Path) -> None:
    """The messages the page shows are translated too."""
    config.report.language = "en"
    client = TestClient(create_app(config, offline=True))

    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"fake")
    response = client.post(
        "/api/reviews", files={"video": ("lecture.mp4", video.read_bytes(), "video/mp4")}
    )
    assert response.status_code == 400
    assert "reference material is required" in response.json()["detail"]

    assert client.get("/api/reviews/nope").json()["detail"] == "Job not found."
