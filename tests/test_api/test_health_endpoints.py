"""Unit tests for API health/readiness endpoint functions."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import Response

from api.routers import health as health_router


def _fake_request(preload_status: dict) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(preload_status=preload_status))
    )


def test_health_returns_typed_liveness_payload():
    payload = health_router.health()
    assert payload.status == "ok"
    assert isinstance(payload.version, str)


def test_ready_returns_200_when_files_exist_and_preload_succeeds(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "pd_test.cbm").write_text("x")
    (model_dir / "cal_test.pkl").write_text("x")
    duckdb_path = tmp_path / "lending_club.duckdb"
    duckdb_path.write_text("x")

    monkeypatch.setattr(health_router, "MODEL_DIR", model_dir)
    monkeypatch.setattr(health_router, "PD_MODEL_FILE", "pd_test.cbm")
    monkeypatch.setattr(health_router, "CALIBRATOR_FILE", "cal_test.pkl")
    monkeypatch.setattr(health_router, "DUCKDB_PATH", duckdb_path)

    request = _fake_request(
        {
            "attempted": True,
            "pd_model_loaded": True,
            "calibrator_loaded": True,
            "error": None,
        }
    )
    response = Response()
    payload = health_router.ready(request, response)

    assert response.status_code == 200
    assert payload.ready is True
    assert payload.checks.pd_model is True
    assert payload.checks.calibrator is True
    assert payload.checks.duckdb is True
    assert payload.preload.pd_model_loaded is True


def test_ready_returns_503_when_preload_failed(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "pd_test.cbm").write_text("x")
    (model_dir / "cal_test.pkl").write_text("x")
    duckdb_path = tmp_path / "lending_club.duckdb"
    duckdb_path.write_text("x")

    monkeypatch.setattr(health_router, "MODEL_DIR", model_dir)
    monkeypatch.setattr(health_router, "PD_MODEL_FILE", "pd_test.cbm")
    monkeypatch.setattr(health_router, "CALIBRATOR_FILE", "cal_test.pkl")
    monkeypatch.setattr(health_router, "DUCKDB_PATH", duckdb_path)

    request = _fake_request(
        {
            "attempted": True,
            "pd_model_loaded": False,
            "calibrator_loaded": True,
            "error": "pd_model: missing",
        }
    )
    response = Response()
    payload = health_router.ready(request, response)

    assert response.status_code == 503
    assert payload.ready is False
    assert payload.preload.error == "pd_model: missing"
    assert payload.preload.pd_model_loaded is False
