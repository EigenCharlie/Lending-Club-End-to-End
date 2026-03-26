"""App shell tests using Streamlit AppTest."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_shell_renders_without_exceptions() -> None:
    app_path = Path(__file__).resolve().parents[2] / "streamlit_app" / "app.py"
    at = AppTest.from_file(app_path, default_timeout=20)
    at.run(timeout=20)
    assert len(at.exception) == 0
    assert len(at.title) >= 1
    assert any("PD & Uncertainty Lab" in str(title.value) for title in at.title)
