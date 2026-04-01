"""Tests for scripts/end_to_end_pipeline.py compatibility shim."""

from __future__ import annotations

from scripts import end_to_end_pipeline as pipeline_mod


def test_main_translates_to_core_canonical_smoke(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_long_main(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(pipeline_mod, "_long_main", _fake_long_main)
    exit_code = pipeline_mod.main(run_name="smoke-test", continue_on_error=False)

    assert exit_code == 0
    assert captured["argv"] == [
        "--run-tag",
        "smoke-test",
        "--pipeline-family",
        "core_canonical",
        "--sampling-profile",
        "smoke",
        "--no-rapids",
        "--no-notebooks",
        "--stop-on-optional-failure",
    ]
    assert captured["kwargs"]["default_pipeline_family"] == "core_canonical"
    assert captured["kwargs"]["default_sampling_profile"] == "smoke"
    assert captured["kwargs"]["compatibility_entrypoint"] == "scripts/end_to_end_pipeline.py"


def test_main_continue_on_error_omits_fail_fast_flag(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_long_main(argv, **kwargs):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(pipeline_mod, "_long_main", _fake_long_main)
    exit_code = pipeline_mod.main(
        run_name="smoke-test",
        continue_on_error=True,
        skip_make_dataset=True,
    )

    assert exit_code == 0
    assert "--stop-on-optional-failure" not in captured["argv"]
    assert "--resume" in captured["argv"]
