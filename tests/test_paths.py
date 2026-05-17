"""Path resolution for dashboard / API (Windows paste in Docker)."""

from pathlib import Path

from app.core.paths import resolve_output_dir, PROJECT_ROOT


def test_windows_output_path_maps_to_project_relative():
    win = (
        "C:/Users/Jiangjiajun/Desktop/RoboVision/"
        "mcap_yolo_quality_gateway/outputs/sample_run"
    )
    root = resolve_output_dir(win)
    assert root == (PROJECT_ROOT / "outputs" / "sample_run").resolve()


def test_relative_output_path():
    root = resolve_output_dir("outputs/sample_run")
    assert root == (PROJECT_ROOT / "outputs" / "sample_run").resolve()


def test_bare_run_name_under_outputs(tmp_path, monkeypatch):
    import app.core.paths as paths_mod

    out = tmp_path / "outputs"
    run = out / "my_run"
    run.mkdir(parents=True)
    monkeypatch.setattr(paths_mod, "OUTPUTS_ROOT", out)
    monkeypatch.setattr(paths_mod, "PROJECT_ROOT", tmp_path)
    assert resolve_output_dir("my_run") == run.resolve()
