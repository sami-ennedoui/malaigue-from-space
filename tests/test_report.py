import numpy as np

from malaigue import report


def test_plot_anomaly_map_writes_png(tmp_path):
    raster = np.random.rand(16, 16)
    out = tmp_path / "anom.png"
    report.plot_anomaly_map(raster, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_write_evaluation_contains_verdict(tmp_path):
    out = tmp_path / "evaluation.md"
    report.write_evaluation({"verdict": "index wins", "spatial_iou": 0.12}, str(out))
    text = out.read_text()
    assert "index wins" in text and "spatial_iou" in text
