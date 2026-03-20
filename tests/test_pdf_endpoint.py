import os
import json
import pytest
from app import app


def setup_module():
    os.makedirs('analysis_report', exist_ok=True)


def teardown_module():
    try:
        os.remove('analysis_report/test-pdf-1.json')
    except Exception:
        pass


def test_download_report_pdf_returns_pdf():
    client = app.test_client()
    analysis_id = 'test-pdf-1'
    report = {
        "meta": {"analysis_id": analysis_id, "timestamp": "2026-02-10T00:00:00Z", "category": "Basic Python", "topic": "Test Topic"},
        "scores": {"pylint_score": 1, "complexity_score": 1, "loc_score": 1, "structural_score": 1, "logic_score": 1, "cyclomatic_score": 1, "mi_score": 1, "composite_score": 1},
        "final_report": "Test report"
    }

    with open(f"analysis_report/{analysis_id}.json", "w", encoding="utf-8") as f:
        json.dump(report, f)

    # tiny 1x1 PNG
    png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAn8B9p3G9AAAAABJRU5ErkJggg=="
    data_uri = f"data:image/png;base64,{png_b64}"

    rv = client.post(f"/report/{analysis_id}/pdf", data={"chart_image": data_uri})

    assert rv.status_code == 200
    assert rv.content_type == "application/pdf"
    assert len(rv.data) > 200
