from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
import io

from catfish_core import load_artifacts, render_scan_summary
from webapp.validation import run_validation_suite


@lru_cache(maxsize=1)
def get_artifacts():
    return load_artifacts()


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "detector_bundle.pkl"
REPORT_PATH = PROJECT_ROOT / "WIA1006_WID3006_Group Assignment.pdf"
VALIDATION_PAYLOAD = json.dumps(run_validation_suite())


def _page_context(artifacts):
    return {
        "artifacts": artifacts,
        "notebook_cells": artifacts.notebook_cells,
        "leaderboard": artifacts.leaderboard.reset_index().to_dict(orient="records"),
        "feature_names": artifacts.feature_names,
        "test_profiles": artifacts.test_profiles,
    }


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            page_title="Catfish Detector | Scanner",
            page_description="Interactive scanner",
            active_page="scanner",
            **_page_context(get_artifacts()),
        )

    @app.get("/notebook")
    def notebook_page():
        return render_template(
            "notebook.html",
            page_title="Notebook",
            page_description="Notebook cells from the assignment",
            active_page="notebook",
            **_page_context(get_artifacts()),
        )

    @app.get("/models")
    def models_page():
        # Also collect available diagnostic plot images so templates can inline thumbnails
        artifacts = get_artifacts()
        plot_dir = PROJECT_ROOT / 'artifacts' / 'plots'
        images = []
        if plot_dir.exists():
            for p in sorted(plot_dir.iterdir()):
                if p.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    images.append(p.name)
        else:
            # Fallback to static copies if available (copied during verification)
            static_plot_dir = PROJECT_ROOT / 'webapp' / 'static' / 'plots'
            if static_plot_dir.exists():
                for p in sorted(static_plot_dir.iterdir()):
                    if p.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                        images.append(p.name)
        return render_template(
            "models.html",
            page_title="Models",
            page_description="Model leaderboard and profiles",
            active_page="models",
            images=images,
            **_page_context(artifacts),
        )

    @app.get("/export")
    def export_page():
        return render_template(
            "export.html",
            page_title="Downloads",
            page_description="Download the cached model bundle and report",
            active_page="downloads",
            model_bundle_url=url_for("download_model_bundle"),
            report_url=url_for("download_report"),
            **_page_context(get_artifacts()),
        )

    @app.get("/download/model-bundle.pkl")
    def download_model_bundle():
        if not MODEL_BUNDLE_PATH.exists():
            return app.response_class("Model bundle is missing.", status=404)
        return send_file(
            MODEL_BUNDLE_PATH,
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name="detector_bundle.pkl",
        )

    @app.get("/download/report.pdf")
    def download_report():
        if not REPORT_PATH.exists():
            return app.response_class("Report file is missing.", status=404)
        return send_file(
            REPORT_PATH,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=REPORT_PATH.name,
        )

    @app.get("/download/notebook.ipynb")
    def download_notebook_ipynb():
        if not NOTEBOOK_PATH.exists():
            return app.response_class("Notebook file is missing.", status=404)
        return send_file(
            NOTEBOOK_PATH,
            mimetype="application/json",
            as_attachment=True,
            download_name=NOTEBOOK_PATH.name,
        )

    @app.get("/download/notebook.py")
    def download_notebook_py():
        if not NOTEBOOK_PATH.exists():
            return app.response_class("Notebook file is missing.", status=404)
        with NOTEBOOK_PATH.open("r", encoding="utf-8") as handle:
            nb = json.load(handle)

        parts = []
        for idx, cell in enumerate(nb.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", [])
            if isinstance(source, list):
                src_text = "".join(source)
            else:
                src_text = str(source)
            parts.append(f"# Cell {idx}\n")
            parts.append(src_text)
            parts.append("\n\n")

        content = "".join(parts)
        return send_file(io.BytesIO(content.encode("utf-8")), mimetype="text/x-python", as_attachment=True, download_name=NOTEBOOK_PATH.with_suffix('.py').name)

    @app.get('/plots')
    def plots_page():
        artifacts = get_artifacts()
        plot_dir = PROJECT_ROOT / 'artifacts' / 'plots'
        images = []
        if plot_dir.exists():
            for p in sorted(plot_dir.iterdir()):
                if p.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    images.append(p.name)
        return render_template('plots.html', page_title='Plots', page_description='Diagnostic plots', active_page='plots', images=images)

    @app.get('/download/plots/<path:filename>')
    def download_plot(filename: str):
        plot_dir = PROJECT_ROOT / 'artifacts' / 'plots'
        target = plot_dir.joinpath(filename).resolve()
        try:
            # Ensure the requested file is inside the plots directory
            if not str(target).startswith(str(plot_dir.resolve())):
                return app.response_class('Invalid file', status=400)
            if not target.exists():
                return app.response_class('Not found', status=404)
            return send_file(target, as_attachment=True)
        except Exception:
            return app.response_class('Error', status=500)

    @app.get('/artifacts/plots/<path:filename>')
    def serve_plot_static(filename: str):
        plot_dir = PROJECT_ROOT / 'artifacts' / 'plots'
        target = plot_dir.joinpath(filename).resolve()
        if not str(target).startswith(str(plot_dir.resolve())):
            return app.response_class('Invalid file', status=400)
        if not target.exists():
            return app.response_class('Not found', status=404)
        return send_file(target)

    @app.post("/api/scan")
    def api_scan():
        artifacts = get_artifacts()
        payload: Dict[str, Any] = request.get_json(force=True, silent=False) or {}
        result = render_scan_summary(payload, artifacts)
        serializable = {
            "behavioral_score": result["behavioral_score"],
            "top_flags": [{"name": name, "value": value} for name, value in result["top_flags"]],
            "model_probs": result["model_probs"],
            "thresholds": artifacts.thresholds,
            "ml_votes": result["ml_votes"],
            "verdict_label": result["verdict_label"],
            "final_verdict": result["final_verdict"],
            "vector": result["vector"].round(4).tolist(),
        }
        return jsonify(serializable)

    @app.get("/api/check")
    def api_check():
        return app.response_class(VALIDATION_PAYLOAD, mimetype="application/json")

    @app.get("/api/validation")
    def api_validation():
        return redirect(url_for("api_check"))

    @app.get("/api/test-profiles")
    def api_test_profiles():
        artifacts = get_artifacts()
        return jsonify(artifacts.test_profiles)

    return app
