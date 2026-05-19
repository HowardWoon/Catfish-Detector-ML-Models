from __future__ import annotations

from typing import Any, Dict, List

from catfish_core import load_artifacts, render_scan_summary


def run_validation_suite() -> List[Dict[str, Any]]:
    artifacts = load_artifacts()
    results: List[Dict[str, Any]] = []

    results.append(
        {
            "name": "artifacts_load",
            "passed": bool(artifacts.models and artifacts.feature_names),
            "details": f"models={len(artifacts.models)} features={len(artifacts.feature_names)}",
        }
    )
    results.append(
        {
            "name": "notebook_cells_present",
            "passed": bool(len(artifacts.notebook_cells) > 0),
            "details": f"cells={len(artifacts.notebook_cells)}",
        }
    )

    contrast_a = render_scan_summary(artifacts.test_profiles["genuine_median"], artifacts)
    contrast_b = render_scan_summary(artifacts.test_profiles["high_activity"], artifacts)
    results.append(
        {
            "name": "scanner_varies_with_input",
            "passed": bool(abs(contrast_a["behavioral_score"] - contrast_b["behavioral_score"]) > 0.01),
            "details": f"{contrast_a['behavioral_score']:.1f}% vs {contrast_b['behavioral_score']:.1f}%",
        }
    )
    results.append(
        {
            "name": "model_signals_available",
            "passed": bool(len(contrast_a["model_probs"]) == len(artifacts.models)),
            "details": f"probabilities={len(contrast_a['model_probs'])}",
        }
    )

    return results