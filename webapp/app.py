"""A small local web interface for watching the agent work.

It adds no logic of its own. Every verdict on this page comes from exactly the
same pipeline the command line and the evaluation harness use — the page just
shows what came back, including the measurements behind each decision.
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from evalkit.dataset import CASES, build
from preflight.config import load_config
from preflight.explain import get_writer
from preflight.explain.writers import env
from preflight.pipeline import run
from preflight.types import Material, ProductSpec, Shape, State

ROOT = Path(__file__).resolve().parent.parent
EVALSET = ROOT / "evalset"
RESULTS = ROOT / "docs" / "results.md"

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
# This is a local tool that people will edit while it runs. Serving a
# stale stylesheet or script after a change is a waste of everyone's time.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

CONFIG = load_config()


@dataclass
class Tally:
    """Counts for this session only. In memory, cleared on restart."""

    checked: int = 0
    by_route: dict[str, int] = field(default_factory=dict)

    def record(self, route: State) -> None:
        self.checked += 1
        self.by_route[route.value] = self.by_route.get(route.value, 0) + 1


TALLY = Tally()

WRITER_NAMES = ("template", "gemini", "ollama")


def _writer(name: str | None):
    """The writer the page asked for. Unknown names get the template."""
    return get_writer(name if name in WRITER_NAMES else "template")


def _product(form) -> ProductSpec:
    return ProductSpec(
        width_in=float(form.get("width_in", 3.0)),
        height_in=float(form.get("height_in", 3.0)),
        shape=Shape(form.get("shape", Shape.DIE_CUT.value)),
        material=Material(form.get("material", Material.WHITE_VINYL.value)),
    )


def _payload(result, product: ProductSpec) -> dict:
    """Everything the page needs, including the numbers behind each verdict."""
    dpi = result.dpi or 0.0
    safe = CONFIG.for_check("SAFE_ZONE")
    return {
        "run_id": result.run_id,
        "route": result.state.value,
        "reason": result.reason,
        "dpi": result.dpi,
        "elapsed_ms": result.elapsed_ms,
        "product": {
            "width_in": product.width_in,
            "height_in": product.height_in,
            "shape": product.shape.value,
            "material": product.material.value,
        },
        "geometry": {
            "dpi": result.dpi,
            # Derived rather than carried back from the canvas: the pipeline
            # returns a verdict, not pixels, and the page only needs these to
            # scale the overlay.
            "px_width": round(dpi * product.width_in),
            "px_height": round(dpi * product.height_in),
            "warn_inset_px": round(float(safe["warn_below_in"]) * dpi, 1),
            "fail_inset_px": round(float(safe["fail_below_in"]) * dpi, 1),
        },
        "findings": [
            {
                **f.to_dict(),
                "overlay": f.overlay,
            }
            for f in result.findings
        ],
        "message": result.message.to_dict() if result.message else None,
    }


@app.get("/")
def index():
    """Serve the page with a version stamp on the stylesheet and script.

    Without it a browser happily keeps an old copy after an edit, which hides
    real changes and, worse, hides real breakage.
    """
    # template_folder / static_folder are relative to the package, not the
    # working directory the server happens to be started from.
    page = Path(app.root_path) / "templates" / "index.html"
    static = Path(app.root_path) / "static"
    version = max(
        int((static / name).stat().st_mtime) for name in ("app.css", "app.js")
    )
    html = page.read_text(encoding="utf-8").replace("/static/app.", f"/static/v{version}/app.")
    return html, 200, {
        "Content-Type": "text/html; charset=utf-8",
        # The page carries the version numbers, so the page itself must never
        # be the stale copy.
        "Cache-Control": "no-store",
    }


@app.get("/static/v<int:version>/<path:filename>")
def versioned_static(version: int, filename: str):
    return send_from_directory(Path(app.root_path) / "static", filename)


@app.get("/api/writers")
def writers():
    """Which message writers can be used right now. Checked, not assumed."""
    import urllib.request

    ollama = get_writer("ollama")
    try:
        with urllib.request.urlopen(f"{ollama.host}/api/tags", timeout=1):
            ollama_ok = True
    except OSError:
        ollama_ok = False
    return jsonify(
        {
            "template": {"ok": True, "label": "No model (fixed wording)"},
            "gemini": {
                "ok": bool(env("GEMINI_API_KEY")),
                "label": f"Gemini ({get_writer('gemini').model})",
            },
            "ollama": {"ok": ollama_ok, "label": f"Ollama ({ollama.model}, local)"},
        }
    )


@app.get("/api/tally")
def tally():
    return jsonify(asdict(TALLY))


@app.post("/api/check")
def check():
    upload = request.files.get("artwork")
    if upload is None or not upload.filename:
        return jsonify({"error": "Choose an artwork file first."}), 400

    product = _product(request.form)
    suffix = Path(upload.filename).suffix or ".png"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        upload.save(path)
        result = run(
            path, product, config=CONFIG, draft=True, writer=_writer(request.form.get("writer"))
        )

    TALLY.record(result.state)
    payload = _payload(result, product)
    payload["filename"] = upload.filename
    return jsonify(payload)


@app.get("/api/samples")
def samples():
    """The labelled evaluation files, so the agent can be tried without hunting
    for artwork. Each one carries the answer a careful person would give."""
    if not (EVALSET / "labels.yaml").exists():
        build(EVALSET)
    return jsonify(
        [
            {
                "name": c.name,
                "note": c.note,
                "expect_route": c.expect_route.value,
                "product": {
                    "width_in": c.product.width_in,
                    "height_in": c.product.height_in,
                    "shape": c.product.shape.value,
                    "material": c.product.material.value,
                },
                "viewable": c.raw_bytes is None,
            }
            for c in CASES
        ]
    )


@app.post("/api/check-sample")
def check_sample():
    body = request.json or {}
    name = body.get("name", "")
    case = next((c for c in CASES if c.name == name), None)
    if case is None:
        return jsonify({"error": f"No sample named {name!r}."}), 404

    if not (EVALSET / case.filename).exists():
        build(EVALSET)

    result = run(
        EVALSET / case.filename,
        case.product,
        config=CONFIG,
        draft=True,
        writer=_writer(body.get("writer")),
    )
    TALLY.record(result.state)

    payload = _payload(result, case.product)
    payload["filename"] = case.filename
    payload["expected_route"] = case.expect_route.value
    payload["note"] = case.note
    return jsonify(payload)


@app.get("/api/sample-image/<name>")
def sample_image(name: str):
    case = next((c for c in CASES if c.name == name), None)
    if case is None or case.raw_bytes is not None:
        return jsonify({"error": "not viewable"}), 404
    if not (EVALSET / case.filename).exists():
        build(EVALSET)
    return send_file(EVALSET / case.filename, mimetype="image/png")


@app.get("/api/results")
def results():
    if not RESULTS.exists():
        return jsonify({"markdown": ""})
    return jsonify({"markdown": RESULTS.read_text(encoding="utf-8")})


def main() -> None:
    print("Pre-flight running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
