from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from pair.scoring.canonicalize import canonicalize_trace
from pair.utils.io import ensure_dir, write_json


def render_trace_pair(family: dict[str, Any], episodes: list[dict[str, Any]], predictions: dict[str, dict[str, Any]], out_path: str | Path) -> str:
    blocks = []
    for ep in sorted(episodes, key=lambda e: e["world_type"]):
        pred = predictions.get(ep["episode_id"], {})
        raw_rows = "".join(
            f"<tr><td>{i}</td><td>{escape(str(s.get('tool')))}</td><td><pre>{escape(str(s.get('args')))}</pre></td><td>{escape(str(s.get('ok')))}</td></tr>"
            for i, s in enumerate(pred.get("trace", []))
        )
        canon_rows = "".join(
            f"<tr><td>{i}</td><td>{escape(op['intent'])}</td><td><pre>{escape(str(op['key_args']))}</pre></td><td>{escape(str(op['touches']))}</td></tr>"
            for i, op in enumerate(canonicalize_trace(pred.get("trace", [])))
        )
        blocks.append(
            f"<section><h2>{escape(ep['episode_id'])}</h2>"
            f"<p>World: <strong>{escape(ep['world_type'])}</strong> | Success: <strong>{escape(str(pred.get('final_success')))}</strong></p>"
            f"<p>Affected cone: <code>{escape(str(ep.get('affected_cone', [])))}</code></p>"
            f"<h3>Raw trace</h3><table><tr><th>#</th><th>Tool</th><th>Args</th><th>OK</th></tr>{raw_rows}</table>"
            f"<h3>Canonical trace</h3><table><tr><th>#</th><th>Intent</th><th>Args</th><th>Touches</th></tr>{canon_rows}</table></section>"
        )
    html = """<!doctype html><html><head><meta charset='utf-8'><title>PAIR trace</title>
<style>body{font-family:system-ui,sans-serif;margin:28px;color:#17202a}section{border-top:1px solid #d8dee4;padding:18px 0}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #d8dee4;padding:6px;vertical-align:top;text-align:left}pre{white-space:pre-wrap;margin:0}</style>
</head><body>""" + f"<h1>{escape(family['family_id'])}</h1>" + "".join(blocks) + "</body></html>"
    out = Path(out_path)
    ensure_dir(out.parent)
    out.write_text(html, encoding="utf-8")
    write_json(out.with_suffix(".manifest.json"), {"family_id": family["family_id"], "episodes": [e["episode_id"] for e in episodes]})
    return str(out)
