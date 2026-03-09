#!/usr/bin/env python3
"""
Consolidate visual artifacts for manuscript narrative curation.

This script copies (non-destructive) image-like files from the configured
source folders into:
  figures/final_all_images/all_sources

It also generates:
  - figures/final_all_images/manifest_all_sources.csv
  - figures/final_all_images/NARRATIVE_IMAGE_ATLAS.md
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


IMAGE_EXTS = {".png", ".pdf", ".jpg", ".jpeg", ".gif"}


@dataclass
class Record:
    consolidated_file: str
    source_relative_path: str
    source_group: str
    narrative_bucket: str
    priority: str
    bytes_size: int


def classify_bucket(path_lower: str) -> str:
    s = path_lower
    if any(k in s for k in ["robust", "montecarlo", "epsilon", "rank_stability", "f10_robustness"]):
        return "cs2b_robustness"
    if any(k in s for k in ["cooling", "persistence", "dissipat", "casestudy2", "cs2"]):
        return "cs2a_night_cooling"
    if any(k in s for k in ["exp3", "experiment3", "ablation", "seq_len", "seq_compare", "memory", "bottleneck"]):
        return "exp3_ablation"
    if any(k in s for k in ["exp2", "station", "meteocat", "scatter", "timeseries_stations", "segment_rmse"]):
        return "exp2_stations"
    if any(k in s for k in ["exp1", "experiment1", "spatial_performance", "qualitative", "mamba_vs_bilinear", "model_grid"]):
        return "exp1_spatial"
    if any(k in s for k in ["heatwave", "case_study", "cs1", "hourly_field_evolution", "hourly_top2", "day_night"]):
        return "cs1_heatwave"
    if any(k in s for k in ["method", "overview", "architecture", "study_area", "pipeline", "map"]):
        return "context_method"
    return "supporting_misc"


def classify_priority(group: str, source_rel_lower: str) -> str:
    if "paper_figures_final" in source_rel_lower or group in {"repro_v2", "imagenes"}:
        return "primary"
    if group in {"legacy_png", "legacy_pdf", "presentation"}:
        return "secondary"
    return "archive"


def safe_consolidated_name(group: str, source_rel: str) -> str:
    rel = source_rel.replace("/", "__")
    candidate = f"{group}__{rel}"
    if len(candidate) <= 220:
        return candidate
    p = Path(source_rel)
    h = hashlib.sha1(source_rel.encode("utf-8")).hexdigest()[:12]
    stem = p.stem[:120]
    return f"{group}__{stem}__{h}{p.suffix.lower()}"


def list_images(root: Path, source_rel: str) -> Iterable[Path]:
    src = root / source_rel
    if not src.exists():
        return []
    files = []
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        # Never recurse into consolidation output.
        if "figures/final_all_images" in p.as_posix():
            continue
        files.append(p)
    return files


def build_markdown(records: List[Record], out_md: Path) -> None:
    bucket_order = [
        "context_method",
        "exp1_spatial",
        "exp2_stations",
        "exp3_ablation",
        "cs1_heatwave",
        "cs2a_night_cooling",
        "cs2b_robustness",
        "supporting_misc",
    ]

    by_bucket: Dict[str, List[Record]] = {b: [] for b in bucket_order}
    for r in records:
        by_bucket.setdefault(r.narrative_bucket, []).append(r)

    def sort_key(rec: Record):
        pr = {"primary": 0, "secondary": 1, "archive": 2}.get(rec.priority, 3)
        ext = Path(rec.consolidated_file).suffix.lower()
        ex = {".pdf": 0, ".png": 1, ".jpg": 2, ".jpeg": 2, ".gif": 3}.get(ext, 4)
        return (pr, ex, -rec.bytes_size, rec.consolidated_file)

    lines: List[str] = []
    lines.append("# Narrative Image Atlas (All Sources)\n")
    lines.append("This atlas consolidates image assets from `imagenes`, `figures`, and `experiments/figures`.\n")
    lines.append(f"- Total consolidated files: **{len(records)}**\n")

    for b in bucket_order:
        rows = sorted(by_bucket.get(b, []), key=sort_key)
        if not rows:
            continue
        lines.append(f"\n## {b}\n")
        lines.append(f"- Files: **{len(rows)}**\n")
        lines.append("| priority | consolidated_file | source_relative_path |")
        lines.append("|---|---|---|")
        for r in rows[:12]:
            lines.append(f"| {r.priority} | {r.consolidated_file} | {r.source_relative_path} |")

    lines.append("\n## Main Manuscript Recommended Set\n")
    # Heuristic top picks by bucket
    for b in ["exp1_spatial", "exp2_stations", "exp3_ablation", "cs1_heatwave", "cs2a_night_cooling", "cs2b_robustness"]:
        rows = sorted(by_bucket.get(b, []), key=sort_key)
        if not rows:
            continue
        top = rows[:3]
        lines.append(f"\n### {b}\n")
        for r in top:
            lines.append(f"- `{r.consolidated_file}` ({r.priority})")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[2]

    source_groups = [
        ("imagenes", "imagenes"),
        ("legacy_png", "figures/png"),
        ("legacy_pdf", "figures/pdf"),
        ("repro_v2", "figures/repro_v2"),
        ("paper_final", "experiments/eval_outputs/paper_figures_final"),
        ("presentation", "experiments/presentation_figures"),
        ("experiments_figures", "experiments/figures"),
    ]

    out_root = root / "figures" / "final_all_images"
    out_all = out_root / "all_sources"
    out_all.mkdir(parents=True, exist_ok=True)

    records: List[Record] = []
    seen_names = set()

    for group, source_rel in source_groups:
        for p in list_images(root, source_rel):
            rel = p.relative_to(root).as_posix()
            out_name = safe_consolidated_name(group, rel)
            # Resolve rare collisions deterministically.
            if out_name in seen_names:
                h = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
                pp = Path(out_name)
                out_name = f"{pp.stem}__{h}{pp.suffix}"
            seen_names.add(out_name)

            dst = out_all / out_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)

            rel_l = rel.lower()
            records.append(
                Record(
                    consolidated_file=f"all_sources/{out_name}",
                    source_relative_path=rel,
                    source_group=group,
                    narrative_bucket=classify_bucket(rel_l),
                    priority=classify_priority(group, rel_l),
                    bytes_size=p.stat().st_size,
                )
            )

    records.sort(key=lambda r: (r.narrative_bucket, r.priority, r.consolidated_file))

    manifest = out_root / "manifest_all_sources.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "consolidated_file",
                "source_relative_path",
                "source_group",
                "narrative_bucket",
                "priority",
                "bytes_size",
            ]
        )
        for r in records:
            w.writerow(
                [
                    r.consolidated_file,
                    r.source_relative_path,
                    r.source_group,
                    r.narrative_bucket,
                    r.priority,
                    r.bytes_size,
                ]
            )

    atlas = out_root / "NARRATIVE_IMAGE_ATLAS.md"
    build_markdown(records, atlas)

    print(f"Consolidated files: {len(records)}")
    print(manifest)
    print(atlas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
