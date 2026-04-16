#!/usr/bin/env python3
"""
Build deterministic publication bundle for paper artifact release.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
from typing import Dict, Iterable, List, Tuple


FIXED_MTIME = 0


def _load_config(path: str) -> dict:
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise SystemExit(
            f"YAML config requested but PyYAML is not available ({e}). "
            "Use JSON or install pyyaml."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(root: str, value: str | None) -> str:
    if not value:
        return ""
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(root, value))


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _required_files(root: str, cfg: dict) -> List[str]:
    n = cfg.get("narrative", {})

    def stage(name: str, fnames: List[str]) -> List[str]:
        out = _resolve_path(root, n.get(name, {}).get("output_dir", ""))
        return [os.path.join(out, f) for f in fnames]

    eval_root = os.path.join(root, "experiments", "eval_outputs")
    files = [
        _resolve_path(root, "config/eval_config.yaml"),
        _resolve_path(root, ".github/workflows/ci-eval.yml"),
        _resolve_path(root, "scripts/evaluation/run_narrative_eval.py"),
        _resolve_path(root, "scripts/evaluation/build_master_report.py"),
        _resolve_path(root, "scripts/evaluation/build_repro_manifest.py"),
        _resolve_path(root, "scripts/evaluation/validate_publication_artifacts.py"),
        _resolve_path(root, "scripts/evaluation/build_publication_bundle.py"),
        _resolve_path(root, "scripts/evaluation/run_exp2_dual_protocol.py"),
        _resolve_path(root, "README.md"),
        _resolve_path(root, "RELEASE_NOTES.md"),
        _resolve_path(root, "CITATION.cff"),
        os.path.join(eval_root, "report_master_narrative.md"),
        os.path.join(eval_root, "report_paper_ready.md"),
        os.path.join(eval_root, "repro_manifest.json"),
        os.path.join(eval_root, "publication_gate_report.json"),
    ]

    files.extend(stage("exp1", ["metrics_aggregate.csv", "metrics_aggregate_ci.csv"]))
    files.extend(
        stage(
            "exp2",
            [
                "stations_eval_models_summary.csv",
                "stations_eval_rank_by_segment.csv",
                "checkpoints_used.csv",
            ],
        )
    )
    files.extend(stage("exp3", ["fullframe_eval_aggregate.csv", "seq_compare.csv"]))
    files.extend(stage("case_study_1", ["metrics_aggregate.csv"]))
    files.extend(stage("case_study_2", ["robustness_summary.csv", "cs2_rank_stability.csv"]))

    dual_dir = _resolve_path(root, n.get("exp2_dual_protocol", {}).get("output_dir", ""))
    if dual_dir:
        files.extend(
            [
                os.path.join(dual_dir, "exp2_protocol_comparison_by_model_segment.csv"),
                os.path.join(dual_dir, "exp2_protocol_rankings.csv"),
                os.path.join(dual_dir, "report_exp2_dual_protocol.md"),
            ]
        )

    # Deduplicate preserving order
    out: List[str] = []
    seen = set()
    for p in files:
        p = os.path.abspath(p)
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _collect_existing(files: Iterable[str]) -> Tuple[List[str], List[str]]:
    existing: List[str] = []
    missing: List[str] = []
    for p in files:
        if os.path.exists(p):
            existing.append(p)
        else:
            missing.append(p)
    return existing, missing


def _build_checksums_text(root: str, files: List[str]) -> bytes:
    lines = []
    for p in sorted(files):
        rel = os.path.relpath(p, root).replace(os.sep, "/")
        lines.append(f"{_sha256_file(p)}  {rel}")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def _add_file(tar: tarfile.TarFile, src: str, arcname: str) -> None:
    st = os.stat(src)
    info = tarfile.TarInfo(name=arcname)
    info.size = st.st_size
    info.mtime = FIXED_MTIME
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    with open(src, "rb") as f:
        tar.addfile(info, f)


def _add_bytes(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    info.mtime = FIXED_MTIME
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def main() -> int:
    ap = argparse.ArgumentParser(description="Build deterministic paper publication bundle.")
    ap.add_argument("--config", required=True, help="Path to eval_config.yaml (or .json)")
    ap.add_argument(
        "--out",
        default="dist/publication_eval_bundle.tar.gz",
        help="Output tar.gz path.",
    )
    ap.add_argument(
        "--allow-missing-optional",
        action="store_true",
        help="Allow missing optional files (dual-protocol artifacts).",
    )
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    cfg = _load_config(cfg_path)
    root = _resolve_path(os.getcwd(), cfg.get("project_root", ".")) or os.getcwd()
    root = os.path.abspath(root)
    out_path = _resolve_path(root, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    required = _required_files(root, cfg)
    existing, missing = _collect_existing(required)

    if missing:
        # Optional dual-protocol files can be skipped when requested.
        if args.allow_missing_optional:
            filtered_existing = []
            for p in existing:
                filtered_existing.append(p)
            optional_markers = [
                "exp2_protocol_comparison_by_model_segment.csv",
                "exp2_protocol_rankings.csv",
                "report_exp2_dual_protocol.md",
            ]
            hard_missing = [
                p
                for p in missing
                if not any(p.endswith(marker) for marker in optional_markers)
            ]
            if hard_missing:
                print("Missing required files:")
                for p in hard_missing:
                    print(f"- {p}")
                return 2
            existing = filtered_existing
        else:
            print("Missing required files:")
            for p in missing:
                print(f"- {p}")
            return 2

    checksums = _build_checksums_text(root, existing)

    manifest = {
        "bundle_version": 1,
        "format": "deterministic",
        "project_root": root,
        "config": cfg_path,
        "file_count": len(existing),
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")

    with open(out_path, "wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=FIXED_MTIME) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for src in sorted(existing):
                    rel = os.path.relpath(src, root).replace(os.sep, "/")
                    arc = f"publication_bundle/{rel}"
                    _add_file(tar, src, arc)
                _add_bytes(tar, "publication_bundle/SHA256SUMS.txt", checksums)
                _add_bytes(tar, "publication_bundle/BUNDLE_MANIFEST.json", manifest_bytes)

    print(out_path)
    print(f"bundle_sha256={_sha256_file(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
