#!/usr/bin/env python3
"""
Fix all stale path references after the scripts/ and docker/ reorganization.
Run from the repository root:
    python3 fix_stale_paths.py
"""

import re
from pathlib import Path

BASE = Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════
# Path replacement rules
# ═══════════════════════════════════════════════════════════════

# scripts/ reorganization: old flat path -> new categorized path
SCRIPT_RENAMES = {
    # Ablation
    "scripts/run_ablation.py":                          "scripts/ablation/run_ablation.py",
    "scripts/run_ablation_tiles.py":                    "scripts/ablation/run_ablation_tiles.py",
    "scripts/run_ablation_tiles_heatwave_caffeinate.sh":"scripts/ablation/run_ablation_tiles_heatwave_caffeinate.sh",
    "scripts/run_ablation_tiles_heatwave_server.sh":    "scripts/ablation/run_ablation_tiles_heatwave_server.sh",
    "scripts/run_experiment3_fullframe_replica.sh":      "scripts/ablation/run_experiment3_fullframe_replica.sh",
    # Training
    "scripts/gpu_server_train.py":      "scripts/training/gpu_server_train.py",
    "scripts/torch_gpu_train.py":       "scripts/training/torch_gpu_train.py",
    "scripts/train_tiles.py":           "scripts/training/train_tiles.py",
    "scripts/train_torch.py":           "scripts/training/train_torch.py",
    "scripts/train_transformer.py":     "scripts/training/train_transformer.py",
    "scripts/fullframe_local_fast.sh":  "scripts/training/fullframe_local_fast.sh",
    "scripts/run_server_fullframe.sh":  "scripts/training/run_server_fullframe.sh",
    "scripts/train_tiles_baselines_no_mamba.sh": "scripts/training/train_tiles_baselines_no_mamba.sh",
    # Evaluation
    "scripts/evaluate_test_set.py":     "scripts/evaluation/evaluate_test_set.py",
    "scripts/evaluate_stations_grib.py":"scripts/evaluation/evaluate_stations_grib.py",
    "scripts/evaluate_for_paper.py":    "scripts/evaluation/evaluate_for_paper.py",
    "scripts/consolidate_experiment1.py":"scripts/evaluation/consolidate_experiment1.py",
    "scripts/consolidate_experiment3.py":"scripts/evaluation/consolidate_experiment3.py",
    "scripts/run_stations_eval_ablation.sh":"scripts/evaluation/run_stations_eval_ablation.sh",
    "scripts/run_p95_eval_caffeinate.sh": "scripts/evaluation/run_p95_eval_caffeinate.sh",
    # Inference
    "scripts/run_inference.py":              "scripts/inference/run_inference.py",
    "scripts/run_inference_tiles_fullframe.py":"scripts/inference/run_inference_tiles_fullframe.py",
    # Figures
    "scripts/generate_presentation_figures.py":"scripts/figures/generate_presentation_figures.py",
    "scripts/generate_paper_figures.py": "scripts/figures/generate_paper_figures.py",
    # Tools
    "scripts/check_data_health.py":     "scripts/tools/check_data_health.py",
    "scripts/derive_aemet_heatwaves.py":"scripts/tools/derive_aemet_heatwaves.py",
    "scripts/make_server_bundle.py":    "scripts/tools/make_server_bundle.py",
    "scripts/make_server_bundle.sh":    "scripts/tools/make_server_bundle.sh",
    "scripts/repair_zarr_nans.py":      "scripts/tools/repair_zarr_nans.py",
    "scripts/validate_data_coupling.py":"scripts/tools/validate_data_coupling.py",
    "scripts/print_active_config.py":   "scripts/tools/print_active_config.py",
}

# Docker compose: old root path -> new docker/ path
DOCKER_RENAMES = {
    "docker-compose.gpu-optimized.yml": "docker/compose.gpu.yml",
    "docker-compose.server-fullframe.yml": "docker/compose.server-fullframe.yml",
    "docker-compose.server-tiles-heatwave.yml": "docker/compose.server-tiles-heatwave.yml",
    "docker-compose.server-exp3-eval.yml": "docker/compose.server-exp3-eval.yml",
    "docker-compose.cpu.yml":           "docker/compose.cpu.yml",
    "docker-compose.figures.yml":       "docker/compose.figures.yml",
    "docker-compose.yml":               "docker/compose.yml",
}

# ═══════════════════════════════════════════════════════════════
# Files to patch
# ═══════════════════════════════════════════════════════════════

# All shell scripts, docker compose files, deploy script, markdown docs
def collect_files():
    files = []
    for pattern in ["*.sh", "*.yml", "*.yaml", "*.md", "*.py"]:
        files.extend(BASE.rglob(pattern))
    # Filter out .git, __pycache__, archive, experiments, data
    filtered = []
    for f in files:
        parts = f.relative_to(BASE).parts
        if any(p in (".git", "__pycache__", "archive", "experiments", "data", ".idea") for p in parts):
            continue
        filtered.append(f)
    return sorted(set(filtered))


def fix_file(filepath: Path) -> int:
    """Apply all path renames to a single file. Returns number of replacements."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return 0

    original = content
    
    # Apply script path renames (longer paths first to avoid partial matches)
    for old, new in sorted(SCRIPT_RENAMES.items(), key=lambda x: -len(x[0])):
        content = content.replace(old, new)

    # Apply docker compose renames (longer paths first)
    for old, new in sorted(DOCKER_RENAMES.items(), key=lambda x: -len(x[0])):
        content = content.replace(old, new)

    if content != original:
        filepath.write_text(content, encoding="utf-8")
        return 1
    return 0


def main():
    print("=" * 60)
    print("Fixing stale path references")
    print("=" * 60)

    files = collect_files()
    updated = 0

    for f in files:
        result = fix_file(f)
        if result:
            rel = f.relative_to(BASE)
            print(f"  ✅ {rel}")
            updated += 1

    print(f"\n📊 Updated {updated} file(s) out of {len(files)} scanned")
    print("\n" + "=" * 60)
    print("✅ All stale paths fixed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
