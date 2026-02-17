"""
Auditoría de variables (LR / HR / estáticos) basada en:
- Missingness y estadísticas básicas
- Correlación (Pearson)
- Multicolinealidad en estáticos (VIF)
- Relación con HR target (t2m) y con el sesgo medio HR-LR (patrón espacial)

Salida en `reports/`:
- feature_audit.md
- corr_lr.csv, corr_static.csv
- corr_lr.png, corr_static.png (opcional, si matplotlib está disponible)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"

PATH_CACHE_ZARR = REPO_ROOT / "data" / "processed" / "weather_cache.zarr"
PATH_STATIC_ZARR = REPO_ROOT / "data" / "processed" / "weather_static_FINAL_stations.zarr"
PATH_STATIC_PROCESSED_NPY = REPO_ROOT / "data" / "processed" / "static_processed.npy"


@dataclass(frozen=True)
class VarSummary:
    name: str
    nan_frac: float
    mean: float
    std: float
    vmin: float
    vmax: float


def _ensure_reports_dir() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _fmt(x: float, nd: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "NaN"
    return f"{x:.{nd}f}"


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    x = x[ok] - np.mean(x[ok])
    y = y[ok] - np.mean(y[ok])
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _corrcoef_matrix(samples_by_var: np.ndarray) -> np.ndarray:
    """
    samples_by_var: (n_samples, n_vars) -> (n_vars, n_vars)
    """
    x = np.asarray(samples_by_var, dtype=np.float64)
    ok = np.isfinite(x).all(axis=1)
    x = x[ok]
    if x.shape[0] < 3:
        return np.full((x.shape[1], x.shape[1]), np.nan, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    denom = np.sqrt((x * x).sum(axis=0, keepdims=True))
    denom = denom.T @ denom
    cov = (x.T @ x)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / denom
    np.fill_diagonal(corr, 1.0)
    return corr


def _vif(samples_by_var: np.ndarray, var_names: list[str]) -> dict[str, float]:
    """
    VIF por variable regresando cada Xi sobre X_{-i}.
    """
    x = np.asarray(samples_by_var, dtype=np.float64)
    ok = np.isfinite(x).all(axis=1)
    x = x[ok]
    n, p = x.shape
    if n < p + 5:
        return {name: float("nan") for name in var_names}

    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, ddof=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    x = x / std

    out: dict[str, float] = {}
    for i, name in enumerate(var_names):
        y = x[:, i]
        others = np.delete(x, i, axis=1)
        beta, *_ = np.linalg.lstsq(others, y, rcond=None)
        y_hat = others @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
        r2 = min(max(r2, 0.0), 0.999999)
        out[name] = float(1.0 / (1.0 - r2))
    return out


def _top_pairs(corr: np.ndarray, names: list[str], threshold: float, top_k: int) -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            r = corr[i, j]
            if np.isfinite(r) and abs(r) >= threshold:
                pairs.append((names[i], names[j], float(r)))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return pairs[:top_k]


def _write_csv_matrix(corr: np.ndarray, names: list[str], out_csv: Path) -> None:
    header = ",".join(["var"] + [str(n) for n in names])
    lines = [header]
    for i, name in enumerate(names):
        row = [name] + [f"{corr[i, j]:.6f}" if np.isfinite(corr[i, j]) else "" for j in range(len(names))]
        lines.append(",".join(row))
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _maybe_plot_heatmap(corr: np.ndarray, names: list[str], out_png: Path, title: str) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".tmp_mpl"))
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 8), dpi=160)
        ax = fig.add_subplot(111)
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(title)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
    except Exception:
        return


def main() -> int:
    _ensure_reports_dir()

    import xarray as xr

    ds_cache = xr.open_zarr(PATH_CACHE_ZARR, consolidated=True)
    hr = ds_cache["hr_target"]
    lr = ds_cache["lr_input"]
    lr_var_names = [str(v) for v in ds_cache["variable"].values.tolist()]

    ds_static = xr.open_zarr(PATH_STATIC_ZARR, consolidated=True)
    static_var_names = [str(v) for v in ds_static.data_vars.keys()]
    static_processed = np.load(PATH_STATIC_PROCESSED_NPY)
    if static_processed.ndim != 3 or static_processed.shape[-1] != len(static_var_names):
        raise RuntimeError(
            f"`{PATH_STATIC_PROCESSED_NPY}` shape {static_processed.shape} no coincide con "
            f"{len(static_var_names)} variables estáticas {static_var_names}."
        )

    # ---------------------------
    # HR: stats globales
    # ---------------------------
    hr_nan = float(hr.isnull().mean().compute().item())
    hr_mean_ts = hr.mean(dim=("latitude", "longitude")).compute().values  # (time,)
    hr_mean = float(np.nanmean(hr_mean_ts))
    hr_std = float(np.nanstd(hr_mean_ts))
    hr_clim = hr.mean(dim="time").compute().values  # (lat, lon)

    # Sesgo medio HR - upsample(mean LR t2m)
    lr_t2m_mean = lr.sel(variable="t2m").mean(dim="time").compute()
    lr_t2m_mean = lr_t2m_mean.rename({"latitude_lr": "latitude", "longitude_lr": "longitude"}).assign_coords(
        latitude=ds_cache["latitude_lr"].values,
        longitude=ds_cache["longitude_lr"].values,
    )
    lr_t2m_mean_up = lr_t2m_mean.interp(latitude=ds_cache["latitude"], longitude=ds_cache["longitude"]).compute().values
    residual_mean_map = hr_clim - lr_t2m_mean_up

    # ---------------------------
    # LR: stats + correlaciones
    # ---------------------------
    lr_summaries: list[VarSummary] = []
    for name in lr_var_names:
        da = lr.sel(variable=name)
        lr_summaries.append(
            VarSummary(
                name=name,
                nan_frac=float(da.isnull().mean().compute().item()),
                mean=float(da.mean().compute().item()),
                std=float(da.std().compute().item()),
                vmin=float(da.min().compute().item()),
                vmax=float(da.max().compute().item()),
            )
        )

    lr_samples = (
        lr.transpose("time", "latitude_lr", "longitude_lr", "variable")
        .stack(sample=("time", "latitude_lr", "longitude_lr"))
        .transpose("sample", "variable")
        .compute()
        .values
    )
    corr_lr = _corrcoef_matrix(lr_samples)

    lr_mean_ts = lr.mean(dim=("latitude_lr", "longitude_lr")).compute().values  # (time, var)
    corr_lr_vs_hr_time_mean = {n: _safe_pearsonr(lr_mean_ts[:, i], hr_mean_ts) for i, n in enumerate(lr_var_names)}

    hr_on_lr = hr.interp(latitude=ds_cache["latitude_lr"], longitude=ds_cache["longitude_lr"])
    hr_on_lr_flat = hr_on_lr.stack(sample=("time", "latitude_lr", "longitude_lr")).compute().values
    corr_lr_vs_hr_local = {n: _safe_pearsonr(lr_samples[:, i], hr_on_lr_flat) for i, n in enumerate(lr_var_names)}

    # ---------------------------
    # Estáticos: stats + correlaciones + VIF
    # ---------------------------
    static_flat = static_processed.reshape(-1, static_processed.shape[-1])
    static_summaries: list[VarSummary] = []
    for i, name in enumerate(static_var_names):
        x = static_flat[:, i]
        static_summaries.append(
            VarSummary(
                name=name,
                nan_frac=float(np.mean(~np.isfinite(x))),
                mean=float(np.nanmean(x)),
                std=float(np.nanstd(x)),
                vmin=float(np.nanmin(x)),
                vmax=float(np.nanmax(x)),
            )
        )

    corr_static = _corrcoef_matrix(static_flat)
    vif_static = _vif(static_flat, static_var_names)

    hr_clim_flat = hr_clim.reshape(-1)
    residual_mean_flat = residual_mean_map.reshape(-1)
    corr_static_vs_hr_clim = {n: _safe_pearsonr(static_flat[:, i], hr_clim_flat) for i, n in enumerate(static_var_names)}
    corr_static_vs_residual_mean = {
        n: _safe_pearsonr(static_flat[:, i], residual_mean_flat) for i, n in enumerate(static_var_names)
    }

    # ---------------------------
    # Guardar artefactos
    # ---------------------------
    out_corr_lr_csv = REPORTS_DIR / "corr_lr.csv"
    out_corr_static_csv = REPORTS_DIR / "corr_static.csv"
    _write_csv_matrix(corr_lr, lr_var_names, out_corr_lr_csv)
    _write_csv_matrix(corr_static, static_var_names, out_corr_static_csv)

    out_corr_lr_png = REPORTS_DIR / "corr_lr.png"
    out_corr_static_png = REPORTS_DIR / "corr_static.png"
    _maybe_plot_heatmap(corr_lr, lr_var_names, out_corr_lr_png, "LR correlation (Pearson)")
    _maybe_plot_heatmap(corr_static, static_var_names, out_corr_static_png, "Static correlation (Pearson)")

    # Rankings útiles
    lr_rank_local = sorted(((n, abs(corr_lr_vs_hr_local[n])) for n in lr_var_names), key=lambda t: t[1], reverse=True)
    static_rank_resid = sorted(
        ((n, abs(corr_static_vs_residual_mean[n])) for n in static_var_names), key=lambda t: t[1], reverse=True
    )
    static_rank_vif = sorted(((n, vif_static.get(n, np.nan)) for n in static_var_names), key=lambda t: t[1], reverse=True)

    # Pares redundantes
    hi_pairs_lr = _top_pairs(corr_lr, lr_var_names, threshold=0.75, top_k=12)
    hi_pairs_static = _top_pairs(corr_static, static_var_names, threshold=0.80, top_k=20)

    # Heurísticas: candidatos a ablation/eliminación (si se quiere reducir canales)
    static_candidates: list[str] = []
    # 1) Multicolinealidad alta
    static_candidates.extend([n for n, v in static_rank_vif if np.isfinite(v) and v >= 8.0])
    # 2) Señal baja vs residual medio (proxy de sesgo a corregir)
    static_candidates.extend([n for n, v in static_rank_resid if np.isfinite(v) and v < 0.20])
    # 3) Pares muy correlacionados: sugerencias concretas conocidas en este dataset
    # (a) altura: avg_height vs max_levels suele ser redundante
    try:
        ia = static_var_names.index("avg_height")
        im = static_var_names.index("max_levels")
        if np.isfinite(corr_static[ia, im]) and abs(corr_static[ia, im]) >= 0.85:
            static_candidates.append("max_levels")
    except ValueError:
        pass
    # (b) densidad vs rugosidad: normalmente muy redundantes
    try:
        ib = static_var_names.index("building_density")
        ir = static_var_names.index("roughness")
        if np.isfinite(corr_static[ib, ir]) and abs(corr_static[ib, ir]) >= 0.85:
            static_candidates.append("roughness")
    except ValueError:
        pass
    static_candidates = sorted(set(static_candidates))

    lr_candidates: list[str] = [n for n in lr_var_names if abs(corr_lr_vs_hr_local.get(n, np.nan)) < 0.10]

    # ---------------------------
    # Reporte Markdown
    # ---------------------------
    report_md = REPORTS_DIR / "feature_audit.md"
    lines: list[str] = []
    lines.append("# Auditoría de variables (LR / HR / estáticos)")
    lines.append("")
    lines.append("## 1) Inventario y dimensiones")
    lines.append(f"- Cache Zarr: `{PATH_CACHE_ZARR}`")
    lines.append(f"- Static Zarr: `{PATH_STATIC_ZARR}`")
    lines.append(f"- Static cache (.npy): `{PATH_STATIC_PROCESSED_NPY}`")
    lines.append("")
    lines.append(
        f"- HR: time={hr.sizes['time']}, lat={hr.sizes['latitude']}, lon={hr.sizes['longitude']} (var: hr_target)"
    )
    lines.append(
        f"- LR: time={lr.sizes['time']}, lat_lr={lr.sizes['latitude_lr']}, lon_lr={lr.sizes['longitude_lr']}, vars={lr.sizes['variable']}"
    )
    lines.append(f"- Static (cache): y={static_processed.shape[0]}, x={static_processed.shape[1]}, vars={static_processed.shape[2]}")
    lines.append("")
    lines.append("## 2) HR (target)")
    lines.append(f"- NaN frac: {_fmt(hr_nan, 5)}")
    lines.append(f"- Serie dominio-mean(t): mean={_fmt(hr_mean)}, std={_fmt(hr_std)}")
    lines.append("")
    lines.append("## 3) LR — stats y correlación con HR")
    lines.append("")
    lines.append("| var | NaN% | mean | std | min | max | corr(HR mean(t)) | corr(HR@LR local) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in lr_summaries:
        lines.append(
            f"| {s.name} | {100*s.nan_frac:.3f} | {_fmt(s.mean)} | {_fmt(s.std)} | {_fmt(s.vmin)} | {_fmt(s.vmax)}"
            f" | {_fmt(corr_lr_vs_hr_time_mean[s.name])} | {_fmt(corr_lr_vs_hr_local[s.name])} |"
        )
    lines.append("")
    lines.append(f"- Matriz corr LR (CSV): `{out_corr_lr_csv}`")
    if out_corr_lr_png.exists():
        lines.append(f"- Heatmap corr LR (PNG): `{out_corr_lr_png}`")
    if hi_pairs_lr:
        lines.append("- Pares LR con |corr| ≥ 0.75 (top):")
        for a, b, r in hi_pairs_lr:
            lines.append(f"  - {a} vs {b}: r={_fmt(r)}")
    else:
        lines.append("- No se detectaron pares LR con |corr| ≥ 0.75.")
    lines.append("")
    lines.append("- Ranking LR por |corr(HR@LR local)| (top 5):")
    for n, v in lr_rank_local[:5]:
        lines.append(f"  - {n}: {_fmt(v)}")
    lines.append("")
    lines.append("## 4) Estáticos — stats, correlación y VIF")
    lines.append("")
    lines.append("| var | NaN% | mean | std | min | max | corr(HR clim) | corr(residual mean) | VIF |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in static_summaries:
        lines.append(
            f"| {s.name} | {100*s.nan_frac:.3f} | {_fmt(s.mean)} | {_fmt(s.std)} | {_fmt(s.vmin)} | {_fmt(s.vmax)}"
            f" | {_fmt(corr_static_vs_hr_clim[s.name])} | {_fmt(corr_static_vs_residual_mean[s.name])} | {_fmt(vif_static[s.name], 2)} |"
        )
    lines.append("")
    lines.append(f"- Matriz corr estáticos (CSV): `{out_corr_static_csv}`")
    if out_corr_static_png.exists():
        lines.append(f"- Heatmap corr estáticos (PNG): `{out_corr_static_png}`")
    if hi_pairs_static:
        lines.append("- Pares estáticos con |corr| ≥ 0.80 (top):")
        for a, b, r in hi_pairs_static:
            lines.append(f"  - {a} vs {b}: r={_fmt(r)}")
    else:
        lines.append("- No se detectaron pares estáticos con |corr| ≥ 0.80.")
    lines.append("")
    lines.append("- Ranking estáticos por |corr(residual mean)| (top 7):")
    for n, v in static_rank_resid[:7]:
        lines.append(f"  - {n}: {_fmt(v)}")
    lines.append("")
    lines.append("- Ranking estáticos por VIF (top 7):")
    for n, v in static_rank_vif[:7]:
        lines.append(f"  - {n}: {_fmt(v, 2)}")
    lines.append("")
    lines.append("## 5) Notas de interpretación")
    lines.append("- `corr(HR clim)`: relación con el patrón espacial de la temperatura media anual.")
    lines.append(
        "- `corr(residual mean)`: relación con el sesgo medio `HR - upsample(mean(LR t2m))` (proxy de UHI/bias a corregir)."
    )
    lines.append("- VIF alto sugiere redundancia lineal (multicolinealidad) entre estáticos.")
    lines.append("")
    lines.append("## 6) Candidatas para reducir canales (ablation primero)")
    if static_candidates:
        lines.append("- Estáticos: " + ", ".join(static_candidates))
    else:
        lines.append("- Estáticos: (sin candidatas automáticas)")
    if lr_candidates:
        lines.append("- LR: " + ", ".join(lr_candidates))
    else:
        lines.append("- LR: (sin candidatas automáticas por baja correlación lineal)")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"✅ Reporte generado: {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
