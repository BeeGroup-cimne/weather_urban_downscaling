#!/usr/bin/env bash
set -euo pipefail

# Crea un tarball listo para "subir a servidor" sin imágenes, modelos entrenados ni outputs de experiments.
# Incluye la estructura (via .gitkeep) y el código necesario para ejecutar scripts/run_ablation.py.

out="${1:-dist/weather_urban_downscaling_server_bundle.tar.gz}"
root_name="weather_urban_downscaling"

mkdir -p "$(dirname "$out")"
tmp_dir="$(dirname "$out")/tmp_bundle"
rm -rf "$tmp_dir"
mkdir -p "$tmp_dir"
stage="${tmp_dir}/${root_name}"

cleanup() { rm -rf "$tmp_dir"; }
trap cleanup EXIT

copy_file() {
  local src="$1"
  # Usar rsync para mayor robustez, ignorar errores si falla un archivo concreto.
  rsync -R "$src" "$stage/" || echo "⚠️ Warning: Failed to copy $src"
}

filter_tracked() {
  local path="$1"

  # Excluir imágenes (en cualquier parte del repo) y archivos problemáticos
  case "$path" in
    *.png|*.jpg|*.jpeg|*.gif|*.svg|*.pdf|ablation_results.md) return 1 ;;
  esac

  # Excluir checkpoints de notebooks
  case "$path" in
    */.ipynb_checkpoints/*) return 1 ;;
  esac

  # Excluir outputs pesados en experiments/ (pero conservar estructura vía .gitkeep)
  case "$path" in
    experiments/*)
      [[ "$path" == */.gitkeep ]] && return 0
      return 1
      ;;
  esac

  # Excluir datasets/caches en data/ (pero conservar estructura vía .gitkeep)
  case "$path" in
    data/*)
      [[ "$path" == */.gitkeep ]] && return 0
      return 1
      ;;
  esac

  return 0
}

while IFS= read -r -d '' f; do
  if filter_tracked "$f"; then
    copy_file "$f"
  fi
done < <(git ls-files -z)

# Incluir archivos “server-ready” aunque aún no estén trackeados (útil antes de commitear).
extra_files=(
  "SERVER_README.md"
  "docker-compose.server-fullframe.yml"
  "scripts/run_server_fullframe.sh"
  "scripts/run_ablation_tiles_heatwave_server.sh"
  "scripts/run_stations_eval_ablation.sh"
  "scripts/consolidate_experiment1.py"
  "scripts/run_experiment3_fullframe_replica.sh"
  "scripts/consolidate_experiment3.py"
  "scripts/make_server_bundle.sh"
)
for f in "${extra_files[@]}"; do
  if [[ -f "$f" ]] && filter_tracked "$f"; then
    copy_file "$f"
  fi
done

# Asegurar estructura mínima aunque falten .gitkeep por algún motivo
mkdir -p \
  "$stage/data/processed/era5land" \
  "$stage/data/raw" \
  "$stage/experiments/models" \
  "$stage/experiments/logs" \
  "$stage/experiments/figures"

tar -C "$tmp_dir" -czf "$out" "$root_name"
echo "✅ Bundle creado: $out"
