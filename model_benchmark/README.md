# Model Benchmark - PhD Research

Benchmark rotativo semanal para evaluar modelos LLM en tareas específicas del PhD.

## Configuración rápida

1. Copia el config template:
```bash
cp config.json.example config.json
```

2. Edita `config.json` con tu API key de OpenRouter

3. Ejecuta el benchmark:
```bash
python3 scripts/run_benchmark.py
```

## Ejecución

### Benchmark completo
```bash
python3 scripts/run_benchmark.py
```

### Solo modelos específicos
```bash
python3 scripts/run_benchmark.py --models "anthropic/claude-sonnet-latest,qwen/qwen3.6-plus"
```

### Solo una tarea
```bash
python3 scripts/run_benchmark.py --task claim_extraction
```

## Tareas evaluadas

| Tarea | Qué mide | Paper de prueba |
|-------|----------|-----------------|
| claim_extraction | Precisión extrayendo claims científicos | Urban wind field prediction |
| integrity_check | Detección de gaps metodológicos | Urban wind field prediction |
| obsidian_note | Formato y fidelidad de nota Obsidian | Urban wind field prediction |
| multi_paper_synthesis | Síntesis cruzada entre papers | Wind field + Climate extremes |
| json_structured | Output JSON válido y correcto | Urban wind field prediction |

## Estructura de resultados

```
results/
  YYYY-MM-DD_HHMM/
    results.json          # Scores detallados
    summary.md            # Resumen legible
    trend.csv             # Histórico acumulativo (raíz)
    *_taskname.txt        # Respuestas crudas por modelo/tarea
```

## Cron automático

El benchmark se ejecuta automáticamente cada lunes a las 9:00 AM.
Para modificar el schedule, edita el cronjob en Hermes.
