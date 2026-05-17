#!/usr/bin/env python3
"""
Model Benchmark for PhD Research Tasks
========================================
Runs 5 tasks against multiple LLMs via OpenRouter, evaluates outputs,
and tracks trends over time.

Usage: python run_benchmark.py [--models model1,model2] [--task task_name]
"""

import json
import os
import sys
import time
import subprocess
import csv
from datetime import datetime
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Fixed paths (home resolution differs in sandbox vs terminal)
BENCHMARK_DIR = Path("/Users/kerincardona/weather_urban_downscaling/model_benchmark")
TESTS_DIR = BENCHMARK_DIR / "tests"
RESULTS_DIR = BENCHMARK_DIR / "results"
# Try multiple sources for API key
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    # Try reading from local config file
    config_file = BENCHMARK_DIR / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
            OPENROUTER_API_KEY = config.get("openrouter_api_key", "")
    # Try reading from hermes config
    if not OPENROUTER_API_KEY:
        hermes_config = Path("/Users/kerincardona/.hermes/config.yaml")
        if hermes_config.exists():
            import re
            with open(hermes_config) as f:
                content = f.read()
                match = re.search(r'openrouter:\s*\n\s*api_key:\s*["\']?([^"\'\s\n]+)', content)
                if match:
                    OPENROUTER_API_KEY = match.group(1)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default models to evaluate (can be overridden via --models)
DEFAULT_MODELS = [
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5.4",
    "qwen/qwen3.6-plus",
    "deepseek/deepseek-v4-flash",
    "google/gemini-3.1-flash-lite",
    "x-ai/grok-4.20",
]

# Task timeout in seconds
TASK_TIMEOUT = 120

# ─── TASK DEFINITIONS ─────────────────────────────────────────────────────────

def get_task_definitions():
    """Return all benchmark tasks with their prompts and evaluation logic."""
    
    # Load test papers
    with open(TESTS_DIR / "test_papers.json") as f:
        test_data = json.load(f)
    
    paper_wind = test_data["papers"][0]
    paper_extremes = test_data["papers"][1]
    
    return {
        "claim_extraction": {
            "name": "Claim Extraction",
            "description": "Extract key scientific claims from an abstract",
            "prompt": f"""Extract the key scientific claims from this abstract. Return each claim as a concise, standalone statement.

Abstract:
{paper_wind['abstract']}

Format your response as a numbered list. Each claim should be:
1. Specific and testable
2. Include quantitative results where available
3. Distinguish between method claims and result claims""",
            "gold_standard": paper_wind["known_claims"],
            "evaluate": evaluate_claim_extraction,
        },
        "integrity_check": {
            "name": "Integrity Check",
            "description": "Identify methodological gaps and potential issues",
            "prompt": f"""Act as a peer reviewer. Identify methodological gaps, unstated assumptions, and potential issues in this abstract:

{paper_wind['abstract']}

For each issue, specify:
- What is missing or unclear
- Why it matters for the validity of the claims
- What additional information would resolve it

Be specific and critical.""",
            "gold_standard": paper_wind["known_issues"],
            "evaluate": evaluate_integrity_check,
        },
        "obsidian_note": {
            "name": "Obsidian Note Generation",
            "description": "Generate a properly formatted Obsidian research note",
            "prompt": f"""Generate an Obsidian research note for this paper following this exact format:

---
title: "{paper_wind['title']}"
authors: []
year: 
doi: 
tags: [urban-climate, ML, wind-field]
status: to-read
---

## Summary
[2-3 sentence summary]

## Method
[Method description]

## Key Results
[Bullet list of results]

## Limitations
[Identified limitations]

## Relation to Thesis
[How this connects to urban climate downscaling and fluid dynamics]

Abstract:
{paper_wind['abstract']}""",
            "gold_standard": {
                "required_sections": ["## Summary", "## Method", "## Key Results", "## Limitations", "## Relation to Thesis"],
                "required_frontmatter": ["title", "authors", "year", "doi", "tags", "status"],
                "min_length": 300,
            },
            "evaluate": evaluate_obsidian_note,
        },
        "multi_paper_synthesis": {
            "name": "Multi-Paper Synthesis",
            "description": "Synthesize relationships between two papers",
            "prompt": f"""Synthesize the relationship between these two papers. Identify:
1. Shared methodological approaches
2. Complementary findings
3. Research gaps that emerge from combining both
4. How they could inform a unified research direction

Paper 1: {paper_wind['title']}
Abstract: {paper_wind['abstract']}

Paper 2: {paper_extremes['title']}
Abstract: {paper_extremes['abstract']}""",
            "gold_standard": paper_wind["key_relationships"],
            "evaluate": evaluate_synthesis,
        },
        "json_structured": {
            "name": "JSON Structured Output",
            "description": "Extract paper metadata as valid JSON",
            "prompt": f"""Extract the following information from this abstract and return ONLY valid JSON matching the schema below. No markdown, no explanation, just JSON.

Abstract: {paper_wind['abstract']}

Required JSON schema:
{{
  "method": "string - the main method/approach",
  "domain": "string - application domain",
  "metrics": ["array of metric names mentioned"],
  "improvement": "string - quantitative improvement claimed",
  "comparison_baseline": "string - what it was compared against",
  "novelty": "string - what is claimed as novel"
}}""",
            "gold_standard": {
                "required_keys": ["method", "domain", "metrics", "improvement", "comparison_baseline", "novelty"],
                "expected_method_keywords": ["GNN", "graph neural network", "auto-encoder", "autoencoder"],
                "expected_domain_keywords": ["urban", "wind"],
                "expected_improvement_pattern": ["50%", "RMSE", "root mean square"],
            },
            "evaluate": evaluate_json_structured,
        },
    }


# ─── EVALUATION FUNCTIONS ─────────────────────────────────────────────────────

def evaluate_claim_extraction(response: str, gold: dict) -> dict:
    """Evaluate claim extraction against gold standard claims."""
    gold_claims = gold["gold_standard"]
    
    # Parse response claims (numbered list)
    response_claims = []
    for line in response.strip().split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            # Remove number/bullet prefix
            claim = line.lstrip('0123456789.-) ').strip()
            if claim:
                response_claims.append(claim)
    
    # Score: how many gold claims are covered (semantic match via keyword overlap)
    matches = 0
    for gold_claim in gold_claims:
        gold_words = set(gold_claim.lower().split())
        for resp_claim in response_claims:
            resp_words = set(resp_claim.lower().split())
            overlap = len(gold_words & resp_words)
            if overlap >= 3 or gold_claim.lower() in resp_claim.lower() or resp_claim.lower() in gold_claim.lower():
                matches += 1
                break
    
    precision = matches / len(response_claims) if response_claims else 0
    recall = matches / len(gold_claims) if gold_claims else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "score": round(min(f1, 1.0), 3),  # Cap at 1.0
        "precision": round(min(precision, 1.0), 3),
        "recall": round(min(recall, 1.0), 3),
        "claims_found": len(response_claims),
        "claims_expected": len(gold_claims),
        "matches": matches,
    }


def evaluate_integrity_check(response: str, gold: dict) -> dict:
    """Evaluate integrity check against known issues."""
    known_issues = gold["gold_standard"]
    
    # Score: how many known issues were identified
    detected = 0
    for issue in known_issues:
        issue_keywords = [w.lower() for w in issue.split() if len(w) > 4]
        for keyword in issue_keywords[:3]:  # Check first 3 meaningful words
            if keyword in response.lower():
                detected += 1
                break
    
    score = detected / len(known_issues) if known_issues else 0
    
    return {
        "score": round(score, 3),
        "issues_detected": detected,
        "issues_expected": len(known_issues),
        "response_length": len(response),
    }


def evaluate_obsidian_note(response: str, gold: dict) -> dict:
    """Evaluate Obsidian note format and content."""
    gs = gold["gold_standard"]
    scores = {}
    
    # Check frontmatter
    has_frontmatter = response.startswith("---") and response.count("---") >= 2
    scores["frontmatter_format"] = 1.0 if has_frontmatter else 0.0
    
    # Check required sections
    sections_found = sum(1 for s in gs["required_sections"] if s in response)
    scores["sections"] = sections_found / len(gs["required_sections"])
    
    # Check required frontmatter fields
    frontmatter_block = response.split("---")[1] if has_frontmatter else ""
    fields_found = sum(1 for f in gs["required_frontmatter"] if f.lower() in frontmatter_block.lower())
    scores["frontmatter_fields"] = fields_found / len(gs["required_frontmatter"])
    
    # Check minimum length
    scores["min_length"] = 1.0 if len(response) >= gs["min_length"] else len(response) / gs["min_length"]
    
    # Overall score (weighted average)
    weights = {"frontmatter_format": 0.2, "sections": 0.4, "frontmatter_fields": 0.2, "min_length": 0.2}
    overall = sum(scores[k] * weights[k] for k in weights)
    
    return {
        "score": round(overall, 3),
        "details": {k: round(v, 3) for k, v in scores.items()},
        "response_length": len(response),
    }


def evaluate_synthesis(response: str, gold: dict) -> dict:
    """Evaluate multi-paper synthesis."""
    relationships = gold["gold_standard"]
    
    # Check if key relationships are mentioned
    mentioned = 0
    for rel in relationships:
        rel_keywords = [w.lower() for w in rel.split() if len(w) > 4]
        for keyword in rel_keywords[:3]:
            if keyword in response.lower():
                mentioned += 1
                break
    
    # Check for synthesis quality indicators
    synthesis_indicators = ["complement", "combine", "unified", "gap", "inform", "bridge", "connect"]
    synthesis_score = sum(1 for ind in synthesis_indicators if ind in response.lower()) / len(synthesis_indicators)
    
    relationship_score = mentioned / len(relationships) if relationships else 0
    
    overall = 0.6 * relationship_score + 0.4 * synthesis_score
    
    return {
        "score": round(overall, 3),
        "relationships_mentioned": mentioned,
        "relationships_expected": len(relationships),
        "synthesis_indicators": round(synthesis_score, 3),
    }


def evaluate_json_structured(response: str, gold: dict) -> dict:
    """Evaluate JSON structured output."""
    gs = gold["gold_standard"]
    
    # Try to parse JSON
    try:
        # Clean response (remove markdown code blocks if present)
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
            cleaned = cleaned.rsplit("```", 1)[0] if "```" in cleaned else cleaned
        cleaned = cleaned.strip()
        
        data = json.loads(cleaned)
    except (json.JSONDecodeError, Exception):
        return {
            "score": 0.0,
            "valid_json": False,
            "error": "Failed to parse JSON",
        }
    
    # Check required keys
    keys_found = sum(1 for k in gs["required_keys"] if k in data)
    key_score = keys_found / len(gs["required_keys"])
    
    # Check expected content
    method_match = any(kw.lower() in data.get("method", "").lower() for kw in gs["expected_method_keywords"])
    domain_match = any(kw.lower() in data.get("domain", "").lower() for kw in gs["expected_domain_keywords"])
    
    improvement_text = data.get("improvement", "").lower()
    improvement_match = any(kw.lower() in improvement_text for kw in gs["expected_improvement_pattern"])
    
    content_score = (method_match + domain_match + improvement_match) / 3
    
    overall = 0.4 * key_score + 0.6 * content_score
    
    return {
        "score": round(overall, 3),
        "valid_json": True,
        "keys_found": keys_found,
        "keys_expected": len(gs["required_keys"]),
        "method_match": method_match,
        "domain_match": domain_match,
        "improvement_match": improvement_match,
    }


# ─── MODEL API CALL ───────────────────────────────────────────────────────────

def call_model(model_id: str, prompt: str) -> tuple:
    """Call a model via OpenRouter API. Returns (response_text, cost_info)."""
    
    cmd = [
        "curl", "-s", "-X", "POST", OPENROUTER_URL,
        "-H", f"Authorization: Bearer {OPENROUTER_API_KEY}",
        "-H", "Content-Type: application/json",
        "-H", "HTTP-Referer: https://phd-research.upc.edu",
        "-H", "X-Title: PhD Model Benchmark",
        "-d", json.dumps({
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,  # Low temp for consistent evaluation
            "max_tokens": 2000,
        }),
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TASK_TIMEOUT)
        if result.returncode != 0:
            return f"Error: {result.stderr}", {"error": result.stderr}
        
        response = json.loads(result.stdout)
        
        # Extract response text
        choices = response.get("choices", [])
        if not choices:
            return f"Error: No choices in response: {result.stdout[:200]}", {"error": "no_choices"}
        
        text = choices[0].get("message", {}).get("content", "")
        
        # Extract cost info
        usage = response.get("usage", {})
        cost = response.get("cost", {})
        
        cost_info = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "cost_usd": cost if isinstance(cost, (int, float)) else 0,
        }
        
        return text, cost_info
        
    except subprocess.TimeoutExpired:
        return "Error: Timeout", {"error": "timeout"}
    except Exception as e:
        return f"Error: {str(e)}", {"error": str(e)}


# ─── MAIN BENCHMARK RUNNER ────────────────────────────────────────────────────

def run_benchmark(models=None, task_filter=None):
    """Run the full benchmark."""
    
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)
    
    models = models or DEFAULT_MODELS
    tasks = get_task_definitions()
    if task_filter:
        tasks = {k: v for k, v in tasks.items() if k == task_filter}
    
    # Create results directory for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = RESULTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Results structure
    all_results = {
        "timestamp": timestamp,
        "models": models,
        "tasks": list(tasks.keys()),
        "scores": {},
        "costs": {},
        "errors": {},
    }
    
    total_tasks = len(models) * len(tasks)
    current_task = 0
    
    print(f"\n{'='*60}")
    print(f"Model Benchmark - {timestamp}")
    print(f"Models: {len(models)} | Tasks: {len(tasks)} | Total: {total_tasks}")
    print(f"{'='*60}\n")
    
    for model_id in models:
        all_results["scores"][model_id] = {}
        all_results["costs"][model_id] = {}
        all_results["errors"][model_id] = []
        
        print(f"\n🔬 Model: {model_id}")
        print(f"{'-'*50}")
        
        for task_name, task_def in tasks.items():
            current_task += 1
            print(f"  [{current_task}/{total_tasks}] {task_def['name']}... ", end="", flush=True)
            
            # Call model
            response, cost_info = call_model(model_id, task_def["prompt"])
            
            # Check for errors
            if response.startswith("Error:"):
                print(f"❌ {response}")
                all_results["errors"][model_id].append({
                    "task": task_name,
                    "error": response,
                })
                all_results["scores"][model_id][task_name] = 0.0
                all_results["costs"][model_id][task_name] = cost_info
                continue
            
            # Evaluate
            eval_result = task_def["evaluate"](response, task_def)
            score = eval_result["score"]
            
            all_results["scores"][model_id][task_name] = score
            all_results["costs"][model_id][task_name] = cost_info
            
            # Store raw response for debugging
            resp_file = run_dir / f"{model_id.replace('/', '_')}_{task_name}.txt"
            with open(resp_file, "w") as f:
                f.write(response)
            
            emoji = "✅" if score >= 0.7 else "⚠️" if score >= 0.4 else "❌"
            print(f"{emoji} {score:.3f}")
            
            # Rate limiting
            time.sleep(2)
    
    # ─── SAVE RESULTS ─────────────────────────────────────────────────────
    
    # Save detailed results
    with open(run_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary
    summary = generate_summary(all_results, tasks)
    with open(run_dir / "summary.md", "w") as f:
        f.write(summary)
    
    # Update trend CSV
    update_trend_csv(all_results)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {run_dir}")
    print(f"Trend updated: {RESULTS_DIR / 'trend.csv'}")
    print(f"{'='*60}")
    
    return all_results


def generate_summary(results, tasks):
    """Generate a human-readable summary."""
    
    lines = [
        f"# Model Benchmark Summary - {results['timestamp']}",
        "",
        "## Scores by Model and Task",
        "",
    ]
    
    # Header
    models = results["models"]
    task_names = list(tasks.keys())
    
    lines.append("| Task | " + " | ".join(m.split("/")[-1] for m in models) + " |")
    lines.append("|------|" + "|".join(["------"] * len(models)) + "|")
    
    for task_name in task_names:
        row = [tasks[task_name]["name"]]
        for model_id in models:
            score = results["scores"].get(model_id, {}).get(task_name, "N/A")
            if isinstance(score, float):
                row.append(f"{score:.3f}")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")
    
    # Averages
    lines.append("")
    lines.append("## Average Score by Model")
    lines.append("")
    
    averages = {}
    for model_id in models:
        scores = [s for s in results["scores"].get(model_id, {}).values() if isinstance(s, float)]
        avg = sum(scores) / len(scores) if scores else 0
        averages[model_id] = avg
    
    for model_id, avg in sorted(averages.items(), key=lambda x: -x[1]):
        lines.append(f"- **{model_id}**: {avg:.3f}")
    
    # Winner
    winner = max(averages.items(), key=lambda x: x[1])
    lines.append("")
    lines.append(f"### 🏆 Overall Winner: {winner[0]} ({winner[1]:.3f})")
    
    # Costs
    lines.append("")
    lines.append("## Costs")
    lines.append("")
    
    for model_id in models:
        total_cost = sum(
            c.get("cost_usd", 0) 
            for c in results["costs"].get(model_id, {}).values() 
            if isinstance(c, dict)
        )
        total_tokens = sum(
            c.get("total_tokens", 0) 
            for c in results["costs"].get(model_id, {}).values() 
            if isinstance(c, dict)
        )
        lines.append(f"- **{model_id}**: ${total_cost:.4f} ({total_tokens} tokens)")
    
    # Errors
    errors = {k: v for k, v in results["errors"].items() if v}
    if errors:
        lines.append("")
        lines.append("## Errors")
        lines.append("")
        for model_id, errs in errors.items():
            lines.append(f"- **{model_id}**: {len(errs)} errors")
            for e in errs:
                lines.append(f"  - {e['task']}: {e['error'][:100]}")
    
    return "\n".join(lines)


def update_trend_csv(results):
    """Append this run's results to the trend CSV."""
    
    trend_file = RESULTS_DIR / "trend.csv"
    
    # Build row
    row = {"timestamp": results["timestamp"]}
    
    for model_id in results["models"]:
        scores = results["scores"].get(model_id, {})
        avg = sum(s for s in scores.values() if isinstance(s, float))
        avg /= len([s for s in scores.values() if isinstance(s, float)]) if any(isinstance(s, float) for s in scores.values()) else 0
        row[f"{model_id}_avg"] = round(avg, 4)
        
        for task_name, score in scores.items():
            if isinstance(score, float):
                row[f"{model_id}_{task_name}"] = score
    
    # Write CSV
    if not trend_file.exists():
        with open(trend_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
    else:
        with open(trend_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)


# ─── CLI ENTRY POINT ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PhD Model Benchmark")
    parser.add_argument("--models", type=str, help="Comma-separated model IDs")
    parser.add_argument("--task", type=str, help="Run only one task")
    args = parser.parse_args()
    
    models = args.models.split(",") if args.models else None
    task = args.task if args.task else None
    
    run_benchmark(models=models, task_filter=task)
