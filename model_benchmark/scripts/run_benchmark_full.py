#!/usr/bin/env python3
"""
Model Benchmark — Claude Code via MCP SDK + OpenCode
=====================================================
Evaluates LLMs on 5 PhD research tasks using:
  - Claude Code (via MCP protocol, Python SDK)
  - OpenCode Go (free models)

Usage:
  python3 run_benchmark_full.py [--models opencode,claude] [--task claim_extraction]
"""
import asyncio
import json
import csv
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

# MCP SDK
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ─── CONFIG ───────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path("/Users/kerincardona/weather_urban_downscaling/model_benchmark")
TESTS_DIR = BENCHMARK_DIR / "tests"
RESULTS_DIR = BENCHMARK_DIR / "results"
OPENCODE_BIN = os.path.expanduser("~/.opencode/bin/opencode")

TASK_TIMEOUT = 300  # seconds per task
OPENCODE_MODELS = [
    "opencode/qwen3.6-plus-free",
    "opencode/deepseek-v4-flash-free",
]

# ─── TASK DEFINITIONS ─────────────────────────────────────────────────────

def load_tasks():
    with open(TESTS_DIR / "test_papers.json") as f:
        td = json.load(f)
    pw = td["papers"][0]
    pe = td["papers"][1]
    return {
        "claim_extraction": {
            "name": "Claim Extraction",
            "prompt": f"""Extract key scientific claims from this abstract. Return each as concise numbered statements with quantitative results.

Abstract:
{pw['abstract']}""",
            "gold_standard": pw["known_claims"],
        },
        "integrity_check": {
            "name": "Integrity Check",
            "prompt": f"""Peer review this abstract. Identify methodological gaps, unstated assumptions, and potential issues:

{pw['abstract']}

For each: what's missing, why it matters, what would resolve it.""",
            "gold_standard": pw["known_issues"],
        },
        "multi_paper_synthesis": {
            "name": "Multi-Paper Synthesis",
            "prompt": f"""Synthesize the relationship between these papers:
1. Shared methods, 2. Complementary findings, 3. Research gaps, 4. Unified direction

Paper 1: {pw['title']}
Abstract: {pw['abstract']}

Paper 2: {pe['title']}
Abstract: {pe['abstract']}""",
            "gold_standard": pw["key_relationships"],
        },
    }

# ─── EVALUATION ───────────────────────────────────────────────────────────

def score_claim_extraction(response: str, gold: list) -> dict:
    resp_lines = [l.strip().lstrip("0123456789.-) ").strip() for l in response.split("\n") if l.strip() and (l.strip()[0].isdigit() or l.strip().startswith("-"))]
    matches = 0
    for gc in gold:
        gw = set(gc.lower().split())
        for rl in resp_lines:
            rw = set(rl.lower().split())
            if len(gw & rw) >= 3 or gc.lower() in rl.lower() or rl.lower() in gc.lower():
                matches += 1
                break
    prec = matches / len(resp_lines) if resp_lines else 0
    rec = matches / len(gold) if gold else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    return {"score": round(min(f1,1.0),3), "precision": round(min(prec,1.0),3), "recall": round(min(rec,1.0),3), "matches": matches, "expected": len(gold)}

def score_integrity(response: str, gold: list) -> dict:
    detected = 0
    for issue in gold:
        for kw in [w.lower() for w in issue.split() if len(w) > 4][:3]:
            if kw in response.lower():
                detected += 1
                break
    return {"score": round(min(detected/len(gold),1.0),3) if gold else 0, "detected": detected, "expected": len(gold)}

def score_synthesis(response: str, gold: list) -> dict:
    mentioned = 0
    for rel in gold:
        for kw in [w.lower() for w in rel.split() if len(w) > 4][:3]:
            if kw in response.lower():
                mentioned += 1
                break
    indicators = sum(1 for ind in ["complement","combine","unified","gap","inform","bridge","connect"] if ind in response.lower())/7
    rel_score = mentioned/len(gold) if gold else 0
    return {"score": round(min(0.6*rel_score+0.4*indicators,1.0),3), "relationships": mentioned, "expected": len(gold), "synthesis": round(indicators,3)}

EVALUATORS = {
    "claim_extraction": lambda r, t: score_claim_extraction(r, t["gold_standard"]),
    "integrity_check": lambda r, t: score_integrity(r, t["gold_standard"]),
    "multi_paper_synthesis": lambda r, t: score_synthesis(r, t["gold_standard"]),
}

# ─── CLAUDE CODE VIA MCP ─────────────────────────────────────────────────

class ClaudeMCPRunner:
    """Runs prompts through Claude Code via MCP SDK."""
    
    async def connect(self):
        self.server_params = StdioServerParameters(
            command="claude",
            args=["mcp", "serve"],
        )
        self.stdio_ctx = stdio_client(self.server_params)
        self.reader, self.writer = await self.stdio_ctx.__aenter__()
        self.session = ClientSession(self.reader, self.writer)
        await self.session.__aenter__()
        await self.session.initialize()
        return self
    
    async def disconnect(self):
        await self.session.__aexit__(None, None, None)
        await self.stdio_ctx.__aexit__(None, None, None)
    
    async def run_prompt(self, prompt: str, task_name: str) -> str:
        """Run a prompt via Claude Code Agent tool."""
        result = await self.session.call_tool("Agent", {
            "description": f"PhD task: {task_name}",
            "prompt": prompt,
        })
        content = result.content if hasattr(result, 'content') else []
        texts = [c.text for c in content if hasattr(c, 'type') and c.type == 'text']
        return "\n".join(texts) if texts else str(content)

# ─── OPENCODE RUNNER ─────────────────────────────────────────────────────

def opencode_call(model_id: str, prompt: str) -> tuple:
    cmd = [OPENCODE_BIN, "run", "-m", model_id, "--pure", prompt]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TASK_TIMEOUT)
        if r.returncode != 0:
            return f"Error: {r.stderr[:300]}", True
        return r.stdout.strip(), False
    except subprocess.TimeoutExpired:
        return "Error: Timeout", True
    except Exception as e:
        return f"Error: {str(e)}", True

# ─── MAIN ─────────────────────────────────────────────────────────────────

async def run_benchmark(models: list = None, task_filter: str = None):
    models = models or ["opencode", "claude"]
    tasks = load_tasks()
    if task_filter:
        tasks = {k: v for k, v in tasks.items() if k == task_filter}
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = RESULTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    results = {"timestamp": timestamp, "models": models, "tasks": list(tasks.keys()), "scores": {}, "errors": {}}
    
    print(f"\n{'='*60}")
    print(f"Model Benchmark - {timestamp}")
    print(f"Models: {models} | Tasks: {list(tasks.keys())}")
    print(f"{'='*60}\n")
    
    # Claude Code via MCP
    if "claude" in models:
        results["scores"]["claude-code"] = {}
        results["errors"]["claude-code"] = []
        print(f"\n--- Claude Code (MCP) ---")
        
        try:
            claude = ClaudeMCPRunner()
            await claude.connect()
            try:
                for task_name, task_def in tasks.items():
                    print(f"  [{task_name}] ... ", end="", flush=True)
                    try:
                        response = await asyncio.wait_for(
                            claude.run_prompt(task_def["prompt"], task_def["name"]),
                            timeout=TASK_TIMEOUT
                        )
                        eval_fn = EVALUATORS[task_name]
                        score = eval_fn(response, task_def)
                        results["scores"]["claude-code"][task_name] = score["score"]
                        
                        with open(run_dir / f"claude-code_{task_name}.txt", "w") as f:
                            f.write(response)
                        
                        emoji = "OK" if score["score"] >= 0.7 else "WARN" if score["score"] >= 0.4 else "FAIL"
                        print(f"[{emoji}] {score['score']:.3f}")
                    except Exception as e:
                        print(f"[ERR] {str(e)[:80]}")
                        results["errors"]["claude-code"].append({"task": task_name, "error": str(e)})
                        results["scores"]["claude-code"][task_name] = 0.0
            finally:
                await claude.disconnect()
        except Exception as e:
            print(f"  MCP CONNECTION FAILED: {str(e)[:200]}")
            results["errors"]["claude-code"].append({"task": "connection", "error": str(e)})
            for task_name in tasks:
                results["scores"]["claude-code"][task_name] = 0.0
    
    # OpenCode models
    if "opencode" in models:
        for model_id in OPENCODE_MODELS:
            short = model_id.replace("opencode/", "")
            results["scores"][short] = {}
            results["errors"][short] = []
            print(f"\n--- {model_id} ---")
            
            for task_name, task_def in tasks.items():
                print(f"  [{task_name}] ... ", end="", flush=True)
                response, is_error = opencode_call(model_id, task_def["prompt"])
                
                if is_error:
                    print(f"[ERR] {response[:80]}")
                    results["errors"][short].append({"task": task_name, "error": response})
                    results["scores"][short][task_name] = 0.0
                    continue
                
                eval_fn = EVALUATORS[task_name]
                score = eval_fn(response, task_def)
                results["scores"][short][task_name] = score["score"]
                
                with open(run_dir / f"{short}_{task_name}.txt", "w") as f:
                    f.write(response)
                
                emoji = "OK" if score["score"] >= 0.7 else "WARN" if score["score"] >= 0.4 else "FAIL"
                print(f"[{emoji}] {score['score']:.3f}")
                time.sleep(2)
    
    # Save results
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_key in results["scores"]:
        scores = [s for s in results["scores"][model_key].values() if isinstance(s, float)]
        avg = sum(scores)/len(scores) if scores else 0
        print(f"  {model_key}: avg={avg:.3f} ({len(scores)} tasks)")
    print(f"\nResults: {run_dir}")
    print(f"{'='*60}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="claude,opencode")
    parser.add_argument("--task", type=str)
    args = parser.parse_args()
    
    models = [m.strip() for m in args.models.split(",")]
    asyncio.run(run_benchmark(models=models, task_filter=args.task))
