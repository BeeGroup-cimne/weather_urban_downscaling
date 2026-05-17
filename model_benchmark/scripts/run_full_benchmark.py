#!/usr/bin/env python3
"""
Benchmark: Claude Code (MCP) vs OpenCode free models.
Runs 3 tasks: claim extraction, integrity check, multi-paper synthesis.
"""
import json, subprocess, sys, time, csv, os
from datetime import datetime
from pathlib import Path

BENCHMARK_DIR = Path("/Users/kerincardona/weather_urban_downscaling/model_benchmark")
TESTS_DIR = BENCHMARK_DIR / "tests"
RESULTS_DIR = BENCHMARK_DIR / "results"
OPENCODE_BIN = "/Users/kerincardona/.opencode/bin/opencode"
TASK_TIMEOUT = 300

def load_tasks():
    with open(TESTS_DIR / "test_papers.json") as f:
        td = json.load(f)
    pw, pe = td["papers"][0], td["papers"][1]
    return {
        "claim_extraction": {
            "name": "Claim Extraction",
            "prompt": f"Extract key scientific claims from this abstract. Return each as concise numbered statements with quantitative results.\n\nAbstract:\n{pw['abstract']}",
            "gold": pw["known_claims"],
        },
        "integrity_check": {
            "name": "Integrity Check",
            "prompt": f"Peer review this abstract. Identify methodological gaps, unstated assumptions, and potential issues:\n\n{pw['abstract']}\n\nFor each: what's missing, why it matters, what would resolve it.",
            "gold": pw["known_issues"],
        },
        "multi_paper_synthesis": {
            "name": "Multi-Paper Synthesis",
            "prompt": f"Synthesize the relationship between these papers:\n1. Shared methods, 2. Complementary findings, 3. Research gaps, 4. Unified direction\n\nPaper 1: {pw['title']}\nAbstract: {pw['abstract']}\n\nPaper 2: {pe['title']}\nAbstract: {pe['abstract']}",
            "gold": pw["key_relationships"],
        },
    }

# ─── SCORING ──────────────────────────────────────────────────────────────

def score_claims(response, gold):
    lines = [l.strip().lstrip("0123456789.-) ").strip() for l in response.split("\n") if l.strip() and (l.strip()[0].isdigit() or l.strip().startswith("-"))]
    matches = 0
    for gc in gold:
        gw = set(gc.lower().split())
        for rl in lines:
            rw = set(rl.lower().split())
            if len(gw & rw) >= 3 or gc.lower() in rl.lower() or rl.lower() in gc.lower():
                matches += 1; break
    prec = matches / len(lines) if lines else 0
    rec = matches / len(gold) if gold else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    return {"score": round(min(f1,1.0),3), "prec": round(min(prec,1.0),3), "rec": round(min(rec,1.0),3), "found": len(lines), "expected": len(gold)}

def score_integrity(response, gold):
    detected = 0
    for issue in gold:
        for kw in [w.lower() for w in issue.split() if len(w) > 4][:3]:
            if kw in response.lower():
                detected += 1; break
    return {"score": round(min(detected/len(gold),1.0),3) if gold else 0, "detected": detected, "expected": len(gold)}

def score_synthesis(response, gold):
    mentioned = 0
    for rel in gold:
        for kw in [w.lower() for w in rel.split() if len(w) > 4][:3]:
            if kw in response.lower():
                mentioned += 1; break
    ind = {"complement","combine","unified","gap","inform","bridge","connect"}
    syn = sum(1 for i in ind if i in response.lower()) / len(ind)
    rel_score = mentioned/len(gold) if gold else 0
    return {"score": round(min(0.6*rel_score+0.4*syn,1.0),3), "score_rel": round(rel_score,3), "score_syn": round(syn,3)}

EVAL = {
    "claim_extraction": lambda r, g: score_claims(r, g),
    "integrity_check": lambda r, g: score_integrity(r, g),
    "multi_paper_synthesis": lambda r, g: score_synthesis(r, g),
}

# ─── RUNNERS ──────────────────────────────────────────────────────────────

def run_claude(prompt):
    cmd = ["claude", "-p", prompt, "--max-turns", "1"]
    env = os.environ.copy()
    env["HOME"] = "/Users/kerincardona"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TASK_TIMEOUT, env=env)
        return r.stdout.strip(), r.returncode == 0
    except subprocess.TimeoutExpired:
        return "TIMEOUT", False
    except Exception as e:
        return f"ERROR: {e}", False

def run_opencode(model_id, prompt):
    cmd = [OPENCODE_BIN, "run", "-m", model_id, "--pure", prompt]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TASK_TIMEOUT)
        return r.stdout.strip(), r.returncode == 0
    except subprocess.TimeoutExpired:
        return "TIMEOUT", False
    except Exception as e:
        return f"ERROR: {e}", False

# ─── MAIN ─────────────────────────────────────────────────────────────────

def main():
    tasks = load_tasks()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = RESULTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "timestamp": timestamp,
        "models": ["claude-code", "opencode/qwen3.6-plus-free", "opencode/deepseek-v4-flash-free"],
        "tasks": list(tasks.keys()),
        "scores": {},
    }
    
    print(f"\n{'='*60}")
    print(f"Model Benchmark - {timestamp}")
    print(f"{'='*60}")
    
    # 1. Claude Code
    print(f"\n--- Claude Code ---")
    mkey = "claude-code"
    results["scores"][mkey] = {}
    for tn, td in tasks.items():
        print(f"  [{tn}] ... ", end="", flush=True)
        resp, ok = run_claude(td["prompt"])
        if ok:
            score = EVAL[tn](resp, td["gold"])
            results["scores"][mkey][tn] = score["score"]
            with open(run_dir / f"{mkey}_{tn}.txt", "w") as f: f.write(resp)
            emoji = "OK" if score["score"] >= 0.7 else "WARN" if score["score"] >= 0.4 else "FAIL"
            print(f"[{emoji}] {score['score']:.3f}")
        else:
            print(f"[ERR] {resp[:80]}")
            results["scores"][mkey][tn] = 0.0
    
    # 2. OpenCode models
    for model_id in ["opencode/qwen3.6-plus-free", "opencode/deepseek-v4-flash-free"]:
        short = model_id.replace("opencode/", "")
        print(f"\n--- {short} ---")
        results["scores"][short] = {}
        for tn, td in tasks.items():
            print(f"  [{tn}] ... ", end="", flush=True)
            resp, ok = run_opencode(model_id, td["prompt"])
            if ok:
                score = EVAL[tn](resp, td["gold"])
                results["scores"][short][tn] = score["score"]
                with open(run_dir / f"{short}_{tn}.txt", "w") as f: f.write(resp)
                emoji = "OK" if score["score"] >= 0.7 else "WARN" if score["score"] >= 0.4 else "FAIL"
                print(f"[{emoji}] {score['score']:.3f}")
            else:
                print(f"[ERR] {resp[:80]}")
                results["scores"][short][tn] = 0.0
            time.sleep(2)
    
    # Save & summary
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for mk in results["scores"]:
        scores = [s for s in results["scores"][mk].values() if isinstance(s, float)]
        avg = sum(scores)/len(scores) if scores else 0
        print(f"  {mk}: avg={avg:.3f}")
        for tn in tasks:
            s = results["scores"][mk].get(tn, "?")
            print(f"    {tn}: {s:.3f}" if isinstance(s, float) else f"    {tn}: {s}")
    print(f"\nResults: {run_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
