#!/usr/bin/env python3
"""
Claude Code Benchmark Runner — via MCP protocol direct connection.

Connects to Claude Code's MCP server and uses it as a benchmark model.
"""
import json
import subprocess
import sys
import time
import signal
from pathlib import Path

BENCHMARK_DIR = Path("/Users/kerincardona/weather_urban_downscaling/model_benchmark")
TESTS_DIR = BENCHMARK_DIR / "tests"

class ClaudeMCPClient:
    """Minimal MCP client connected to `claude mcp serve` via stdio."""
    
    def __init__(self):
        self.proc = subprocess.Popen(
            ["claude", "mcp", "serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._msg_id = 0
        self._init()
    
    def _send(self, method, params=None):
        self._msg_id += 1
        msg = {
            "jsonrpc": "2.0",
            "id": self._msg_id,
            "method": method,
        }
        if params:
            msg["params"] = params
        line = json.dumps(msg) + "\n"
        self.proc.stdin.write(line)
        self.proc.stdin.flush()
        resp = self.proc.stdout.readline()
        return json.loads(resp.strip())
    
    def _init(self):
        # Initialize
        init = self._send("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "benchmark-runner", "version": "1.0"}
        })
        assert "result" in init, f"Init failed: {init}"
        # Notify initialized
        self._send("notifications/initialized")
        # Discover tools
        tools = self._send("tools/list")
        self.tools = {t["name"]: t for t in tools["result"]["tools"]}
        print(f"[MCP] Connected. Tools available: {len(self.tools)}", file=sys.stderr)
        for tname in sorted(self.tools):
            print(f"  - {tname}", file=sys.stderr)
    
    def call_tool(self, name, arguments=None):
        """Call an MCP tool and return the result."""
        return self._send("tools/call", {
            "name": name,
            "arguments": arguments or {}
        })
    
    def run_prompt(self, prompt, agent_type="general-purpose"):
        """Run a prompt through Claude Code's Agent tool."""
        print(f"\n[MCP] Calling Agent (type={agent_type})...", file=sys.stderr)
        result = self.call_tool("Agent", {
            "subagent_type": "general-purpose",
            "prompt": prompt + "\n\nRespond with the EXACT output, no extra commentary."
        })
        if "error" in result:
            error_detail = result["error"]
            print(f"[MCP] Agent error: {error_detail}", file=sys.stderr)
            if "Available agents" in str(error_detail):
                # Extract agent types
                agents_str = str(error_detail)
                print(f"[MCP] Available agents in error: {agents_str}", file=sys.stderr)
            return f"Error: {error_detail}"
        return str(result.get("result", result))
    
    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=5)


def main():
    # Load test papers
    with open(TESTS_DIR / "test_papers.json") as f:
        test_data = json.load(f)
    
    paper_wind = test_data["papers"][0]
    paper_extremes = test_data["papers"][1]
    
    print(f"Papers loaded: {paper_wind['title'][:50]}...", file=sys.stderr)
    
    # Connect to Claude MCP
    client = ClaudeMCPClient()
    
    # Simple test: claim extraction
    prompt = f"""Extract the key scientific claims from this abstract. Return each claim as a concise, standalone numbered statement. Be specific and include quantitative results.

Abstract:
{paper_wind['abstract']}"""
    
    result = client.run_prompt(prompt)
    print(f"\n=== RESULT ===\n{result[:2000]}")
    
    client.close()

if __name__ == "__main__":
    main()
