#!/usr/bin/env python3
"""Test Claude Code via MCP protocol - direct stdio connection."""
import subprocess
import json
import sys

# Start claude mcp serve
proc = subprocess.Popen(
    ["claude", "mcp", "serve"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

def send(msg):
    """Send JSON-RPC message and read response."""
    line = json.dumps(msg) + "\n"
    proc.stdin.write(line)
    proc.stdin.flush()
    # Read response (newline-delimited JSON)
    resp_line = proc.stdout.readline()
    return json.loads(resp_line.strip())

# Step 1: Initialize
init_resp = send({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "hermes-benchmark", "version": "1.0.0"}
    }
})
print(f"Init: {json.dumps(init_resp, indent=2)[:300]}")

# Step 2: Notify initialized
send({"jsonrpc": "2.0", "method": "notifications/initialized"})

# Step 3: List tools
tools_resp = send({
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list"
})
print(f"\nTools: {json.dumps(tools_resp, indent=2)[:500]}")

if "result" in tools_resp and "tools" in tools_resp["result"]:
    for t in tools_resp["result"]["tools"]:
        print(f"\n  Tool: {t['name']}")
        print(f"  Desc: {t.get('description', '')[:200]}")
        print(f"  Schema: {json.dumps(t.get('inputSchema', {}), indent=2)[:200]}")
elif "error" in tools_resp:
    print(f"\nError listing tools: {tools_resp['error']}")

proc.terminate()
proc.wait(timeout=5)
