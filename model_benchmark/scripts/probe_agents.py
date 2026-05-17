#!/usr/bin/env python3
"""Probe Claude MCP to find available Agent types."""
import json
import subprocess

proc = subprocess.Popen(
    ["claude", "mcp", "serve"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, bufsize=1
)

def send(method, params=None):
    msg = {"jsonrpc": "2.0", "method": method}
    if params: msg["params"] = params
    # requests without id (notifications)
    if method != "notifications/initialized":
        msg["id"] = 1
    line = json.dumps(msg) + "\n"
    proc.stdin.write(line); proc.stdin.flush()
    return json.loads(proc.stdout.readline().strip())

# Init
print("INIT:", json.dumps(send("initialize", {
    "protocolVersion": "2025-03-26",
    "capabilities": {},
    "clientInfo": {"name": "probe", "version": "1.0"}
}), indent=2)[:200])

send("notifications/initialized")

# Try Agent tool with no type to see full error
result = send("tools/call", {
    "name": "Agent",
    "arguments": {"subagent_type": "unknown", "prompt": "test"}
})
print("\nAGENT ERROR:", json.dumps(result, indent=2))

# Try empty type
result = send("tools/call", {
    "name": "Agent",
    "arguments": {"prompt": "test"}
})
print("\nAGENT NO TYPE:", json.dumps(result, indent=2)[:500])

# Try common agent types
for agent_type in ["code", "default", "research", "planner", "coder", "developer", "assistant", "general"]:
    result = send("tools/call", {
        "name": "Agent",
        "arguments": {"subagent_type": agent_type, "prompt": "say hello"}
    })
    text = result.get("result", {}).get("content", [{}])[0].get("text", "")
    if "not found" not in text.lower():
        print(f"\n✅ FOUND agent type '{agent_type}': {text[:100]}")
    err_text = result.get("error", "")
    if err_text and "Available agents:" in str(err_text):
        available = str(err_text).split("Available agents:")[1]
        print(f"\nAvailable agents (from error): {available[:500]}")

proc.terminate()
proc.wait()
