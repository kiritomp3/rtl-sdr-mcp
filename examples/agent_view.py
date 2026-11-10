"""Pretty-print a Claude Code stream-json run: thoughts, tool calls, final timing."""
import json
import sys

for line in sys.stdin:
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        continue
    if e.get("type") == "assistant":
        for c in e["message"]["content"]:
            if c["type"] == "text" and c["text"].strip():
                print("\n💬", c["text"].strip(), flush=True)
            elif c["type"] == "tool_use":
                name = c["name"].split("__")[-1]
                print(f"\n📡 {name}({json.dumps(c['input'], ensure_ascii=False)})", flush=True)
    elif e.get("type") == "result":
        print(f"\n— done in {e.get('duration_ms', 0) / 1000:.0f} s, {e.get('num_turns')} turns", flush=True)
