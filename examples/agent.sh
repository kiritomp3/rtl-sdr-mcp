#!/usr/bin/env bash
# Run an autonomous radio investigation with Claude Code in headless mode.
#   ./examples/agent.sh "Find a station where people are talking right now and summarize it"
# Requires: `claude` CLI logged in, and `sdr-mcp` on PATH (pip install / uv tool install).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TASK="${1:-Explore the radio spectrum around me and report the most interesting thing you find.}"
CFG="$(mktemp)"; trap 'rm -f "$CFG"' EXIT
SDR="$(command -v sdr-mcp)"
cat > "$CFG" <<JSON
{"mcpServers":{"sdr":{"command":"$SDR","env":{"SDR_LAT":"${SDR_LAT:-}","SDR_LON":"${SDR_LON:-}"}}}}
JSON
mkdir -p logs
echo "$TASK" | claude -p ${MODEL:+--model "$MODEL"} \
  --mcp-config "$CFG" --strict-mcp-config --allowedTools "mcp__sdr__*" \
  --output-format stream-json --verbose \
  --append-system-prompt "You are a radio investigator with a real SDR. Form hypotheses and test them with the tools. Finish with a report: what you found, on which frequencies, how confident you are and why." \
  | tee "logs/agent_$(date +%Y%m%d_%H%M%S).jsonl" | python3 -u "$HERE/agent_view.py"
