#!/usr/bin/env bash
# Launch Hi3D Studio: start the local server (if not already running) and open it in an app window.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="$ROOT/config.json"
[ -f "$CFG" ] || { notify-send "Hi3D Studio" "Run install.sh first" 2>/dev/null; echo "Run install.sh first"; exit 1; }

read_cfg() { python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2],''))" "$CFG" "$1"; }
VENV="$(read_cfg venv)"; PORT="$(read_cfg port)"; PORT="${PORT:-7860}"
URL="http://127.0.0.1:$PORT"
mkdir -p "$ROOT/logs"

if ! curl -fs "$URL/api/ping" >/dev/null 2>&1; then
  cd "$ROOT"
  nohup "$VENV/bin/python" -m uvicorn app.server:app --host 127.0.0.1 --port "$PORT" \
    > "$ROOT/logs/server.log" 2>&1 &
  for _ in $(seq 1 60); do curl -fs "$URL/api/ping" >/dev/null 2>&1 && break; sleep 0.25; done
fi

for b in chromium chromium-browser google-chrome google-chrome-stable brave-browser microsoft-edge; do
  if command -v "$b" >/dev/null; then
    exec "$b" --app="$URL" --class=hi3d-studio --user-data-dir="$HOME/.local/share/hi3d-studio/browser" >/dev/null 2>&1
  fi
done
exec xdg-open "$URL"
