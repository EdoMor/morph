#!/usr/bin/env bash
# Codespaces bootstrap: Python deps, Ollama, and a Gemma model.
#
# Sized for the default 4-core / 16 GB Codespace. gemma3:4b fits comfortably and
# is fast enough on CPU to run the self-improvement loop unattended. Set
# MORPH_MODEL=gemma3:12b on a 8-core / 32 GB machine for noticeably better edits.

set -euo pipefail

MODEL="${MORPH_MODEL:-gemma3:4b}"

echo "==> Installing Morph"
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

echo "==> Installing Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "==> Starting Ollama"
if ! pgrep -x ollama >/dev/null 2>&1; then
  nohup ollama serve > /tmp/ollama.log 2>&1 &
fi

for _ in $(seq 1 30); do
  if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "==> Pulling ${MODEL} (this takes a few minutes on first run)"
ollama pull "${MODEL}" || echo "WARNING: could not pull ${MODEL}; set MORPH_PROVIDER=echo to work offline"

echo "==> Verifying"
python -m pytest tests -q || echo "WARNING: test suite is not green"
python -m bench.runner --quiet --skip-requirements || true

cat <<'EOF'

Morph is ready.

  morph serve                     # web app on :8787 — open the forwarded URL on your phone
  morph chat "explain this repo"  # one-shot agent run
  python -m bench.runner          # score the current code
  python -m selfimprove.loop --iterations 3   # let Gemma improve Morph

EOF
