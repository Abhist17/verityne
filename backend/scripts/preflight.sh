#!/usr/bin/env bash
# Demo pre-flight. Prints one PASS/FAIL line per check and exits non-zero if
# anything is red, so it can gate a recording session or a CI smoke test.
#
# Deliberately a script and not three lines in a README: the README version
# interleaved commands with their expected output, and pasting the whole block
# into a shell ran the output lines as commands.
ok=1
health=$(curl -s -m 5 localhost:8000/health)

if printf '%s' "$health" | grep -q '"scoring_available":true'; then
  echo "  [PASS] backend scoring_available: true"
else
  echo "  [FAIL] backend scoring_available - is 'make backend' running?"; ok=0
fi

device=$(printf '%s' "$health" | python3 -c 'import json,sys;print(json.load(sys.stdin)["device"])' 2>/dev/null)
if [ "$device" = "cuda" ]; then
  echo "  [PASS] device: cuda"
else
  echo "  [FAIL] device: ${device:-unreachable} (expected cuda)"; ok=0
fi

code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' localhost:3000)
if [ "$code" = "200" ]; then
  echo "  [PASS] dashboard localhost:3000: 200"
else
  echo "  [FAIL] dashboard localhost:3000: $code - is 'make frontend' running?"; ok=0
fi

if [ "$ok" = 1 ]; then
  echo "  ALL GREEN - clear to record"
else
  echo "  NOT READY - fix the FAIL above before recording"
  exit 1
fi
