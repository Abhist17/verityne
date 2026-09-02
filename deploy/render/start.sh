#!/bin/sh
# Both halves, one container. The API on a fixed internal port; the dashboard on
# whatever Render hands us. Only the dashboard is reachable from outside.
set -e

uvicorn verityne.main:app --host 127.0.0.1 --port 8000 --app-dir /app/backend &
API=$!

# Do not serve a dashboard whose API is not answering yet: the first request
# would render an error box and a reviewer would call the deploy broken.
i=0
until curl -sf -o /dev/null http://127.0.0.1:8000/health; do
  i=$((i + 1))
  [ "$i" -gt 60 ] && { echo "api did not come up in 60s"; exit 1; }
  kill -0 "$API" 2>/dev/null || { echo "api died on startup"; wait "$API"; exit 1; }
  sleep 1
done
echo "api ready, starting dashboard on ${PORT}"

cd /app/web
exec node_modules/.bin/next start -p "${PORT}" -H 0.0.0.0
