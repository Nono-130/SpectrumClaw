#!/usr/bin/env bash
set -euo pipefail

wait_for_url() {
  local url="$1"
  local attempts="${2:-40}"
  for _ in $(seq 1 "$attempts"); do
    if curl -fsS -m 4 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "[network] timeout waiting for $url" >&2
  return 1
}

echo "[network] restarting server backend link ..."
systemctl --user restart spectrumclaw-links.service

wait_for_url http://127.0.0.1:8230/health

# The LAN console is served by the persistent local frontend service.  Keep
# this branch explicit: previous recovery output incorrectly advertised the
# loopback-only remote backend as the browser entry point, which encouraged
# clients to bypass the `/backend` proxy used by camera and noVNC resources.
LOCAL_FRONTEND_ACTIVE=0
if systemctl --user is-active --quiet spectrumclaw-frontend.service; then
  LOCAL_FRONTEND_ACTIVE=1
  wait_for_url http://127.0.0.1:5173/
fi

LOCAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '
  /src/ { for (i = 1; i <= NF; i++) if ($i == "src") { print $(i + 1); exit } }
')"
if [ -z "$LOCAL_IP" ]; then
  LOCAL_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi

if [ "$LOCAL_FRONTEND_ACTIVE" = "1" ]; then
  echo "[network] LAN frontend and backend proxy are ready"
  echo "[network] local loopback URL: http://127.0.0.1:5173/"
  if [ -n "$LOCAL_IP" ]; then
    echo "[network] browser URL:      http://$LOCAL_IP:5173/"
  fi
  echo "[network] media and noVNC must use /backend; do not open port 8230 directly"
else
  echo "[network] backend link is ready; local frontend service is disabled"
  echo "[network] local loopback URL: http://127.0.0.1:8230/"
fi
