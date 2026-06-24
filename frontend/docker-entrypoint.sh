#!/bin/sh
set -e

# Same-origin proxy via nginx (/api -> backend). Render sets full API URL via env.
API_URL="${API_URL:-/api}"

cat > /usr/share/nginx/html/config.js <<EOF
window.__RUNTIME_CONFIG__ = { API_URL: "${API_URL}" };
EOF

exec nginx -g 'daemon off;'
