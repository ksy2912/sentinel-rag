#!/bin/sh
set -e

API_URL="${API_URL:-http://localhost:8001}"

cat > /usr/share/nginx/html/config.js <<EOF
window.__RUNTIME_CONFIG__ = { API_URL: "${API_URL}" };
EOF

exec nginx -g 'daemon off;'
