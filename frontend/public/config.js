// Local dev default; Docker entrypoint overwrites this in production.
window.__RUNTIME_CONFIG__ = window.__RUNTIME_CONFIG__ || {
  API_URL: "http://localhost:8001",
};
