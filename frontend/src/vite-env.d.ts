/// <reference types="vite/client" />

interface RuntimeConfig {
  API_URL?: string;
}

interface Window {
  __RUNTIME_CONFIG__?: RuntimeConfig;
}

interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
