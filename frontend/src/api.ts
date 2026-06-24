const API = (
  window.__RUNTIME_CONFIG__?.API_URL ||
  import.meta.env.VITE_API_URL ||
  "/api"
).replace(/\/$/, "");

export type DocumentInfo = { id: string; filename: string; chunk_count: number };

export type AskResponse = {
  answer: string;
  confidence: number;
  sources: string[];
  metrics: {
    grounding_score: number;
    retrieval_quality_score: number;
    critic_passed: boolean;
    retry_count: number;
  };
  total_latency_ms?: number;
};

function url(path: string) {
  return `${API}${path.startsWith("/") ? path : `/${path}`}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url(path), init);
  } catch {
    throw new Error("Cannot reach API. Is the backend running?");
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const detail = err.detail;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail) || res.statusText);
  }
  return res.json();
}

export const fetchDocuments = () => request<DocumentInfo[]>("/documents");

export async function uploadDocument(file: File) {
  const form = new FormData();
  form.append("file", file);
  await request("/upload", { method: "POST", body: form });
}

export const askQuestion = (question: string) =>
  request<AskResponse>("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: 8 }),
  });
