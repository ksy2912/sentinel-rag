const API_URL =
  window.__RUNTIME_CONFIG__?.API_URL ||
  import.meta.env.VITE_API_URL ||
  "http://localhost:8001";

export type DocumentInfo = {
  id: string;
  filename: string;
  mime_type: string | null;
  chunk_count: number;
  created_at: string;
};

export type Citation = {
  document_id: string;
  filename: string;
  chunk_index: number;
  chunk_id: string;
  similarity: number;
};

export type AskResponse = {
  answer: string;
  confidence: number;
  sources: string[];
  citations: Citation[];
  retrieved_chunks: {
    chunk_id: string;
    filename: string;
    chunk_index: number;
    chunk_text: string;
    similarity: number;
  }[];
  metrics: {
    grounding_score: number;
    retrieval_quality_score: number;
    confidence: number;
    critic_passed: boolean;
    retry_count: number;
    final_query: string;
  };
  run_id?: string;
  total_latency_ms?: number;
};

export async function fetchDocuments(): Promise<DocumentInfo[]> {
  const res = await fetch(`${API_URL}/documents`);
  if (!res.ok) throw new Error("Failed to load documents");
  return res.json();
}

export async function uploadDocument(file: File): Promise<void> {
  const form = new FormData();
  form.append("file", file);
  form.append("chunk_size", "1000");
  form.append("chunk_overlap", "150");
  const res = await fetch(`${API_URL}/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Upload failed");
  }
}

export async function askQuestion(question: string, topK = 8): Promise<AskResponse> {
  const res = await fetch(`${API_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Ask failed");
  }
  return res.json();
}
