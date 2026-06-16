import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import {
  AskResponse,
  DocumentInfo,
  askQuestion,
  fetchDocuments,
  uploadDocument,
} from "./api";

type ChatMessage = {
  role: "user" | "assistant";
  text: string;
  response?: AskResponse;
};

export default function App() {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const loadDocs = useCallback(async () => {
    try {
      setDocs(await fetchDocuments());
    } catch {
      setError("Could not load documents. Is the API running?");
    }
  }, []);

  useEffect(() => {
    loadDocs();
  }, [loadDocs]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files;
    if (!files?.length) return;
    setUploading(true);
    setError(null);
    try {
      for (const file of Array.from(files)) {
        await uploadDocument(file);
      }
      await loadDocs();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const q = input.trim();
    if (!q || loading) return;

    setInput("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setLoading(true);
    setError(null);

    try {
      const response = await askQuestion(q);
      setMessages((m) => [...m, { role: "assistant", text: response.answer, response }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>Sentinel RAG</h1>
        <p className="subtitle">Upload docs · Ask questions · See citations</p>

        <label className="upload-btn">
          {uploading ? "Uploading…" : "Upload PDF / DOCX / TXT"}
          <input type="file" accept=".pdf,.docx,.txt" multiple hidden onChange={handleUpload} />
        </label>

        <h2>Documents ({docs.length})</h2>
        <ul className="doc-list">
          {docs.length === 0 && <li className="muted">No documents yet</li>}
          {docs.map((d) => (
            <li key={d.id}>
              <strong>{d.filename}</strong>
              <span>{d.chunk_count} chunks</span>
            </li>
          ))}
        </ul>
      </aside>

      <main className="chat">
        <div className="messages">
          {messages.length === 0 && (
            <div className="empty">
              <p>Ask a question about your uploaded documents.</p>
              <p className="muted">Example: What endpoints does FastAPI expose?</p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`bubble ${msg.role}`}>
              <div className="bubble-text">{msg.text}</div>
              {msg.response && (
                <div className="meta">
                  <div className="scores">
                    <span title="Confidence">Conf {(msg.response.confidence * 100).toFixed(0)}%</span>
                    <span title="Grounding">Ground {(msg.response.metrics.grounding_score * 100).toFixed(0)}%</span>
                    <span title="Retrieval quality">Retr {(msg.response.metrics.retrieval_quality_score * 100).toFixed(0)}%</span>
                    <span className={msg.response.metrics.critic_passed ? "pass" : "fail"}>
                      {msg.response.metrics.critic_passed ? "Critic ✓" : "Critic ✗"}
                    </span>
                    {msg.response.metrics.retry_count > 0 && (
                      <span>Retries {msg.response.metrics.retry_count}</span>
                    )}
                    {msg.response.total_latency_ms != null && (
                      <span>{msg.response.total_latency_ms}ms</span>
                    )}
                  </div>
                  {msg.response.sources.length > 0 && (
                    <div className="sources">
                      <strong>Sources:</strong>{" "}
                      {msg.response.sources.map((s, j) => (
                        <code key={j}>{s}</code>
                      ))}
                    </div>
                  )}
                  {msg.response.citations.length > 0 && (
                    <details>
                      <summary>Citations ({msg.response.citations.length})</summary>
                      <ul>
                        {msg.response.citations.map((c) => (
                          <li key={c.chunk_id}>
                            {c.filename}#chunk_{c.chunk_index} — sim {(c.similarity * 100).toFixed(0)}%
                          </li>
                        ))}
                      </ul>
                    </details>
                  )}
                </div>
              )}
            </div>
          ))}

          {loading && <div className="bubble assistant loading">Thinking…</div>}
          <div ref={bottomRef} />
        </div>

        {error && <div className="error">{error}</div>}

        <form className="input-row" onSubmit={handleSubmit}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your documents…"
            disabled={loading}
          />
          <button type="submit" disabled={loading || !input.trim()}>
            Send
          </button>
        </form>
      </main>
    </div>
  );
}
