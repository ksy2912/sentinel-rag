import { FormEvent, useEffect, useState } from "react";
import { AskResponse, DocumentInfo, askQuestion, fetchDocuments, uploadDocument } from "./api";

type Message = { role: "user" | "assistant"; text: string; meta?: AskResponse };

export default function App() {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDocuments().then(setDocs).catch(() => setError("API not reachable"));
  }, []);

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    try {
      await uploadDocument(file);
      setDocs(await fetchDocuments());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      e.target.value = "";
    }
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setLoading(true);
    setError(null);
    try {
      const res = await askQuestion(q);
      setMessages((m) => [...m, { role: "assistant", text: res.answer, meta: res }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  const pct = (n: number) => `${(n * 100).toFixed(0)}%`;

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>Sentinel RAG</h1>
        <label className="upload-btn">
          Upload PDF / DOCX / TXT
          <input type="file" accept=".pdf,.docx,.txt" hidden onChange={onUpload} />
        </label>
        <ul className="doc-list">
          {docs.map((d) => (
            <li key={d.id}>{d.filename} <span>({d.chunk_count})</span></li>
          ))}
        </ul>
      </aside>

      <main className="chat">
        <div className="messages">
          {messages.map((m, i) => (
            <div key={i} className={`bubble ${m.role}`}>
              <p>{m.text}</p>
              {m.meta && (
                <div className="meta">
                  Conf {pct(m.meta.confidence)} · Ground {pct(m.meta.metrics.grounding_score)} ·{" "}
                  {m.meta.metrics.critic_passed ? "✓" : "✗"}
                  {m.meta.sources[0] && <div className="sources">{m.meta.sources.join(", ")}</div>}
                </div>
              )}
            </div>
          ))}
          {loading && <div className="bubble assistant">Thinking…</div>}
        </div>
        {error && <p className="error">{error}</p>}
        <form className="input-row" onSubmit={onSubmit}>
          <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask a question…" disabled={loading} />
          <button type="submit" disabled={loading || !input.trim()}>Send</button>
        </form>
      </main>
    </div>
  );
}
