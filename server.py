"""AI Document Chat (RAG) - backend.

Self-contained HTTP server (stdlib only, plus optional pypdf/python-docx).
Provides:
  GET  /            -> index.html
  GET  /static/*    -> static assets
  POST /api/upload   -> ingest a document, return chunk count
  POST /api/chat     -> answer a question against the ingested document
  POST /api/reset    -> clear the current document
"""
import os
import re
import json
import math
import uuid
import mimetypes
import http.server
import socketserver
from http import HTTPStatus
from urllib.parse import urlparse, unquote
from collections import Counter, defaultdict

# ---------- optional document parsers ----------
try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    try:
        from PyPDF2 import PdfReader
    except Exception:
        PdfReader = None

try:
    import docx as _docx
except Exception:
    _docx = None

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
CHUNK_SIZE = 900          # characters per chunk
CHUNK_OVERLAP = 150
TOP_K = 5

# In-memory session store: session_id -> {"name":..., "chunks":[str], "index":...}
SESSIONS = {}


# ---------- text extraction ----------
def extract_text(filename, data):
    lower = filename.lower()
    if lower.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError("pypdf is not installed; cannot read PDF")
        import io
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
        return "\n".join(parts)
    if lower.endswith(".docx"):
        if _docx is None:
            raise RuntimeError("python-docx is not installed; cannot read DOCX")
        import io
        doc = _docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    # txt / md / anything text-like
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def chunk_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + CHUNK_SIZE, n)
        # try to break on sentence boundary
        if end < n:
            window = text[i:end]
            last = max(window.rfind(". "), window.rfind("! "), window.rfind("? "), window.rfind("\n"))
            if last > CHUNK_SIZE // 2:
                end = i + last + 1
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        i = max(end - CHUNK_OVERLAP, i + 1)
    return chunks


# ---------- retrieval (TF-IDF cosine) ----------
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(s):
    return [t.lower() for t in TOKEN_RE.findall(s)]


def build_index(chunks):
    # document frequency
    df = Counter()
    tokenized = []
    for c in chunks:
        toks = set(tokenize(c))
        tokenized.append(toks)
        for t in toks:
            df[t] += 1
    n = len(chunks)
    idf = {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df.items()}
    # tf-idf vectors as dicts
    vecs = []
    for toks in tokenized:
        tf = Counter(toks)
        vec = {t: (tf[t] / len(toks) if toks else 0) * idf.get(t, 1.0) for t in toks}
        vecs.append(vec)
    return {"idf": idf, "vecs": vecs, "tokenized": tokenized}


def cosine(a, b):
    if not a or not b:
        return 0.0
    dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve(session, query, k=TOP_K):
    idx = session["index"]
    idf = idx["idf"]
    qtoks = tokenize(query)
    if not qtoks:
        return []
    qtf = Counter(qtoks)
    qvec = {t: (qtf[t] / len(qtoks)) * idf.get(t, 1.0) for t in set(qtoks)}
    scored = []
    for i, v in enumerate(idx["vecs"]):
        scored.append((cosine(qvec, v), i))
    scored.sort(reverse=True)
    out = []
    for score, i in scored[:k]:
        if score <= 0:
            continue
        out.append((score, session["chunks"][i]))
    # fallback: if nothing matched, return first chunks
    if not out and session["chunks"]:
        out = [(0.0, c) for c in session["chunks"][:k]]
    return out


# ---------- answer generation ----------
def openai_available():
    return bool(os.environ.get("OPENAI_API_KEY"))


def generate_answer(question, passages):
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            import urllib.request
            context = "\n\n---\n\n".join(p for _, p in passages)
            payload = {
                "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [
                    {"role": "system", "content": (
                        "You are a helpful assistant. Answer the user's question using ONLY the "
                        "provided document passages. If the answer is not in the passages, say you "
                        "don't know. Cite the relevant passage briefly."
                    )},
                    {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
                ],
                "temperature": 0.2,
            }
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip(), "openai"
        except Exception as e:
            # fall through to extractive
            err = str(e)
    else:
        err = None

    # Extractive fallback: stitch top passages, highlight sentences containing query terms.
    qterms = set(tokenize(question))
    best_sentences = []
    for score, passage in passages:
        for sent in re.split(r"(?<=[.!?])\s+", passage):
            st = set(tokenize(sent))
            overlap = len(st & qterms)
            if overlap > 0:
                best_sentences.append((overlap, sent.strip()))
    best_sentences.sort(reverse=True)
    if best_sentences:
        chosen = [s for _, s in best_sentences[:4]]
        answer = " ".join(chosen)
    else:
        answer = passages[0][1] if passages else "No relevant content found."
    if err:
        answer = f"(LLM unavailable: {err})\n\n" + answer
    return answer, "extractive"


# ---------- HTTP handler ----------
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=os.path.dirname(os.path.abspath(__file__)), **kw)

    def _json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload":
            self.handle_upload()
        elif parsed.path == "/api/chat":
            self.handle_chat()
        elif parsed.path == "/api/reset":
            self.handle_reset()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def handle_upload(self):
        ctype = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_UPLOAD_BYTES:
            self._json(413, {"error": "file too large (max 10MB)"})
            return
        raw = self.rfile.read(length)
        # parse multipart
        if "multipart/form-data" in ctype:
            boundary = None
            for part in ctype.split(";"):
                part = part.strip()
                if part.startswith("boundary="):
                    boundary = part[len("boundary="):].strip().strip('"')
            if not boundary:
                self._json(400, {"error": "missing boundary"})
                return
            files = parse_multipart(raw, boundary)
            if not files:
                self._json(400, {"error": "no file uploaded"})
                return
            f = files[0]
            filename = f["filename"]
            data = f["data"]
        else:
            # raw body: filename in X-Filename header
            filename = self.headers.get("X-Filename", "document.txt")
            data = raw

        try:
            text = extract_text(filename, data)
        except Exception as e:
            self._json(400, {"error": str(e)})
            return
        chunks = chunk_text(text)
        if not chunks:
            self._json(400, {"error": "no text could be extracted from the document"})
            return
        sid = str(uuid.uuid4())
        SESSIONS[sid] = {
            "name": filename,
            "chunks": chunks,
            "index": build_index(chunks),
            "text_len": len(text),
        }
        self._json(200, {
            "session_id": sid,
            "filename": filename,
            "chunks": len(chunks),
            "characters": len(text),
        })

    def handle_chat(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._json(400, {"error": "invalid json"})
            return
        sid = body.get("session_id")
        question = (body.get("question") or "").strip()
        if not sid or sid not in SESSIONS:
            self._json(400, {"error": "no document loaded; upload one first"})
            return
        if not question:
            self._json(400, {"error": "question is required"})
            return
        session = SESSIONS[sid]
        passages = retrieve(session, question)
        answer, mode = generate_answer(question, passages)
        self._json(200, {
            "answer": answer,
            "mode": mode,
            "sources": [{"score": round(s, 3), "text": p[:400]} for s, p in passages],
        })

    def handle_reset(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            body = {}
        sid = body.get("session_id")
        if sid in SESSIONS:
            del SESSIONS[sid]
        self._json(200, {"ok": True})


def parse_multipart(raw, boundary):
    """Minimal multipart parser returning list of {filename, data}."""
    b = boundary.encode()
    delim = b"--" + b
    parts = raw.split(delim)
    files = []
    for part in parts:
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        # split headers/body
        if b"\r\n\r\n" in part:
            header_block, body = part.split(b"\r\n\r\n", 1)
        elif b"\n\n" in part:
            header_block, body = part.split(b"\n\n", 1)
        else:
            continue
        # strip trailing boundary marker
        if body.endswith(b"--"):
            body = body[:-2]
        body = body.rstrip(b"\r\n")
        headers = header_block.decode("utf-8", errors="replace")
        if "filename=" not in headers:
            continue
        # filename
        m = re.search(r'filename="([^"]*)"', headers)
        filename = m.group(1) if m else "upload.bin"
        # content-type ignored
        files.append({"filename": filename, "data": body})
    return files


def main():
    port = int(os.environ.get("PORT", "8000"))
    httpd = socketserver.ThreadingTCPServer(("0.0.0.0", port), Handler)
    httpd.allow_reuse_address = True
    print(f"AI RAG server running on http://0.0.0.0:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
