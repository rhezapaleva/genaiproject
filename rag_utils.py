# rag_utils.py — PDF reading (Docling/pdfplumber/PyPDF), cleaning, chunking, indexing, retrieval
from typing import List, Tuple
import re
import faiss
from sentence_transformers import SentenceTransformer

# ---------- PDF readers ----------
def read_resume_best(file) -> str:
    """
    Unified loader: Docling -> pdfplumber -> PyPDF.
    Returns extracted text as a single string.
    """
    # 1) Docling (best structure)
    try:
        from docling import Document
        doc = Document(file)
        return doc.get_text()
    except Exception:
        pass

    # 2) pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(file) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)
    except Exception:
        pass

    # 3) PyPDF (fallback)
    try:
        from pypdf import PdfReader
        reader = PdfReader(file)
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")

# ---------- cleaning ----------
CONTACT_PAT = re.compile(r'(@|https?://|\+?\d[\d\s\-]{7,}|\bLinkedIn\b|\bGitHub\b)', re.I)

def clean_text_for_resume(txt: str) -> str:
    """
    Removes contact lines & page artifacts; fixes common PDF glue (e.g., 'Scaledthe' -> 'Scaled the').
    """
    txt = txt.replace("•", "- ").replace("–", "-").replace("—", "-")
    lines = []
    for line in txt.splitlines():
        l = line.strip()
        if not l:
            continue
        if CONTACT_PAT.search(l):
            continue
        if l.lower().startswith(("document", "page ", "table ")):
            continue
        lines.append(l)
    txt = " ".join(lines)
    # Insert spaces between camelCase-ish joins
    txt = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', txt)
    # Tidy punctuation/whitespace
    txt = re.sub(r'/+', ' / ', txt)
    txt = re.sub(r'\s+,', ',', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

# ---------- chunking ----------
def chunk_text(text: str, max_chars=900, overlap=150) -> List[str]:
    """
    Section-aware chunking with heading prefix to aid retrieval.
    """
    text = text.replace("•", "- ")
    text = re.sub(r"[ \t]+", " ", text)

    parts = re.split(
        r"(?im)^\s*(experience|work experience|education|projects|skills|awards|certifications)\s*[:\n]",
        text
    )
    labeled = []
    if len(parts) > 1:
        pre = parts[0]
        if pre.strip():
            labeled.append(("Profile", pre))
        for i in range(1, len(parts), 2):
            heading = parts[i].strip().title()
            body = parts[i+1]
            labeled.append((heading, body))
    else:
        labeled = [("Document", text)]

    chunks: List[str] = []
    for heading, body in labeled:
        start = 0
        while start < len(body):
            end = min(start + max_chars, len(body))
            piece = body[start:end].strip()
            if piece:
                # Prefix heading for semantic context
                chunks.append(f"[{heading}] {piece}")
            if end == len(body):
                break
            start = max(0, end - overlap)
    return chunks

# ---------- indexing & retrieval ----------
def build_index(chunks: List[str], embedder: SentenceTransformer):
    """
    Build a FAISS IP index over L2-normalized embeddings (IP ≈ cosine).
    """
    embs = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx, embs

def retrieve(query: str, embedder, index, chunks: List[str], k=4) -> List[Tuple[float, str]]:
    """
    Return top-k (score, chunk) pairs using cosine-like IP on normalized vectors.
    """
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q, k)
    return [(float(s), chunks[i]) for s, i in zip(D[0], I[0]) if i != -1]