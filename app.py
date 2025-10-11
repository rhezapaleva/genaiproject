# app.py
from dotenv import load_dotenv; load_dotenv()
import time, streamlit as st
from sentence_transformers import SentenceTransformer
from rag_utils import read_resume_best, clean_text_for_resume, chunk_text, build_index, retrieve
from gen_utils import load_generator, build_prompt, generate_answer
import re
def dedupe_chunks(chunks, max_keep=8):
    seen, keep = set(), []
    for c in chunks:
        fp = re.sub(r"[^a-z0-9 ]+", " ", c.lower())
        fp = re.sub(r"\s+", " ", fp).strip()[:120]  # fingerprint
        if fp in seen: 
            continue
        seen.add(fp)
        keep.append(c)
        if len(keep) >= max_keep: 
            break
    return keep

def is_summary_query(q: str, embedder, threshold: float = 0.35) -> bool:
    """
    Returns True if the query likely asks for a profile/summary/overview.
    Uses semantic similarity against a small set of summary intent prototypes.
    threshold in [0,1]; 0.30-0.40 is a good range for all-mpnet-base-v2.
    """
    # Fast lexical early-exit (cheap)
    ql = q.lower()
    if ("summ" in ql) or any(t in ql for t in ["overview", "profile", "highlights", "recap", "bio", "about you", "about me"]):
        return True

    # Semantic prototypes (paraphrases)
    prototypes = [
        "summarize my experience",
        "write a profile summary",
        "give me an overview of the resume",
        "tell me about this person",
        "what's their background",
        "elevator pitch",
        "short bio based on the resume"
    ]

    q_emb = embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    p_embs = embedder.encode(prototypes, convert_to_numpy=True, normalize_embeddings=True)

    # cosine similarities (dot product for normalized vectors)
    sims = (q_emb @ p_embs.T).ravel()
    return sims.max() >= threshold

def build_summary_prompt(contexts: list[str]) -> str:
    ctx = "\n\n---\n".join(contexts)
    return (
        "You are a resume summarizer. Using ONLY the context, write ONE cohesive paragraph "
        "(90–120 words) that covers roles, organizations, skills, and quantified impact. "
        "Rules: do not repeat phrases, do not include contact details, avoid bullet formatting, "
        "and vary sentence openings.\n\n"
        f"Context:\n{ctx}\n\nSummary:"
    )
st.set_page_config(page_title="Resume-QA", page_icon="💬", layout="wide")
st.title("💬 Resume-QA Chatbot (RAG)")

@st.cache_resource
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("intfloat/e5-base-v2")

embedder = load_embedder()
generator = load_generator()

with st.sidebar:
    st.header("Knowledge Base")
    uploaded = st.file_uploader("Upload resume (PDF or TXT)", type=["pdf", "txt"])
    chunk_size = st.slider("Chunk size", 500, 1200, 900, 50)
    overlap = st.slider("Overlap", 0, 300, 150, 10)
    top_k = st.slider("Top-K retrieval", 1, 12, 8)         # was 1..8 default 4
    thr = st.slider("Min retrieval score (guardrail)", 0.0, 1.0, 0.18, 0.01)   # was 0.35
    build_btn = st.button("Build / Rebuild Index")

if "chunks" not in st.session_state: st.session_state.chunks = []
if "idx" not in st.session_state: st.session_state.idx = None
if "hist" not in st.session_state: st.session_state.hist = []

# Build index when user clicks the button
if build_btn and uploaded:
    if uploaded.name.endswith(".pdf"):
        text = read_resume_best(uploaded)
    else:
        text = uploaded.read().decode("utf-8", errors="ignore")
    # Clean parsed text to remove contact lines, page artifacts, and glued words
    text = clean_text_for_resume(text)
    st.session_state.chunks = chunk_text(text, max_chars=chunk_size, overlap=overlap)
    st.session_state.idx, _ = build_index(st.session_state.chunks, embedder)
    st.success(f"Indexed {len(st.session_state.chunks)} chunks ✅")

# Show history
for role, msg, ctx in st.session_state.hist:
    with st.chat_message(role):
        st.markdown(msg)
        if ctx:
            with st.expander("Sources"):
                for i, (score, snippet) in enumerate(ctx, 1):
                    st.markdown(f"**{i}.** _score {score:.3f}_ — {snippet[:400]}...")

# Chat input
q = st.chat_input("Ask something about your resume…")
if q:
    if st.session_state.idx is None:
        st.warning("Upload your resume and click **Build / Rebuild Index** first.")
    else:
        with st.chat_message("user"): st.markdown(q)

        t0 = time.time()
        ctx = retrieve(q, embedder, st.session_state.idx, st.session_state.chunks, k=top_k)
        max_score = max([s for s,_ in ctx], default=0.0)

        if is_summary_query(q, embedder):
            #Cast a wider net for summaries
            if top_k <8:
                ctx = retrieve(q, embedder, st.session_state.idx, st.session_state.chunks, k=8)
            # NEW: dedupe before prompting
            ctx_texts = dedupe_chunks([c for _, c in ctx], max_keep=8)
            ans = generate_answer(generator, build_summary_prompt(ctx_texts), max_new_tokens=220)
        else:
            if max_score < thr:
                ans = "I don't know. Try rephrasing or ask about another section."
            else:
                ans = generate_answer(generator, build_prompt(q, [c for _, c in ctx]))

        latency_ms = int((time.time() - t0) * 1000)

        with st.chat_message("assistant"):
            st.markdown(ans + f"\n\n_<latency: {latency_ms} ms>_")
            with st.expander("Sources"):
                for i, (score, snippet) in enumerate(ctx, 1):
                    st.markdown(f"**{i}.** _score {score:.3f}_ — {snippet[:400]}...")

        st.session_state.hist += [("user", q, None), ("assistant", ans, ctx)]