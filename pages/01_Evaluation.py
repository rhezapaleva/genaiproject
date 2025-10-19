# pages/01_Evaluation.py
import json, time, numpy as np, streamlit as st
from sentence_transformers import SentenceTransformer
from rag_utils import retrieve
from gen_utils import build_prompt, generate_answer, load_generator

def build_eval_prompt(q, ctx):
    ctx = "\n\n---\n".join(ctx)
    return (
        "Answer ONLY from the context. Copy the minimal phrase that answers the question. "
        'If not present, reply exactly: "I don’t know." \n\n'
        f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"
    )

st.set_page_config(page_title="Evaluation", page_icon="📊", layout="wide")
st.title("📊 Evaluation")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("intfloat/e5-base-v2")

embedder = load_embedder()
generator = load_generator()

chunks = st.session_state.get("chunks", [])
idx = st.session_state.get("idx", None)

# Show which embedder was used to build the index (if stored by app.py)
embedder_id = st.session_state.get("embedder_id", "sentence-transformers/all-mpnet-base-v2")
st.caption(f"Embedder in use: {embedder_id}")

# Dimension sanity check to avoid FAISS assertion errors
try:
    qdim = load_embedder().encode(["test"], convert_to_numpy=True, normalize_embeddings=True).shape[1]
    idx_dim = getattr(idx, "d", None) if idx is not None else None
    if idx is not None and idx_dim is not None and qdim != idx_dim:
        st.error(f"Embedding size mismatch — query dim {qdim} vs index dim {idx_dim}. "
                 "Rebuild the index on the main page or switch this page to the same embedder.")
        st.stop()
except Exception:
    pass

st.write("Upload a small test set of Q/A pairs.")
file = st.file_uploader("eval_set.json", type=["json"])
k = st.slider("Recall@K", 1, 8, 4)
sim_tau = st.slider("Semantic similarity threshold", 0.50, 0.95, 0.75, 0.01)
chunk_sim_tau = st.slider("Chunk match threshold (semantic, for retrieval OK)", 0.30, 0.90, 0.45, 0.01)

if file and idx and chunks:
    eval_set = json.load(file)
    n = len(eval_set)
    correct_retrieval = 0
    correct_answers = 0
    latencies = []

    for ex in eval_set:
        q, gold = ex["q"], ex["a"]
        t0 = time.time()
        ctx = retrieve(q, embedder, idx, chunks, k=k)
        ctx_texts = [c for _, c in ctx]

        # Retrieval metric: Substring OR semantic match against any top-k chunk
        retrieved_ok = any(gold.lower() in c.lower() for c in ctx_texts)
        best_chunk_sim = None
        if not retrieved_ok and ctx_texts:
            g_emb = embedder.encode([gold], convert_to_numpy=True, normalize_embeddings=True)[0]
            c_embs = embedder.encode(ctx_texts, convert_to_numpy=True, normalize_embeddings=True)
            sims = (c_embs @ g_emb).tolist()
            best_chunk_sim = max(sims)
            retrieved_ok = best_chunk_sim >= chunk_sim_tau
        correct_retrieval += int(retrieved_ok)

        # Generate only if something relevant was retrieved
        pred = generate_answer(generator, build_eval_prompt(q, ctx_texts)) if retrieved_ok else "I don't know."
        latencies.append((time.time()-t0)*1000)

        # Semantic correctness (cosine sim)
        emb = embedder.encode([pred, gold], convert_to_numpy=True, normalize_embeddings=True)
        sim = float(np.dot(emb[0], emb[1]))
        correct_answers += int(sim >= sim_tau)

        with st.expander(q):
            st.write("**Gold:**", gold)
            st.write("**Pred:**", pred)
            st.write(f"**Semantic sim:** {sim:.3f}")
            st.write("**Retrieved relevant?**", "✅" if retrieved_ok else "❌")
            if best_chunk_sim is not None:
                st.write(f"**Best chunk semantic sim to gold:** {best_chunk_sim:.3f}")

    st.subheader("Summary")
    st.metric("Recall@K (retrieval)", f"{correct_retrieval}/{n}", f"{100*correct_retrieval/n:.1f}%")
    st.metric("Answer accuracy (semantic)", f"{correct_answers}/{n}", f"{100*correct_answers/n:.1f}%")
    st.metric("Avg latency (ms)", f"{np.mean(latencies):.0f}")
elif not (idx and chunks):
    st.info("Go to the main page, upload your resume, and Build the index first.")