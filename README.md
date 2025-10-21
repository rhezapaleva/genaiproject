# Resume-QA Chatbot (RAG)

Upload a resume (PDF/TXT), the app parses and indexes it, then answers questions grounded in the resume using a local LLM (Ollama / Llama 3) with a safe HuggingFace fallback.

---

## Features

- PDF ingestion: Docling → pdfplumber → PyPDF (best-effort cascade)
- Cleaning: removes contact lines / page artifacts; fixes PDF “glued words”
- Chunking: section-aware, overlapping chunks (size/overlap tunable in UI)
- Embeddings: intfloat/e5-base-v2 with query/passage prefixes
- Vector index: FAISS (cosine via L2-normalized inner-product)
- Generator: Ollama (Llama 3 by default) or HF fallback (Flan-T5 small)
- Intent-aware prompting: one prompt adapts to yes/no, QA, and summary
- Evaluation: Recall@K, semantic accuracy, latency

---

## Installation

```bash
git clone https://github.com/rhezapaleva/genaiproject.git
cd genaiproject/individual

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

ollama pull llama3

export USE_OLLAMA=1
export GEN_MODEL_NAME="llama3:latest"

streamlit run app.py