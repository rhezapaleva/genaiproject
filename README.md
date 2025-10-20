# Resume-QA Chatbot (RAG)

An interactive Retrieval-Augmented Generation (RAG) system built for my **GenAI Individual Assignment**.  
This Streamlit app allows users to upload a resume (PDF/TXT), automatically chunk and embed it,  
then query the content through a local **LLM (Ollama / Llama 3)** or a fallback **Flan-T5 model**.  
It also includes an **evaluation page** to measure retrieval and generation quality.

---

## Features

- **PDF ingestion with Docling/PyPDF** → text extraction and cleaning  
- **Semantic chunking** → resume split into overlapping sections  
- **Embedding retrieval** using `sentence-transformers/all-mpnet-base-v2`  
- **Local generation** via **Ollama (Llama3 / GPT-OSS)** or fallback `flan-t5-small`  
- **Summary and insight modes** (for generating resume profiles or career trajectories)  
- **Evaluation dashboard** (Recall@K, semantic similarity accuracy, latency)  

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