# gen_utils.py — supports Ollama first, with a safe Transformers fallback
import os
from typing import List

# ---- Config via env vars ----
# To use Ollama (recommended for your setup):
#   export USE_OLLAMA=1
#   export GEN_MODEL_NAME="llama3:latest"   # or "gpt-oss:20b"
USE_OLLAMA = os.getenv("USE_OLLAMA", "1") == "1"
GEN_MODEL = os.getenv("GEN_MODEL_NAME", "llama3:latest")

# Fallback HF model if Ollama isn't installed/available
FALLBACK_HF_MODEL = os.getenv("FALLBACK_HF_MODEL", "google/flan-t5-small")


# ---------- Prompt builders ----------
def build_prompt(question: str, contexts: List[str]) -> str:
    ctx = "\n\n---\n".join(contexts)
    return (
        "Answer ONLY from the context below. If not present, say \"I don't know\".\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )



# ---------- Backends ----------
def _load_ollama_callable():
    try:
        import ollama  # lazy import so app still runs without it
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Ollama Python client is not installed. Run:\n"
            "  pip install ollama\n"
            "Or set USE_OLLAMA=0 to use the Transformers fallback."
        ) from e

    model = GEN_MODEL or "llama3:latest"

    def _call(prompt: str, max_new_tokens: int = 160) -> str:
        r = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "Answer using ONLY the provided context."},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": 0.0,
                "num_predict": max_new_tokens,
                "repeat_penalty": 1.12
            }
        )
        return r["message"]["content"].strip()

    return _call


def _load_transformers_pipeline():
    # Minimal, CPU-safe fallback using HF transformers (FLAN-T5-small)
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

    tok = AutoTokenizer.from_pretrained(FALLBACK_HF_MODEL)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(
        FALLBACK_HF_MODEL,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False
    )
    return pipeline("text2text-generation", model=mdl, tokenizer=tok)


def load_generator():
    """
    Returns a callable 'generator(prompt: str, max_new_tokens: int) -> str'
    Prefers Ollama if USE_OLLAMA=1; otherwise falls back to HF transformers.
    """
    if USE_OLLAMA:
        try:
            return _load_ollama_callable()
        except RuntimeError:
            # fall back silently to transformers if ollama is missing
            pass
    # HF fallback
    hf_pipe = _load_transformers_pipeline()
    def _call(prompt: str, max_new_tokens: int = 160) -> str:
        out = hf_pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            no_repeat_ngram_size=4,
            repetition_penalty=1.12
        )[0]["generated_text"]
        return out.strip()
    return _call


def generate_answer(gen, prompt: str, max_new_tokens: int = 160) -> str:
    """Unified wrapper — 'gen' is a callable in both Ollama and HF fallback modes."""
    return gen(prompt, max_new_tokens)
