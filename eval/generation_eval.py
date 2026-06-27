"""
Evaluates generation quality by measuring faithfulness and answer relevancy.
Uses an LLM-as-a-judge via the Groq API and local embeddings via SentenceTransformers.
Processes outputs from the evaluation pipeline to compute final metric scores.
"""

import json
import os
import re
from config.settings import settings
from pathlib import Path
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

def load_env_file():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v

load_env_file()

GROQ_BASE = "https://api.groq.com/openai/v1"

JUDGE_MODEL = "llama-3.3-70b-versatile"
EMB_MODEL = settings.embedding_model

INPUT_FILE = SCRIPT_DIR / "data" / "raw_outputs.json"
OUTPUT_FILE = SCRIPT_DIR / "data" / "generation_metrics.json"

MAX_CTX_DOCS = 6
N_GEN_QS = 3
SAMPLE_START = 0
SAMPLE_END = None

_request_counter = 0

def get_groq_client():
    global _request_counter
    keys_str = os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY")
    if not keys_str:
        raise ValueError("No GROQ_API_KEY or GROQ_API_KEYS found in environment.")
    keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    key = keys[_request_counter % len(keys)]
    _request_counter += 1
    return OpenAI(base_url=GROQ_BASE, api_key=key)

def groq_chat(messages, max_tokens=2048, temperature=0):
    """
    Sends a prompt to the Groq API using a configured judge model.
    Includes robust retries and exponential backoff to handle rate limits.
    """
    import time
    max_retries = 6
    delay = 2.0
    for attempt in range(max_retries):
        try:
            client = get_groq_client()
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error calling Groq after {max_retries} attempts: {e}")
                raise e
            time.sleep(delay)
            delay *= 2.0


try:
    groq_chat([{"role": "user", "content": "Reply with only: OK"}], max_tokens=5)
except Exception as e:
    raise RuntimeError(f"Failed to connect to Groq: {e}")

try:
    embedder = SentenceTransformer(EMB_MODEL)
except Exception as e:
    raise RuntimeError(f"Failed to load local embedding model {EMB_MODEL}: {e}")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)

all_samples = []
for rec in raw:
    query = rec.get("query", "")
    body = rec.get("response_body") or {}
    gen = body.get("generation_result") or {}
    answer = gen.get("answer", "")
    retrieval = body.get("retrieval_result") or {}
    contexts = []
    for eb in retrieval.get("evaluation_bundles", []):
        bundle = eb.get("bundle") or {}
        for item in bundle.get("items", []):
            t = item.get("text")
            if t:
                contexts.append(t)
    if answer and contexts:
        all_samples.append({"question": query, "answer": answer, "contexts": contexts})

samples = all_samples[SAMPLE_START:SAMPLE_END]

def faithfulness(answer: str, contexts: list[str]) -> float | None:
    ctx = "\n---\n".join(contexts[:MAX_CTX_DOCS])
    prompt = (
        "You are a factual grounding evaluator for a RAG system.\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Step 1: List each distinct factual claim made in ANSWER.\n"
        "Step 2: For each claim, mark it SUPPORTED or UNSUPPORTED based only on CONTEXT.\n"
        "Step 3: Special rule -- if ANSWER correctly states that the context does "
        "not contain enough information to answer the question (a correct refusal), "
        "and the context genuinely does not support an answer, this counts as fully "
        "faithful: score it 1.0. Do not penalize correct abstention.\n"
        "Step 4: Compute the fraction of claims that are SUPPORTED "
        "(or apply the Step 3 rule if it's a correct abstention).\n\n"
        "Output your claim list and reasoning, then on the FINAL line output "
        "only the score in this exact format:\n"
        "SCORE: <decimal between 0.0 and 1.0>"
    )
    try:
        text = groq_chat([{"role": "user", "content": prompt}], max_tokens=1024, temperature=0)
        match = re.search(r"SCORE\s*:\s*\*?\s*([01]\.?\d*)", text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        fallback = re.findall(r"\b([01]\.\d+)\b", text)
        if fallback:
            return float(fallback[-1])
    except Exception:
        pass
    return None

def answer_relevancy(question: str, answer: str) -> float | None:
    gen_prompt = (
        f"Given this answer, write exactly {N_GEN_QS} questions it is answering.\n"
        f"Answer: {answer}\n"
        f"Output exactly {N_GEN_QS} questions, one per line, no numbering or bullets."
    )
    try:
        text = groq_chat([{"role": "user", "content": gen_prompt}], max_tokens=512, temperature=0.3)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        gen_qs = lines[:N_GEN_QS]
        if not gen_qs:
            return None
        texts = [question] + gen_qs
        vecs = embedder.encode(texts)
        orig = vecs[0]
        sims = [
            float(np.dot(orig, v) / (np.linalg.norm(orig) * np.linalg.norm(v)))
            for v in vecs[1:]
        ]
        return float(np.mean(sims))
    except Exception:
        pass
    return None

results = []
for s in tqdm(samples, desc="Evaluating"):
    f_score = faithfulness(s["answer"], s["contexts"])
    ar_score = answer_relevancy(s["question"], s["answer"])
    results.append({
        "question": s["question"],
        "answer": s["answer"],
        "faithfulness": f_score,
        "answer_relevancy": ar_score,
    })
    import time
    time.sleep(1.0)

f_scores = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
ar_scores = [r["answer_relevancy"] for r in results if r["answer_relevancy"] is not None]

print("\n==================== RESULTS ====================")
print(f"Total samples processed: {len(results)}")
print(f"Faithfulness scored    : {len(f_scores)}/{len(results)}")
if f_scores:
    print(f"  Mean               : {np.mean(f_scores):.4f}")
    print(f"  Min / Max          : {min(f_scores):.4f} / {max(f_scores):.4f}")
print()
print(f"Answer Relevancy scored: {len(ar_scores)}/{len(results)}")
if ar_scores:
    print(f"  Mean                 : {np.mean(ar_scores):.4f}")
    print(f"  Min / Max            : {min(ar_scores):.4f} / {max(ar_scores):.4f}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump({
        "summary": {
            "judge_model": JUDGE_MODEL,
            "judge_provider": "groq",
            "n_samples": len(results),
            "faithfulness": {
                "mean": round(float(np.mean(f_scores)), 4) if f_scores else None,
                "min": round(float(min(f_scores)), 4) if f_scores else None,
                "max": round(float(max(f_scores)), 4) if f_scores else None,
            },
            "answer_relevancy": {
                "mean": round(float(np.mean(ar_scores)), 4) if ar_scores else None,
                "min": round(float(min(ar_scores)), 4) if ar_scores else None,
                "max": round(float(max(ar_scores)), 4) if ar_scores else None,
            },
        },
        "per_sample": results,
    }, f, indent=2)

print(f"\nMetrics saved to {OUTPUT_FILE}")