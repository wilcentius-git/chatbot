# app/main.py
from __future__ import annotations

import csv
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import requests
from difflib import SequenceMatcher

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ================= PATH =================
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"

FAQ_CSV = DATA_DIR / "kemenkum_faq.csv"
DJKI_FAQ_CSV = DATA_DIR / "djki_faq.csv"
LEGAL_CSV = DATA_DIR / "legal_kuhp.csv"
LOG_CSV = BASE_DIR / "logs.csv"


# ================= OLLAMA =================
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "tinyllama"
OLLAMA_TIMEOUT = 60
OLLAMA_TYPO_TIMEOUT = 12


# ================= UTIL =================
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().strip())


def ensure_log():
    if LOG_CSV.exists():
        return
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["timestamp", "mode", "intent", "ref", "question", "score", "reply"]
        )


def log_chat(mode, intent, ref, question, score, reply):
    ensure_log()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                mode,
                intent,
                ref,
                question,
                round(score, 3),
                reply,
            ]
        )


# ================= OLLAMA (STRICT FAQ REWRITE ONLY) =================
def call_ollama_strict(context: str) -> str:
    """
    Dipakai HANYA untuk FAQ.
    Kalau ada indikasi nambah isi / langkah / bahasa asing → fallback ke teks asli.
    """

    # Jika konteks berupa langkah-langkah → jangan pakai LLM
    if re.search(r"(^|\n)\s*(\d+\)|\d+\.)\s+", context):
        return context

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Anda editor bahasa resmi.\n"
                    "DILARANG menambah atau mengurangi isi.\n"
                    "DILARANG membuat langkah baru.\n"
                    "WAJIB Bahasa Indonesia.\n"
                    "Jika ragu, kembalikan teks apa adanya."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Rapikan bahasa teks berikut TANPA mengubah makna.\n"
                    "Jangan menambah kalimat baru.\n\n"
                    f"{context}"
                ),
            },
        ],
        "stream": False,
        "options": {"temperature": 0.0, "top_p": 0.8},
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
        out = r.json()["message"]["content"].strip()

        # Guard 1: bahasa Inggris / tutorial tone
        if re.search(r"\b(sure|here's|click|open|step)\b", out.lower()):
            return context

        # Guard 2: kemiripan harus sangat tinggi
        sim = SequenceMatcher(None, normalize(context), normalize(out)).ratio()
        if sim < 0.92:
            return context

        # Guard 3: panjang tidak boleh nambah signifikan
        if len(out) > int(len(context) * 1.05):
            return context

        return out
    except Exception:
        return context


def fix_typos_ollama(text: str) -> str:
    """
    Minta Ollama perbaiki salah ketik saja. Jika gagal atau respons tidak wajar,
    kembalikan teks asli. Dipakai sebelum FAQ matching agar "shaer" → "share" dll.
    """
    text = (text or "").strip()
    if not text or len(text) > 500:
        return text
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Anda hanya memperbaiki salah ketik (typo) pada kalimat bahasa Indonesia. "
                    "Jangan ubah makna, jangan tambah atau kurangi kata. "
                    "Jawab HANYA dengan satu kalimat yang sudah diperbaiki, tanpa penjelasan lain."
                ),
            },
            {"role": "user", "content": text},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 100},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TYPO_TIMEOUT)
        r.raise_for_status()
        out = r.json()["message"]["content"].strip()
        # Hapus tanda kutip jika Ollama membungkus dengan "..."
        out = re.sub(r"^[\"']|[\"']$", "", out).strip()
        if not out or len(out) < 2:
            return text
        # Jangan pakai jika jawaban terlalu panjang (bukan satu kalimat)
        if len(out) > len(text) * 2:
            return text
        return out
    except Exception:
        return text


# ================= LOAD FAQ =================
faq_parts = [pd.read_csv(FAQ_CSV, encoding="utf-8-sig")]
if DJKI_FAQ_CSV.exists():
    faq_parts.append(pd.read_csv(DJKI_FAQ_CSV, encoding="utf-8-sig"))
faq_df = pd.concat(faq_parts, ignore_index=True)

faq_df["answer_compiled"] = (
    faq_df["answer_steps"].astype(str).fillna("").str.strip()
    + "\n\nEskalasikan bila perlu: "
    + faq_df["escalation"].astype(str).fillna("").str.strip()
)

faq_df["question_norm"] = faq_df["question_pattern"].apply(normalize)

# Word n-grams: bagus untuk pertanyaan yang ejaan benar
faq_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
faq_tfidf = faq_vectorizer.fit_transform(faq_df["question_norm"].tolist())

# Character n-grams: tahan typo (mis. "shaer" vs "share" punya overlap karakter)
faq_vectorizer_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
faq_tfidf_char = faq_vectorizer_char.fit_transform(faq_df["question_norm"].tolist())


def faq_query(user_text: str) -> Dict[str, Any]:
    q = normalize(user_text)
    qv = faq_vectorizer.transform([q])
    qv_char = faq_vectorizer_char.transform([q])
    sims_word = cosine_similarity(qv, faq_tfidf).flatten()
    sims_char = cosine_similarity(qv_char, faq_tfidf_char).flatten()
    # Ambil skor terbaik per baris: typo bisa turunkan word_sim tapi char_sim tetap tinggi
    sims = np.maximum(sims_word, sims_char)

    idx = int(sims.argmax())
    score = float(sims[idx])
    row = faq_df.iloc[idx]

    return {
        "intent": row["intent"],
        "ref": row["app"],
        "answer": row["answer_compiled"],
        "score": score,
    }


# ================= LOAD LEGAL (KUHP) =================
legal_df = pd.read_csv(LEGAL_CSV, encoding="utf-8-sig")

# Gunakan ref + title untuk pencarian pasal
legal_df["legal_key"] = (
    legal_df["ref"].astype(str).fillna("") + " " +
    legal_df["title"].astype(str).fillna("")
).apply(normalize)

legal_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
legal_tfidf = legal_vectorizer.fit_transform(legal_df["legal_key"].tolist())


def legal_query(user_text: str) -> Dict[str, Any]:
    q = normalize(user_text)
    qv = legal_vectorizer.transform([q])
    sims = cosine_similarity(qv, legal_tfidf).flatten()

    idx = int(sims.argmax())
    score = float(sims[idx])
    row = legal_df.iloc[idx]

    return {
        "ref": row["ref"],     # contoh: "Pasal 362"
        "text": row["text"],   # isi pasal (verbatim)
        "score": score,
    }


# ================= FASTAPI =================
app = FastAPI(title="Kemenkum Chatbot (Local)", version="1.0")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
def home():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>UI not found</h3>")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    msg = req.message.strip()
    if not msg:
        return {"reply": "Silakan tulis pertanyaan terlebih dahulu."}

    m = normalize(msg)

    # ===== ROUTE LEGAL (KUHP) =====
    if "pasal" in m and "kuhp" in m:
        result = legal_query(msg)

        if result["score"] < 0.25:
            reply = "Saya belum menemukan pasal KUHP yang dimaksud. Pastikan nomor pasal disebutkan dengan jelas."
            log_chat("LEGAL", "", "", msg, result["score"], reply)
            return {"reply": reply, "score": result["score"], "mode": "LEGAL"}

        reply = f"{result['ref']}\n\n{result['text']}"
        log_chat("LEGAL", "", result["ref"], msg, result["score"], reply)
        return {"reply": reply, "score": result["score"], "mode": "LEGAL"}

    # ===== ROUTE FAQ =====
    if "password" in m and "sso" not in m:
        return {"reply": "Reset password untuk aplikasi apa? Contoh: ketik **lupa password sso**."}

    # Koreksi typo dulu (Ollama) agar "shaer" → "share" dll.; fallback ke teks asli
    msg_for_faq = fix_typos_ollama(msg)
    result = faq_query(msg_for_faq)

    if result["score"] < 0.20:
        reply = (
            "Saya belum yakin dengan maksud pertanyaan Anda.\n"
            "Bisa dijelaskan lebih spesifik (aplikasi dan kendalanya)?"
        )
        log_chat("FAQ", result["intent"], result["ref"], msg, result["score"], reply)
        return {"reply": reply, "score": result["score"], "mode": "FAQ"}

    base_answer = result["answer"]
    final_answer = call_ollama_strict(base_answer)

    log_chat("FAQ", result["intent"], result["ref"], msg, result["score"], final_answer)
    return {
        "reply": final_answer,
        "intent": result["intent"],
        "score": result["score"],
        "mode": "FAQ",
    }
