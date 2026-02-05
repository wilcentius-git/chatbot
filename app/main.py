from fastapi import FastAPI
from pydantic import BaseModel
import os

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.faq import FAQEngine
from app.legal import LegalRefEngine
from app.logger import log_chat

app = FastAPI(title="Kemenkum Bot (UI + FAQ + Legal KUHP)")

# =========================
# Schema
# =========================
class ChatRequest(BaseModel):
    message: str


# =========================
# Intent Gate
# =========================
def classify_intent(text: str) -> str:
    t = (text or "").lower()
    legal_keywords = ["pasal", "uu", "undang", "kuhap", "kuhp", "permen", "pp"]
    if any(k in t for k in legal_keywords):
        return "LEGAL"
    return "FAQ"


# =========================
# UX Wrappers (lebih hidup)
# =========================
def wrap_faq_response(answer: str, matched: str | None) -> str:
    intro = "Oke, saya bantu ya."
    if matched:
        intro = f"Oke, ini biasanya terkait **{matched}**."
    follow_up = "Kalau masih bermasalah, kamu bisa jelaskan error yang muncul (atau screenshot error-nya)."
    return f"{intro}\n\n{answer}\n\n{follow_up}"


def wrap_legal_ref_response(answer: str) -> str:
    intro = "Saya bisa bantu tampilkan teks pasalnya sebagai referensi (read-only)."
    follow_up = "Kalau mau, sebutkan pasal lain yang ingin dilihat."
    return f"{intro}\n\n{answer}\n\n{follow_up}"


# =========================
# Init Engines
# =========================
BASE_DIR = os.path.dirname(__file__)

FAQ_CSV = os.path.join(BASE_DIR, "data", "kemenkum_faq.csv")
faq = FAQEngine(FAQ_CSV)
FAQ_THRESHOLD = 0.25

LEGAL_KUHP_CSV = os.path.join(BASE_DIR, "data", "legal_kuhp.csv")
legal_ref = LegalRefEngine(LEGAL_KUHP_CSV)

# =========================
# Static UI
# =========================
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    msg = (req.message or "").strip()
    if not msg:
        return {"intent": "SYSTEM", "reply": "Silakan tulis pertanyaan terlebih dahulu."}

    intent = classify_intent(msg)

    # ===== FAQ MODE (TF-IDF) =====
    if intent == "FAQ":
        result = faq.search(msg)

        log_chat(
            message=msg,
            intent="FAQ",
            matched=result.get("matched"),
            score=result.get("score"),
        )

        if result.get("score", 0.0) < FAQ_THRESHOLD:
            return {
                "intent": "FAQ",
                "matched": result.get("matched"),
                "score": result.get("score"),
                "reply": (
                    "Saya belum yakin maksudnya yang mana.\n\n"
                    "Ini lebih ke:\n"
                    "- reset password\n"
                    "- akun terkunci\n"
                    "- OTP tidak masuk\n\n"
                    "Balas dengan salah satu ya."
                ),
            }

        return {
            "intent": "FAQ",
            "matched": result.get("matched"),
            "score": result.get("score"),
            "reply": wrap_faq_response(
                answer=result.get("answer", ""),
                matched=result.get("matched"),
            ),
        }

    # ===== LEGAL REF MODE (KUHP CSV) =====
    result = legal_ref.lookup(msg)

    log_chat(
        message=msg,
        intent="LEGAL_REF",
        matched=result.get("ref"),
        score=None,
    )

    return {
        "intent": "LEGAL_REF",
        "matched": result.get("ref"),
        "reply": wrap_legal_ref_response(result.get("answer", "")),
    }
