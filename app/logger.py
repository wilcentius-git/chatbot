import csv
import os
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "logs.csv")

def log_chat(message: str, intent: str, matched: str | None, score: float | None):
    is_new = not os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp", "message", "intent", "matched", "score"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), message, intent, matched or "", score if score is not None else ""])
