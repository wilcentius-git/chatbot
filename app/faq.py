import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FAQEngine:
    def __init__(self, csv_path: str):
        try:
            self.df = pd.read_csv(csv_path, encoding="utf-8", sep=",")
        except UnicodeDecodeError:
            self.df = pd.read_csv(csv_path, encoding="cp1252", sep=",")

        required = {"question_pattern", "answer_steps", "escalation"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"CSV missing columns: {missing}. Kolom terbaca: {list(self.df.columns)}"
            )

        self.df["question_pattern"] = self.df["question_pattern"].fillna("").astype(str)

        # TF-IDF index
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.df["question_pattern"].tolist())

    def search(self, query: str) -> dict:
        q = (query or "").strip()
        if not q:
            row = self.df.iloc[0]
            return {"answer": f"{row['answer_steps']}\n\nEskalasi: {row['escalation']}",
                    "matched": None,
                    "score": 0.0}

        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.matrix)[0]

        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])

        row = self.df.iloc[best_idx]
        answer = f"{row['answer_steps']}\n\nEskalasi: {row['escalation']}"

        return {"answer": answer, "matched": str(row["question_pattern"]), "score": best_score}
