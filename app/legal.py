import pandas as pd

class LegalRefEngine:
    def __init__(self, csv_path: str):
        # Robust load (Windows encoding guard)
        try:
            self.df = pd.read_csv(csv_path, encoding="utf-8", sep=",")
        except UnicodeDecodeError:
            self.df = pd.read_csv(csv_path, encoding="cp1252", sep=",")

        required = {"doc", "year", "ref", "title", "text"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"legal CSV missing columns: {missing}. "
                f"Kolom terbaca: {list(self.df.columns)}"
            )

        # Normalisasi ref agar gampang lookup
        self.df["ref_norm"] = (
            self.df["ref"].fillna("").astype(str).str.lower().str.replace(" ", "")
        )

    def lookup(self, query: str) -> dict:
        q = (query or "").lower().replace(" ", "")

        # cari pola "pasal###"
        # contoh: "pasal362", "pasal 362"
        import re
        m = re.search(r"pasal(\d+)", q)
        if not m:
            return {
                "found": False,
                "ref": None,
                "answer": "Sebutkan pasal yang dicari, contoh: 'Pasal 362'."
            }

        target = f"pasal{m.group(1)}"
        hit = self.df[self.df["ref_norm"] == target]

        if hit.empty:
            return {
                "found": False,
                "ref": f"Pasal {m.group(1)}",
                "answer": "Saya belum menemukan pasal itu di dataset KUHP yang saya punya."
            }

        row = hit.iloc[0]
        answer = (
            f"{row['doc']} {row['year']} — {row['ref']} ({row['title']})\n"
            f"{row['text']}\n\n"
            "Catatan: Ini referensi teks, bukan pendapat hukum."
        )

        return {
            "found": True,
            "ref": str(row["ref"]),
            "answer": answer
        }
