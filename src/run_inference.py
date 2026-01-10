from pathlib import Path
import pandas as pd
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer, util
import faiss

nltk.download("punkt", quiet=True)

# ---------------- CONFIG ----------------
TOP_K = 5
SIM_THRESHOLD = 0.55   # 🔥 THIS is the key

# ---------------- HELPERS ----------------
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def extract_claims(text):
    return nltk.sent_tokenize(text)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    test_df = pd.read_csv("data/test.csv")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    labels = []
    rationales = []

    book_cache = {}  # cache embeddings per book

    for i, row in test_df.iterrows():
        print(f"🔄 Processing {i+1}/{len(test_df)} | ID={row['id']}")

        book_name = row["book_name"]
        backstory = row["content"]

        # ---------- Load + index book ONCE ----------
        if book_name not in book_cache:
            book_path = Path(f"data/books/{book_name}.txt")
            novel_text = load_text(book_path)
            chunks = chunk_text(novel_text)

            chunk_emb = model.encode(
                chunks, convert_to_numpy=True, normalize_embeddings=True
            ).astype("float32")

            index = faiss.IndexFlatIP(chunk_emb.shape[1])
            index.add(chunk_emb)

            book_cache[book_name] = (chunks, index, chunk_emb)

        chunks, index, chunk_emb = book_cache[book_name]

        # ---------- Check claims ----------
        claims = extract_claims(backstory)
        claim_scores = []

        for claim in claims:
            q_emb = model.encode(
                claim, convert_to_numpy=True, normalize_embeddings=True
            ).astype("float32").reshape(1, -1)

            scores, idxs = index.search(q_emb, TOP_K)
            best_score = float(scores[0][0])
            claim_scores.append(best_score)

        # ---------- FINAL DECISION ----------
        min_score = min(claim_scores)

        if min_score < SIM_THRESHOLD:
            label = 0
            rationale = (
                f"At least one claim is unsupported by the source text "
                f"(min similarity = {min_score:.2f})"
            )
        else:
            label = 1
            rationale = (
                f"All claims are semantically supported by the narrative "
                f"(min similarity = {min_score:.2f})"
            )

        labels.append(label)
        rationales.append(rationale)

    # ---------- OUTPUT ----------
    test_df["label"] = labels
    test_df["rationale"] = rationales
    test_df.to_csv("results.csv", index=False)

    print("\n✅ DONE: results.csv generated with correct format and real 0/1 variation")
