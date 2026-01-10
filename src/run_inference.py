"""
run_inference.py
----------------
Final inference pipeline for Kharagpur Data Science Hackathon 2026 (Track A)

- Uses Pathway for long-context storage + retrieval
- Performs evidence-based narrative consistency classification
- Outputs results.csv in REQUIRED format:
  id,prediction,rationale
"""

from pathlib import Path
import pandas as pd
import nltk
from sentence_transformers import SentenceTransformer

import pathway as pw

# ------------------ SETUP ------------------
nltk.download("punkt", quiet=True)
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ HELPERS ------------------
def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text: str, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def extract_claims(backstory: str):
    return nltk.sent_tokenize(backstory)

def classify_claim(claim, evidence_chunks):
    """
    Heuristic constraint-based classifier
    """
    claim_lower = claim.lower()

    support_words = ["leader", "led", "trusted", "fearless", "brave"]
    contradict_words = ["afraid", "feared", "avoided", "hesitant", "distrusted"]

    score = 0
    for chunk in evidence_chunks:
        text = chunk.lower()
        for w in support_words:
            if w in claim_lower and w in text:
                score += 1
        for w in contradict_words:
            if w in claim_lower and w in text:
                score -= 1

    if score > 0:
        return "SUPPORTED"
    elif score < 0:
        return "CONTRADICTED"
    else:
        return "UNCLEAR"

def aggregate_verdicts(verdicts):
    return 0 if "CONTRADICTED" in verdicts else 1

# ------------------ PATHWAY INDEX ------------------
class NovelSchema(pw.Schema):
    book_name: str
    chunk: str
    embedding: list[float]

def build_pathway_index(book_name, chunks):
    rows = []
    for ch in chunks:
        emb = model.encode(ch).tolist()
        rows.append((book_name, ch, emb))

    table = pw.Table.from_rows(
        rows,
        schema=NovelSchema
    )

    index = pw.ml.index.KNNIndex(
        table.embedding,
        table,
        k=5
    )
    return index

# ------------------ MAIN PIPELINE ------------------
if __name__ == "__main__":

    test_df = pd.read_csv("data/test.csv")

    book_cache = {}   # book_name → Pathway index
    results = []

    for i, row in test_df.iterrows():
        story_id = row["id"]
        book_name = row["book_name"]
        backstory = row["content"]

        print(f"🔄 Processing {i+1}/{len(test_df)} | ID={story_id}")

        # Build Pathway index ONCE per book
        if book_name not in book_cache:
            book_path = Path(f"data/books/{book_name}.txt")
            novel_text = load_text(book_path)
            chunks = chunk_text(novel_text)
            index = build_pathway_index(book_name, chunks)
            book_cache[book_name] = index

        index = book_cache[book_name]

        claims = extract_claims(backstory)
        verdicts = []

        for claim in claims:
            query_emb = model.encode(claim).tolist()
            retrieved = index.query(query_emb)
            evidence = [r["chunk"] for r in retrieved]
            verdicts.append(classify_claim(claim, evidence))

        final_label = aggregate_verdicts(verdicts)

        # ---- REQUIRED OUTPUT FORMAT ----
        rationale = (
            f"{verdicts.count('CONTRADICTED')} contradictions detected across "
            f"{len(claims)} extracted claims"
        )

        results.append({
            "id": story_id,
            "prediction": final_label,
            "rationale": rationale
        })

    # ---- SAVE RESULTS ----
    pd.DataFrame(results).to_csv("results.csv", index=False)
    print("\n✅ DONE: results.csv generated in correct submission format")

