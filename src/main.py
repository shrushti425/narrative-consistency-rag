"""
main.py
-------
Sanity-check + demo entrypoint for Narrative Consistency system
Uses Pathway-backed retrieval (Track A compliant)
"""

from pathlib import Path
import nltk
from sentence_transformers import SentenceTransformer

from pathway_store import build_index
from run_inference import (
    load_text,
    chunk_text,
    extract_claims,
    classify_claim,
    aggregate_verdicts,
)

# ---------------- SETUP ----------------
nltk.download("punkt", quiet=True)
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- DEMO RUN ----------------
if __name__ == "__main__":

    # ---- Example input (single case demo) ----
    book_name = "The Count of Monte Cristo"
    backstory = (
        "Edmond Dantès was always trusted by his peers and showed fearless "
        "leadership early in his career."
    )

    print("\n📘 Book:", book_name)
    print("🧾 Backstory:", backstory)

    # ---- Load novel ----
    book_path = Path(f"data/books/{book_name}.txt")
    novel_text = load_text(book_path)

    # ---- Chunk + Pathway index ----
    chunks = chunk_text(novel_text)
    pathway_index = build_index({book_name: chunks})

    # ---- Claim extraction ----
    claims = extract_claims(backstory)
    verdicts = []

    print("\n📜 Claims:")
    for c in claims:
        print(" -", c)

    # ---- Evidence retrieval + reasoning ----
    for claim in claims:
        query_emb = model.encode(claim).tolist()
        results = pathway_index.query(query_emb)

        evidence_chunks = [r["chunk"] for r in results]
        verdict = classify_claim(claim, evidence_chunks)
        verdicts.append(verdict)

        print(f"\nCLAIM → {claim}")
        print("VERDICT →", verdict)
        print("EVIDENCE:")
        for ev in evidence_chunks[:2]:
            print(" •", ev[:180].replace("\n", " "))

    # ---- Final decision ----
    final_label = aggregate_verdicts(verdicts)

    print("\n📊 Claim Verdicts:", verdicts)
    print("🏁 FINAL CONSISTENCY LABEL:", final_label)

