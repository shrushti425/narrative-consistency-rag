# Narrative Consistency RAG

A system for evaluating whether a generated or user-provided narrative remains **consistent with an original source text** using retrieval and semantic similarity.

This project goes beyond traditional Retrieval-Augmented Generation (RAG) by introducing **claim-level verification**, where narratives are broken into atomic statements and validated against source material.

---

## 🚀 Overview

Maintaining consistency in long-form AI-generated content is a major challenge. This project addresses that by:

- Decomposing narratives into **individual claims**
- Retrieving relevant evidence from source texts
- Measuring semantic alignment using embeddings
- Applying a **strict consistency rule** to determine final correctness

The system ensures that even a single unsupported claim flags the entire narrative as inconsistent.

---

## 🧠 Core Idea

```
Narrative → Claims → Embeddings → Retrieval → Similarity Scoring → Final Verdict
```

Instead of treating text as a whole, the system evaluates **each claim independently**, making it more robust and interpretable.

---

## ⚙️ Key Features

- **Claim-Level Verification**  
  Splits narratives into sentences and evaluates each one independently

- **FAISS-Based Retrieval**  
  Efficient vector similarity search over large text corpora

- **Semantic Matching (Embeddings)**  
  Uses `all-MiniLM-L6-v2` for fast and effective similarity scoring

- **Strict Consistency Rule**  
  Final decision is based on the *weakest claim* (minimum similarity)

- **Batch Inference Pipeline**  
  Processes multiple samples from a dataset and outputs structured results

- **Caching Optimization**  
  Each book is indexed only once, improving performance significantly

---

## 📁 Project Structure

```
src/
│── main.py              # Demo pipeline with claim reasoning + evidence printing
│── run_inference.py     # Core batch inference logic (FAISS-based)
│── config.py            # Configuration (if extended)
│── pathway_store.py     # Retrieval index builder (Pathway-based variant)
```

---

## 🔍 How It Works

### 1. Data Loading
- Book text:
  ```
  data/books/<book_name>.txt
  ```
- Test data:
  ```
  data/test.csv
  ```

---

### 2. Text Chunking
- Splits book into overlapping chunks  
- Default:
  - Chunk size: 500  
  - Overlap: 100  

---

### 3. Embedding + Indexing
- Converts chunks into embeddings  
- Stores them in a **FAISS index** using cosine similarity  

---

### 4. Claim Extraction
- Splits narrative into sentences  
- Each sentence = one claim  

---

### 5. Retrieval + Scoring
For each claim:
- Retrieve top-K relevant chunks  
- Compute similarity  
- Take the best score  

---

### 6. Final Decision Rule

```
If any claim < threshold → Inconsistent (0)
Else → Consistent (1)
```

- Threshold: `0.55`

---

### 7. Output

Generates:

```
results.csv
```

With:
- `label` → 0 (inconsistent) or 1 (consistent)  
- `rationale` → explanation with similarity score  

---

## ▶️ Usage

### Run Batch Inference
```bash
python src/run_inference.py
```

### Run Demo
```bash
python src/main.py
```

---

## ⚙️ Configuration

Modify in `run_inference.py`:

```python
TOP_K = 5
SIM_THRESHOLD = 0.55
```

---

## 🧩 Two Inference Strategies

### 1. FAISS-Based (Primary)
- Fast and scalable  
- Pure similarity-based  

### 2. Pathway-Based (Demo)
- Used in `main.py`  
- Supports retrieval + reasoning (`classify_claim`)  
- Outputs evidence for interpretability  

---

## 📌 Use Cases

- Validating AI-generated stories  
- Checking character consistency  
- Narrative fact-checking  
- Memory-aware AI systems  

---

## ⚡ Strengths

- Interpretable decision logic  
- Efficient retrieval with FAISS  
- Scalable batch processing  
- Optimized with caching  
- Robust to partial inconsistencies  

---

## ⚠️ Limitations

- Sentence-based claim extraction  
- No explicit contradiction detection  
- Uses only top similarity score  
- Limited reasoning in batch mode  

---

## ✨ Author

Shrushti  
https://github.com/shrushti425
