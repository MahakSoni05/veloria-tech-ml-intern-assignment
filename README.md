<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&pause=1000&color=2196F3&center=true&vCenter=true&width=1000&lines=Cricket+ML+Intelligence+Platform;Web+Scraping+%E2%86%92+Prediction+%E2%86%92+Semantic+Search;End-to-End+AI+%2F+ML+Pipeline" alt="Typing SVG" />

<br/>

**A production-ready AI pipeline that scrapes cricket data, predicts match outcomes with Random Forest, and enables natural language search using Sentence Transformer vector embeddings — all in Python.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NLP](https://img.shields.io/badge/Sentence_Transformers-NLP-4CAF50?style=for-the-badge)](https://sbert.net)
[![RAG](https://img.shields.io/badge/RAG-Vector_Search-9C27B0?style=for-the-badge)](https://github.com/MahakSoni05/cricket-ml-intelligence-platform)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

[📌 About](#-about) · [🔧 Pipeline](#-pipeline) · [🚀 Quick Start](#-quick-start) · [📊 Results](#-results) · [🧠 How It Works](#-how-it-works) · [👩‍💻 Author](#-author)

</div>

---

## 📌 About

Most sports analytics tools either predict OR search — this project does both, in one clean pipeline.

**What it solves:**
- Automates cricket match data collection (no manual CSV downloads)
- Predicts which team wins based on historical patterns
- Lets you search match history in plain English — *"Show me matches where Australia won at Melbourne"* — using semantic similarity, not keyword matching

**Why it matters to recruiters:**
This demonstrates three skills that matter in production ML roles — **data engineering**, **supervised learning**, and **semantic retrieval (RAG)** — in a single cohesive project.

---

## 🔧 Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    3-STAGE ML PIPELINE                           │
├──────────────┬─────────────────────┬────────────────────────────┤
│   STAGE 1    │      STAGE 2        │         STAGE 3  ✦         │
│              │                     │                            │
│  Web Scraping│  Match Prediction   │    Semantic Search         │
│              │                     │                            │
│  requests +  │  Random Forest      │  Sentence Transformers     │
│  BS4 →       │  Classifier →       │  384-dim Embeddings →      │
│  match_data  │  Accuracy · F1 ·    │  Cosine Similarity →       │
│  .csv        │  Confusion Matrix   │  Top-3 Relevant Matches    │
└──────────────┴─────────────────────┴────────────────────────────┘
```

> ✦ Stage 3 is the standout component — it implements a lightweight RAG (Retrieval-Augmented Generation) architecture used in production AI systems at scale.

---

## 📁 Project Structure

```
cricket-ml-intelligence-platform/
│
├── 📄 scraper.py          ← Stage 1: Automated data collection
├── 📄 model.py            ← Stage 2: ML training + evaluation
├── 📄 rag_search.py       ← Stage 3: Semantic search engine
├── 📊 match_data.csv      ← Scraped dataset (10 match records)
├── 📋 requirements.txt    ← All dependencies
└── 📖 README.md
```

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/MahakSoni05/cricket-ml-intelligence-platform.git
cd cricket-ml-intelligence-platform

# Install
pip install -r requirements.txt

# Run all 3 stages
python scraper.py      # → generates match_data.csv
python model.py        # → trains model, prints results
python rag_search.py   # → interactive semantic search
```

---

## 📊 Results

### Stage 1 — Data Collected

| Field | Description |
|-------|-------------|
| `match_date` | Date of the match |
| `team_1` / `team_2` | Competing teams |
| `venue` | Stadium · City |
| `winner` | Match result |
| `top_scorer` | Player name |
| `top_score` | Runs scored |

10 structured records scraped and saved to `match_data.csv`.

---

### Stage 2 — ML Model Output

```
Algorithm        : Random Forest Classifier
Features         : team_1, team_2, venue (label-encoded)
Train / Test     : 80% / 20%
Accuracy Score   : Reported (baseline on 10-record dataset)
F1 Score         : Calculated via Scikit-learn
Confusion Matrix : Generated ✓
```

> **Engineering note:** Baseline performance reflects a 10-sample dataset. The pipeline is built to scale — replacing `match_data.csv` with a 500+ record Kaggle dataset requires zero code changes and will significantly improve metrics.

---

### Stage 3 — Semantic Search Output ✦

```bash
$ python rag_search.py
> Query: "Show me matches where Australia won"

Most Relevant Matches:

[1] Australia vs India · Melbourne Cricket Ground · 2024-12-26
    Result: Australia won | Top scorer: Steve Smith · 140 runs

[2] Australia vs India · Sydney Cricket Ground · 2025-01-03
    Result: Australia won | Top scorer: Rishabh Pant · 61 runs

[3] Australia vs India · Brisbane Cricket Ground · 2024-12-14
    Result: Draw | Top scorer: Travis Head · 152 runs
```

---

## 🧠 How It Works

### Semantic Search — Under the Hood

```python
# Each match record → natural language sentence
sentence = "Australia vs India at Melbourne on 2024-12-26. Australia won. Top scorer: Steve Smith with 140 runs."

# Sentence → 384-dimensional dense vector
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(sentence)   # shape: (384,)

# Query → cosine similarity → top-3 results
scores = cosine_similarity([query_vector], corpus_vectors)
top_3 = scores.argsort()[-3:][::-1]
```

This is the same retrieval mechanism used in production RAG pipelines at companies like Google, Notion, and Perplexity — just at a smaller scale.

---

### Why These Tech Choices?

| Decision | Reasoning |
|----------|-----------|
| **Random Forest over Logistic Regression** | Handles label-encoded categorical features (team, venue) without assuming independence. More robust on small, noisy tabular datasets. |
| **all-MiniLM-L6-v2** | Best quality-to-cost ratio for semantic embeddings — 384 dims, runs on CPU, no GPU needed. |
| **NumPy cosine similarity over ChromaDB** | For 10 records, a vector DB adds unnecessary overhead. Architecture is modular — swap to FAISS/ChromaDB in one function for production scale. |

---

## 📈 Scalability Path

This project is intentionally modular:

```
Current state:  10 records → baseline metrics
Scale to:       500+ Kaggle records → production-grade accuracy
Add:            Home/away flags + head-to-head history → richer features  
Extend to:      LLM (GPT / Claude) + vector search → full RAG Q&A system
Deploy via:     Streamlit / FastAPI → shareable demo link
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|-------|-----------|
| Language | Python 3.8+ |
| Scraping | `requests` · `BeautifulSoup4` |
| Data Processing | `Pandas` · `NumPy` |
| Machine Learning | `Scikit-learn` — Random Forest |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector Search | Cosine Similarity (NumPy) |
| Environment | VS Code · macOS |

</div>

---

## 👩‍💻 Author

<div align="center">

**Mahak Soni**
B.Tech CSE · Data Science Specialization
Bennett University · Greater Noida · Batch 2028

[![GitHub](https://img.shields.io/badge/GitHub-MahakSoni05-181717?style=for-the-badge&logo=github)](https://github.com/MahakSoni05)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-mahak--soni-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/mahak-soni-097905303)

*Open to AI/ML · Data Science · Python internship opportunities*

</div>

---

<div align="center">

⭐ If this project helped you or impressed you — consider starring the repo!

</div>
