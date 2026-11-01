# Meeting Task Extractor (V1 & V2)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-4.40%2B-yellow.svg)](https://huggingface.co/)
[![spaCy](https://img.shields.io/badge/spaCy-3.8%2B-09A3D5.svg)](https://spacy.io/)
[![Faster-Whisper](https://img.shields.io/badge/Faster--Whisper-1.0%2B-green.svg)](https://github.com/SYSTRAN/faster-whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade AI system for automatically transcribing multi-party meeting audio and extracting actionable tasks, assignees, deadlines, and supporting linguistic evidence from transcripts (plain text, WebVTT, or audio recordings).

> **Privacy & Offline Architecture Specification:**  
> **100% Local, Offline Execution.** This system does **not** rely on external LLM APIs, cloud inference, or third-party web services. All ASR transcription, transformer classification, dependency parsing, and slot filling execute entirely on the local machine.

---

## 1. System Architecture

```
 meeting-task-extractor/
 ├── backend/
 │   ├── app/
 │   │   ├── api/
 │   │   │   ├── __init__.py
 │   │   │   └── routes.py             # REST endpoints (extract, transcribe, CRUD, benchmark, export)
 │   │   ├── core/
 │   │   │   ├── config.py             # App settings, paths, model thresholds
 │   │   │   └── database.py           # SQLite connection & SQLAlchemy session management
 │   │   ├── models/
 │   │   │   ├── meeting.py            # Meeting ORM model
 │   │   │   └── task.py               # Task ORM model
 │   │   ├── schemas/
 │   │   │   ├── meeting.py            # Pydantic validation schemas for Meetings
 │   │   │   └── task.py               # Pydantic validation schemas for Tasks
 │   │   ├── services/
 │   │   │   ├── audio_transcriber.py  # V2 Offline Faster-Whisper ASR pipeline
 │   │   │   ├── preprocessing.py      # Stage A: WebVTT/TXT parser, sentence tokenization, speaker attribution
 │   │   │   ├── transformer_extractor.py # V2 Fine-tuned RoBERTa-base inference with temperature calibration
 │   │   │   ├── action_classifier.py  # Stage B: V1 TF-IDF+LogReg ML, rule baseline, and consensus boosting
 │   │   │   ├── task_extractor.py     # Stage C: Dependency parsing & predicate chunk extraction
 │   │   │   ├── assignee_extractor.py # Stage D: Speaker attribution, spaCy NER, vocatives, pronoun resolution
 │   │   │   ├── deadline_extractor.py # Stage E: dateparser & relative temporal rules with reference date
 │   │   │   ├── validation.py         # Stage F: Confidence calibration & deduplication
 │   │   │   └── pipeline.py           # Master end-to-end NLP pipeline orchestrator
 │   │   └── main.py                   # FastAPI app entry point, CORS, and routing
 │   ├── training/
 │   │   ├── dataset_large.json        # 15,179 AMI Meeting Corpus utterances (train/test split by meeting)
 │   │   ├── model.joblib              # Serialized V1 TF-IDF + LogisticRegression model
 │   │   ├── v2_distilbert/            # V2 RoBERTa-base model weights (model.safetensors, config, tokenizer)
 │   │   ├── benchmark_report.json     # Empirical benchmark evaluation across 7 architectures
 │   │   ├── evaluate_models.py        # Multi-model evaluation and metrics aggregation script
 │   │   ├── train_classifier.py       # V1 Classical ML training script
 │   │   └── Meeting_Task_Extractor_V2_Colab.ipynb # High-accuracy GPU Colab training notebook
 │   ├── tests/
 │   │   ├── test_nlp_pipeline.py      # Linguistic test cases (actionable, completed, suggestions, negation)
 │   │   └── test_api.py               # API endpoints, CRUD, persistence, export, error handling
 │   └── requirements.txt
 ├── frontend/
 │   ├── src/
 │   │   ├── components/
 │   │   │   ├── Navbar.jsx            # Top navigation & system status badge
 │   │   │   ├── Sidebar.jsx           # Clean left navigation (Dashboard, New Extraction, Saved Meetings, Benchmark)
 │   │   │   ├── Dashboard.jsx         # KPI metric cards & recent meetings overview
 │   │   │   ├── TranscriptEditor.jsx  # Workspace studio: text/vtt/audio upload, engine selector, sample loaders
 │   │   │   ├── TaskTable.jsx         # Structured table, inline status cycles, quick confirm, review flags
 │   │   │   ├── BenchmarkView.jsx     # Model evaluation matrix, AMI metrics, and latency comparison
 │   │   │   ├── TaskDetailModal.jsx   # Source evidence modal & confidence penalty breakdown
 │   │   │   ├── EditTaskModal.jsx     # Full task editor modal
 │   │   │   └── ConfirmModal.jsx      # Accessible confirmation modal
 │   │   ├── services/
 │   │   │   └── api.js                # Frontend API client
 │   │   ├── styles/
 │   │   │   ├── index.css             # Design tokens, typography, clean enterprise theme
 │   │   │   └── App.css               # Component layout, table styling, animations
 │   │   ├── App.jsx                   # Application routing and view state coordination
 │   │   └── main.jsx
 │   ├── index.html
 │   └── package.json
 └── README.md
```

---

## 2. Dual-Engine NLP Pipeline: V1 vs V2

The system features two complementary inference engines that users can select in the workspace:

```
Raw Meeting Audio (.mp3, .wav) ──► [Faster-Whisper ASR (CTranslate2 INT8)]
                                              │
Transcript (.txt, .vtt) ──────────────────────┤
                                              ▼
                             [Stage A: Preprocessing & Parsing]
                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
              ▼                                                               ▼
   [V1 Classical ML Engine]                                        [V2 Neural Transformer]
• TF-IDF (1-2 ngrams) + LogReg                                  • Fine-Tuned RoBERTa-base (125M params)
• Sub-millisecond latency (0.015 ms)                            • Deep conversational context & phrasing
• Zero GPU required                                             • 78.3% Action-Item Recall
              │                                                               │
              └───────────────────────────────┬───────────────────────────────┘
                                              ▼
                             [Neural-Symbolic Consensus Layer]
                             • Temperature Calibration (T=0.55)
                             • Heuristic Confidence Boost (0.94-0.97)
                             • Procedural Chatter Filtering
                                              │
                                              ▼
                         [Slot-Filling & Information Extraction]
                         • Task Predicate Extraction (spaCy Dependency Tree)
                         • Assignee Resolution (Vocative + Turn Speaker + Pronouns)
                         • Deadline Normalization (Relative temporal phrases to ISO 8601)
                                              │
                                              ▼
                         [Validation, Review Flags & Deduplication]
```

### Extraction Stages:
* **Stage A: Preprocessing (`preprocessing.py`)**: WebVTT header and cue timestamp stripping, speaker turn attribution (`Speaker: Utterance`), and sentence segmentation.
* **Stage B: Dialogue Act Classification (`action_classifier.py` / `transformer_extractor.py`)**:
  * `ACTION_ITEM`: Tasks or commitments requiring execution.
  * `SUGGESTION`: Hypothetical ideas or proposals without firm commitment.
  * `COMPLETED_ACTION`: Explicitly finished tasks (*"I already updated the docs yesterday"*).
  * `NON_ACTIONABLE`: Casual conversation, status updates, and procedural chatter (*"Good morning everyone"*).
* **Stage C: Task Description Extraction (`task_extractor.py`)**: Extracts the head verb predicate chunk using spaCy dependency parse trees while removing conversational preambles (*"Rahul, please..."*, *"I will..."*) and temporal modifiers.
* **Stage D: Assignee Resolution (`assignee_extractor.py`)**: Resolves directed vocatives (*"Amit, please review..."* &rarr; `Amit`), first-person commitments (*"I will..."* &rarr; Speaker), and indefinite pronouns (*"We need to..."* &rarr; `Unassigned` with `needs_review = True`).
* **Stage E: Deadline Normalization (`deadline_extractor.py`)**: Resolves relative expressions (*"by Wednesday"*, *"before next Friday"*, *"in 3 days"*) against the meeting reference date into standardized **ISO 8601 (`YYYY-MM-DD`)** dates.
* **Stage F: Confidence Calibration & Deduplication (`validation.py`)**: Multi-factor calibration deducting penalties for missing assignees, ambiguous deadlines, or weak predicates, while flagging uncertain items for human review.

---

## 3. Empirical Benchmark & Model Evaluation

Trained and evaluated on the **15,179 dialogue utterances** of the **AMI Meeting Corpus (1.6.2)** across 86 multi-party meetings, partitioned strictly by meeting ID (64 train / 22 test) to prevent cross-meeting data leakage:

### Multi-Model Performance Matrix (3,794 Held-Out AMI Test Utterances):

| Model Architecture | Accuracy | Action-Item F1 | Action Recall | Macro F1 | Weighted F1 | CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V2 Neural Transformer (RoBERTa-base)** ⭐ | **76.88%** | **54.40%** | **78.34%** | **0.6821** | **0.7765** | **18.6 ms** |
| **V1 Best ML (Logistic Regression Balanced)** | 73.38% | 46.59% | 56.68% | 0.6547 | 0.7395 | **0.015 ms** |
| **Linear SVM (`LinearSVC`)** | 74.01% | 44.50% | 40.55% | 0.6401 | 0.7369 | 0.014 ms |
| **Random Forest (100 Trees)** | 74.14% | 40.77% | 34.56% | 0.6311 | 0.7384 | 0.047 ms |
| **Multinomial Naive Bayes** | 73.01% | 26.80% | 18.89% | 0.4137 | 0.7017 | 0.015 ms |
| **Rule-Based Baseline (spaCy + Patterns)** | 64.34% | 20.64% | 62.40% | 0.4243 | 0.5637 | 5.426 ms |

### Key Findings:
1. **Action Item Recall Breakthrough**: V2 RoBERTa achieves **78.34% Recall** on real action items (catching 170 out of 217 commitments in held-out test dialogue), representing a **+21.6% absolute gain** over V1 Logistic Regression (56.68%).
2. **Elimination of Majority-Class Collapse**: Using **Stratified High-Signal Balancing**, the model overcomes the 66.5% non-actionable chatter distribution, achieving non-zero, robust F1 across all four dialogue act classes.
3. **Speed vs Precision Tradeoff**: V1 executes in **0.015 ms** (66,000 utterances/sec on CPU) for instant edge filtering, while V2 provides deep contextual understanding in **18.6 ms**.

---

## 4. Technology Stack

* **Frontend**: React 19, Vite, Lucide React, Vanilla CSS with custom tokens (Linear/Notion aesthetic).
* **Backend**: Python 3.10+, FastAPI, Pydantic V2, Uvicorn, SQLAlchemy, SQLite.
* **NLP & Deep Learning**:
  * **PyTorch & Hugging Face Transformers**: Fine-tuned `roberta-base` sequence classification.
  * **spaCy (`en_core_web_sm`)**: Syntactic dependency parsing, POS tagging, and entity boundary detection.
  * **scikit-learn**: TF-IDF vectorization and classical ML baselines.
  * **Faster-Whisper**: Local speech-to-text transcription via CTranslate2 (INT8).
  * **dateparser**: Multi-format relative and explicit temporal resolution.

---

## 5. Getting Started

### Prerequisites
* Python 3.10 to 3.13
* Node.js 18+ and npm

### 1. Backend Setup
```bash
# Navigate to backend
cd backend

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm

# Start FastAPI server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Backend API will be available at: **`http://127.0.0.1:8000`**  
Interactive Swagger docs: **`http://127.0.0.1:8000/docs`**

### 2. Frontend Setup
```bash
# Navigate to frontend in a new terminal
cd frontend

# Install dependencies
npm install

# Start Vite dev server
npm run dev
```
Web Application will be available at: **`http://127.0.0.1:5173`**

---

## 6. Running Tests

The test suite validates both API functionality and linguistic edge cases:

```bash
# Run all unit tests
pytest backend/tests/ -v
```
All **32 tests** pass with 100% test success rate.

---

## 7. Export & Integration

Extracted tasks can be exported directly from the UI or API:
* **CSV Export**: Fully RFC-4180 compliant with UTF-8 BOM (`\uFEFF`) for seamless opening in Microsoft Excel, Apple Numbers, and Google Sheets.
* **JSON Export**: Structured export formatted with full evidence context, normalized deadlines, and confidence metadata.
