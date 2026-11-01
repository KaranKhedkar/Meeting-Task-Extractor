# Meeting Task Extractor — Project Overview & Technical Report

**Project Title**: Meeting Task Extractor (V1 Classical ML & V2 Deep Neural Transformer)  
**Academic Level**: Final Year B.Tech / B.E. AI & ML Capstone Project  
**Current Phase**: V1 & V2 Fully Implemented, Fine-Tuned, Evaluated, and Deployed  
**Document Generated**: November 2026  

---

## 1. Executive Summary & Objective

**Meeting Task Extractor** is a full-stack, intelligent web application designed to ingest multi-party conversational meeting transcripts or raw audio recordings, analyze dialogue acts, and automatically extract actionable task commitments, responsible assignees, normalized deadlines, and supporting linguistic evidence.

### Core Directives:
1. **V1 Deterministic Baseline**: Establish a privacy-first, low-compute baseline using **classical Natural Language Processing (NLP), rule-based information extraction, and supervised classical Machine Learning (scikit-learn + spaCy)**. V1 executes at microsecond latency on CPU with zero cloud dependencies.
2. **V2 Deep Learning & Audio Intelligence**: Upgrade the classification backbone to a fine-tuned **Bidirectional Transformer (`roberta-base`, 124.6M parameters)** to capture colloquial, nuanced commitments that n-gram models miss, combined with local **Faster-Whisper speech-to-text** for offline audio transcription.
3. **Zero Cloud Dependencies**: Guarantee 100% offline local execution, eliminating data leakage, vendor lock-in, and cloud API latency.

---

## 2. System Architecture & Technology Stack

The project follows a clean client-server architecture with strict separation between user interface, RESTful backend services, database persistence, and NLP/ML extraction modules.

```
┌─────────────────────────────────────────────────────────────┐
│                       Frontend UI                           │
│        React 19 (Vite) + Lucide Icons + Enterprise CSS      │
│  [Sidebar] [Dashboard] [Transcript Studio] [Task Table]     │
│             [Model Benchmark & Evaluation Matrix]           │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / JSON REST APIs
┌──────────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend Server                   │
│   Uvicorn Server • Pydantic V2 Validation • SQLite ORM      │
│   Routes: /extract, /transcribe-audio, /meetings, /benchmark│
└──────────────────────────────┬──────────────────────────────┘
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│    V1 Classical Engine      │ │   V2 Deep Neural Engine     │
│ • TF-IDF (1,2-grams)        │ │ • RoBERTa-base (125M param) │
│ • Balanced Logistic Reg.    │ │ • Faster-Whisper ASR (INT8) │
│ • 0.015 ms/utt CPU Latency  │ │ • 78.34% Action-Item Recall │
└──────────────┬──────────────┘ └──────────────┬──────────────┘
               │                               │
               └───────────────┬───────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            Hybrid Neural-Symbolic Consensus Layer           │
│  • Temperature Calibration (T=0.55)                         │
│  • Rule-Guided Confidence Boosting (0.94 - 0.97)            │
│  • Procedural Chatter Guardrails ("let's start the meeting")│
│  • Slot Filling: Dependency Parsing + dateparser Normalizer │
│  • Human-in-the-Loop Review Flagging (needs_review = True)  │
└─────────────────────────────────────────────────────────────┘
```

### Complete Stack Breakdown:
* **Frontend**:
  * React 19 (Vite build engine, builds in <1 second)
  * Custom CSS design tokens (Linear- and Notion-inspired minimalist productivity interface)
  * Lucide React icons
  * Client-side RFC-4180 compliant CSV (with UTF-8 BOM) and JSON exports
* **Backend**:
  * Python 3.10+ / 3.13+
  * FastAPI (RESTful API framework with async lifecycle)
  * Pydantic V2 (strict request/response validation schemas)
  * Uvicorn (high-concurrency ASGI server)
  * SQLite & SQLAlchemy (zero-configuration local relational persistence)
* **NLP, Speech & Deep Learning**:
  * **Hugging Face Transformers & PyTorch**: Fine-tuned `roberta-base` for dialogue act sequence classification.
  * **Faster-Whisper (`tiny.en`)**: Local speech-to-text transcription via CTranslate2 with INT8 quantization.
  * **spaCy (`en_core_web_sm`)**: Sentence boundary detection, tokenization, POS tagging, and syntactic dependency parsing.
  * **scikit-learn**: TF-IDF vectorization, Logistic Regression, Linear SVM, Random Forest, and Naive Bayes baselines.
  * **dateparser**: Relative and explicit temporal entity resolution.

---

## 3. End-to-End Extraction Pipeline

The extraction engine is implemented across independent, modular services inside `backend/app/services/`:

```
Raw Meeting Audio (.mp3, .wav) OR Transcript (.txt, .vtt)
   │
   ▼
[Stage A: Preprocessing & Parsing]      ──► Strip WebVTT cues, segment sentences, attribute speakers
   │
   ▼
[Stage B: Dual-Engine Classification]   ──► RoBERTa Neural Transformer OR Balanced Logistic Regression
   │
   ▼
[Consensus & Confidence Calibration]    ──► Temperature scaling (T=0.55) + heuristic consensus boost
   │
   ▼
[Stage C: Task Description Extraction]  ──► spaCy dependency parse tree extracts head verb predicate
   │
   ▼
[Stage D: Assignee Resolution]          ──► Vocatives ([Name], please) + First-person turns + Pronouns
   │
   ▼
[Stage E: Deadline Normalization]       ──► Maps relative phrases to ISO 8601 dates (YYYY-MM-DD)
   │
   ▼
[Stage F: Review Flagging & Dedup]      ──► Flags ambiguous cases (needs_review) + Levenshtein dedup
   │
   ▼
Structured Task Response (Stored in SQLite & Displayed in React UI)
```

### Module Breakdown:

1. **Stage A: Preprocessing (`preprocessing.py`)**:
   * Strips WebVTT headers (`WEBVTT`), cue timestamps (`00:00:01.000 --> 00:00:04.000`), and voice tags (`<v Speaker>`).
   * Parses speaker-labeled dialogues across formats (`Speaker: Text`, `[Speaker]: Text`, `(Speaker): Text`).
   * Splits multi-sentence turns into granular, indexed utterances while preserving speaker attribution.

2. **Stage B: Action-Item Classification (`action_classifier.py` / `transformer_extractor.py`)**:
   * Evaluates each utterance into 4 distinct dialogue acts:
     * `ACTION_ITEM`: Direct commitments or requests requiring future execution.
     * `SUGGESTION`: Proposals or ideas without definitive commitment (*"What if we try...", "Maybe we could..."*).
     * `COMPLETED_ACTION`: Explicitly finished tasks (*"The bug was fixed yesterday", "I deployed the update"*).
     * `NON_ACTIONABLE`: General discussion, status queries, observations, and procedural chitchat.
   * Offers dual engines: **V2 RoBERTa-base Neural Transformer** (maximum recall) and **V1 Balanced Logistic Regression** (sub-millisecond CPU speed).

3. **Stage C: Task Description Extraction (`task_extractor.py`)**:
   * Analyzes the spaCy dependency tree of actionable sentences.
   * Strips conversational filler prefixes (*"I think I will...", "Rahul, please..."*).
   * Normalizes imperative verb phrases into clean, executive task titles (*"Finish the backend API"* instead of *"I'll finish the backend API by Wednesday"*).

4. **Stage D: Assignee Resolution (`assignee_extractor.py`)**:
   * **First-Person Commitments**: Maps *"I will..."* / *"I'll take care of..."* directly to the active utterance speaker.
   * **Directed Vocatives**: Resolves syntax patterns like `[Name], please [verb]...` or `Could [Name] review...` to the target individual.
   * **Indefinite Pronouns**: Identifies unresolved obligations (*"We need to..."*, *"Someone should..."*) and explicitly tags them as `Unassigned`, setting `needs_review = True`.

5. **Stage E: Temporal & Deadline Extraction (`deadline_extractor.py`)**:
   * Recognizes relative temporal expressions (*"by tomorrow"*, *"before Friday"*, *"next week"*, *"within three days"*).
   * Uses the user-specified **Meeting Reference Date** to deterministically resolve relative dates into standardized ISO 8601 calendar strings (`YYYY-MM-DD`).
   * Flags ambiguous deadlines (*"before the next sync"*) for human confirmation.

6. **Stage F: Calibration & Deduplication (`validation.py`)**:
   * Multi-factor confidence calibration combining neural softmax probabilities with heuristic penalty deductions.
   * Removes duplicate tasks generated across repeated conversational turns using fuzzy string similarity.

---

## 4. Benchmark Dataset & Research Methodology

The system is trained and benchmarked on the gold-standard research corpus in multi-party meeting dialogue analysis:

### The AMI Meeting Corpus & Enterprise Benchmark
* **Primary Source**: The **AMI Meeting Corpus (Release 1.6.2)** (Carletta et al., University of Edinburgh / IDIAP) combined with multi-domain enterprise engineering syncs.
* **Total Meetings**: **86 multi-party meetings**
* **Total Utterances**: **15,179 fully annotated utterances**
* **Zero-Leakage Partitioning**: Partitioned **strictly by whole meeting IDs (Group Split)** so that the test set contains 0% overlapping meetings, topics, or speaker voices with the training set.

| Split | Meetings | Total Utterances | Purpose |
| :--- | :---: | :---: | :--- |
| **Train Set** | 64 meetings | **11,385** | Vocabulary building, feature extraction, and model fine-tuning |
| **Held-Out Test Set** | 22 meetings | **3,794** | Unbiased generalization evaluation on unseen meetings |

#### Class Distribution:
* **`NON_ACTIONABLE`**: 10,096 utterances (7,574 train / 2,522 test) — 66.5% of raw dialogue
* **`SUGGESTION`**: 3,989 utterances (2,968 train / 1,021 test) — 26.3% of raw dialogue
* **`ACTION_ITEM`**: 945 utterances (728 train / 217 test) — 6.2% of raw dialogue
* **`COMPLETED_ACTION`**: 149 utterances (115 train / 34 test) — 1.0% of raw dialogue

---

## 5. Machine Learning Failure Analysis: Solving Majority-Class Collapse

During initial V2 transformer fine-tuning on raw dialogue data, the model encountered a critical failure mode:

### The Problem:
Because **66.5%** of raw meeting dialogue consists of non-actionable chatter, an unconstrained Cross-Entropy loss function rewards the network for predicting Class 0 (`NON_ACTIONABLE`) for 100% of samples:
$$\frac{2,522 \text{ Non-Actionable}}{3,794 \text{ Total Test}} = 0.6647 \quad (66.47\% \text{ baseline accuracy})$$
The model achieved 66.47% accuracy while scoring **0.0000 Precision and Recall on Action Items**, completely collapsing into the majority class.

### The Solution: Stratified High-Signal Balancing
We designed a stratified resampling pipeline in the training notebook:
1. **Action Items**: Duplicated ($728 \to 1,456$ samples) to give them high signal density.
2. **Completed Actions**: Duplicated ($115 \to 345$ samples).
3. **Suggestions**: Sampled 1,200 utterances.
4. **Non-Actionable Chatter**: Downsampled from 7,574 to 1,600 utterances.
5. **Result**: Action items increased from **6.2% to 31.6%** of training batches, forcing the transformer to actively learn action lexical markers (`"I will"`, `"can you"`, `"deadline"`, `"by Friday"`).

---

## 6. Multi-Model Benchmark Results

Evaluated on all **3,794 unseen held-out AMI test utterances**:

| Model Architecture | Accuracy | Action-Item F1 | Action Recall | Macro F1 | Weighted F1 | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V2 Neural Transformer (RoBERTa-base)** ⭐ | **76.88%** | **54.40%** | **78.34%** | **0.6821** | **0.7765** | **18.6 ms** |
| **V1 Best ML (Logistic Regression Balanced)** | 73.38% | 46.59% | 56.68% | 0.6547 | 0.7395 | **0.015 ms** |
| **Linear SVM (`LinearSVC`)** | 74.01% | 44.50% | 40.55% | 0.6401 | 0.7369 | 0.014 ms |
| **Random Forest (100 Trees)** | 74.14% | 40.77% | 34.56% | 0.6311 | 0.7384 | 0.047 ms |
| **Multinomial Naive Bayes** | 73.01% | 26.80% | 18.89% | 0.4137 | 0.7017 | 0.015 ms |
| **Rule-Based Baseline (spaCy + Patterns)** | 64.34% | 20.64% | 62.40% | 0.4243 | 0.5637 | 5.426 ms |

### RoBERTa Test Set Confusion Matrix (3,794 Utterances):
```
                 Pred Non-Action   Pred Suggestion   Pred Action   Pred Completed
True Non-Action       2024               310             176             12
True Suggestion        257               697              62              5
True Action             27                19             170              1
True Completed           8                 0               0             26
```
* **True Action Items Identified**: **170 out of 217** (78.34% recall).
* **True Completed Actions Identified**: **26 out of 34** (76.47% recall).

---

## 7. NER & Slot-Filling Evaluation

* **Assignee Extraction (PERSON Slot Filling)**:
  * Raw spaCy `PERSON` recall on meeting speech: **4.22%**
  * Grammar-guided vocative and pronoun resolution pipeline: **33.73%** (**8.0x improvement**)
* **Deadline Extraction (TIMEX Slot Filling)**:
  * Temporal phrase detection recall: **25.76%**
  * ISO 8601 calendar normalization rate: **22.73%**

---

## 8. Verification & Quality Assurance

* **Pytest Test Suite**: **32 passing tests (100% pass rate)** in `backend/tests/`:
  * `test_api.py`: Endpoint health, extraction requests, file upload handling, CRUD operations, CSV/JSON exports.
  * `test_nlp_pipeline.py`: Linguistic test cases covering first-person commitments, directed imperatives, modal questions, collective obligations, negation filtering, and past completed action differentiation.
* **Production Build Verification**: Frontend compiles via Vite in **873 ms** with zero errors or warnings.
* **Zero Cloud Dependencies**: 100% verified offline local execution.
