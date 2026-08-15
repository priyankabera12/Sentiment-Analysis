# Comparative Sentiment Analysis, Advanced RAG, and Adversarial Fake Data Detection System

An end-to-end Natural Language Processing (NLP) framework featuring sentiment analysis benchmarking, an Advanced Retrieval-Augmented Generation (RAG) system with RLHF alignment, and an adversarial fake news detection pipeline.

---

## 📌 Project Overview

Developed by **Priyanka Bera** (Feb 2026 – Jul 2026) within the Department of Natural Language Processing & Machine Learning, this project systematically addresses data scarcity in sentiment classification, optimizes context accuracy in retrieval systems, and detects adversarial misinformation at scale.

**##  Background & Core ObjectivesData Scarcity vs. Fine-Tuning Efficiency: Explored how zero-shot, few-shot, and supervised fine-tuning scale across varying dataset sizes. The study highlights the pitfalls of extreme data scarcity (where few-shot models suffer severe overfitting) and establishes empirical data thresholds required to achieve high-accuracy sentiment classification on noisy social media text. 

**##  Next-Generation Retrieval Architectures: Addressed the structural limitations of Naive RAG baselines—such as context fragmenting and lost document boundaries—by implementing Advanced RAG techniques (sentence-window retrieval and auto-merging). The system was further aligned using Reinforcement Learning from Human Feedback (RLHF) to evaluate optimization stability and reward function trade-offs. 

**## Adversarial Fake News Detection: Designed an end-to-end pipeline to analyze and classify deceptive content within a 23,481-article corpus (fake.csv). By combining VADER sentiment profiling (which uncovered extreme bimodal polarization in fake news), FAISS-backed dense semantic retrieval (all-MiniLM-L6-v2), and TF-IDF linear classification (PassiveAggressive Classifier), the framework achieves high-precision detection on unseen holdout data. 
____________________________________________________________________________________________________________________________________________________________________

## 🚀 Key Modules & Performance Benchmarks

### 1. Sentiment Analysis Optimization

* Benchmarked zero-shot **BART** (`facebook/bart-large-mnli`) against minimal-sample few-shot **DistilBERT** models.


* **Zero-Shot Baseline:** BART achieved ~35% accuracy (F1-score: 0.38).


* **Few-Shot Overfitting:** DistilBERT trained on 16 samples yielded 10% accuracy due to severe data scarcity.


* **Supervised Fine-Tuning:** Expanded training data on an 80-20 train-test split, elevating classification accuracy to **87%** with 500 samples and **91%** with 2,000 samples.

---

### 2. Advanced RAG System & RLHF

Replaced Naive RAG baselines with **sentence-window retrieval** and **auto-merging** architectures in Python.

| Metric | Naive RAG Baseline | Advanced RAG | Improvement |
| --- | --- | --- | --- |
| **Context Relevance** | 65% | **82%** | +17%

 |
| **Groundedness** | 70% | **88%** | +18%

 |
| **Answer Relevance** | 72% | **90%** | +18%|

* **RLHF Alignment:** Integrated Reinforcement Learning from Human Feedback, reaching **~89% accuracy** while evaluating trade-offs from binary reward constraints.



---

### 3. Adversarial Fake Data Detection System

Constructed an end-to-end detection pipeline using 23,481 clean news articles (`fake.csv`).

* Combined dense vector retrieval using **FAISS** indexed with `all-MiniLM-L6-v2` embeddings alongside a **TF-IDF PassiveAggressive Classifier**.


* **Test Performance:** Achieved **96.51% classification accuracy** (Weighted Precision: 0.965, Recall: 0.965, F1-Score: 0.965) on an unseen holdout set of 4,697 articles.



---

## 📊 Exploratory Data Analysis (EDA)

### Topic & Term Frequency Analysis

The dataset spans multiple political and news categories, heavily concentrated in general news and politics.

Lexical frequency profiling highlights key political entities driving TF-IDF feature extraction:

### VADER Sentiment Profiling

Sentiment analysis revealed strong **bimodal polarization** in fake news articles, with compound scores clustering heavily at extreme negative (`-1.0`) and extreme positive (`+1.0`) values.
<img width="1007" height="482" alt="image" src="https://github.com/user-attachments/assets/27932610-b62b-4910-8d78-218dd2e8404f" />
<img width="990" height="581" alt="image" src="https://github.com/user-attachments/assets/f9ed0dab-be47-43d4-b210-c607a5098321" />
<img width="997" height="572" alt="image" src="https://github.com/user-attachments/assets/3d88e995-0a7f-4772-84ce-9ce3c78a8530" />

---

## 🛠️ Tech Stack & Dependencies

* **Core Language:** Python


* **Transformers & Models:** `facebook/bart-large-mnli`, `distilbert-base-uncased`, `sentence-transformers` (`all-MiniLM-L6-v2`)


* **Vector Search & ML:** FAISS, Scikit-learn (TF-IDF, PassiveAggressive Classifier), VADER
