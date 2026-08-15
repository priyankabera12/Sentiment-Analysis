# Sentiment-Analysis
An end-to-end Natural Language Processing (NLP) framework implementing sentiment analysis benchmarking, an Advanced Retrieval-Augmented Generation (RAG) system, and an adversarial fake news detection pipeline.


# Comparative Sentiment Analysis, Advanced RAG, and Adversarial Fake Data Detection System

An end-to-end Natural Language Processing (NLP) framework featuring sentiment analysis benchmarking, an Advanced Retrieval-Augmented Generation (RAG) system with RLHF alignment, and an adversarial fake news detection pipeline.

## 📌 Project Overview

Developed by **Priyanka Bera** (Feb 2026 – Jul 2026), this project systematically addresses data scarcity in sentiment classification, improves context accuracy in retrieval systems, and detects adversarial misinformation at scale.

## 🚀 Key Modules & Architecture

### 1. Sentiment Analysis Optimization

* Benchmarked zero-shot **BART** (`facebook/bart-large-mnli`) against minimal-sample few-shot **DistilBERT** models.


* Applied supervised fine-tuning to elevate classification accuracy from a ~35% baseline up to **91%** across 2,000 samples.



### 2. Advanced RAG System & RLHF

* Replaced Naive RAG baselines with **sentence-window retrieval** and **auto-merging**.


* Achieved significant performance gains across standard RAG metrics:
* **Context Relevance:** 82% (+17% over baseline)


* **Groundedness:** 88% (+18% over baseline)


* **Answer Relevance:** 90% (+18% over baseline)


* Integrated Reinforcement Learning from Human Feedback (**RLHF**), reaching ~89% alignment accuracy.



### 3. Adversarial Fake Data Detection

* Analyzed 23,481 news articles using **VADER** sentiment analysis to reveal bimodal polarization patterns in fake news.


* Implemented dense retrieval with **FAISS** vector search (`all-MiniLM-L6-v2` embeddings) paired with a **TF-IDF PassiveAggressive Classifier**.


* Final Pipeline Accuracy: **96.51%** (F1-score: 0.965) on a 4,697-article holdout test set.
<img width="1007" height="482" alt="image" src="https://github.com/user-attachments/assets/0c6492db-7b99-412a-9ff4-41e2fb87a2e0" />

<img width="990" height="581" alt="image" src="https://github.com/user-attachments/assets/935b1338-2c79-48d9-aab2-07f6830d6465" />

<img width="997" height="572" alt="image" src="https://github.com/user-attachments/assets/a3e3661d-098b-4da4-870e-339b0596826c" />

## 🛠️ Tech Stack & Dependencies

* **NLP & Embeddings:** Hugging Face Transformers (`bart-large-mnli`, `distilbert-base-uncased`), `sentence-transformers` (`all-MiniLM-L6-v2`), VADER


* **Vector Search & ML:** FAISS, Scikit-learn (TF-IDF, PassiveAggressive Classifier)


* **Core Language:** Python

## 🛠️ Tech Stack & Dependencies

* **NLP & Embeddings:** Hugging Face Transformers (`bart-large-mnli`, `distilbert-base-uncased`), `sentence-transformers` (`all-MiniLM-L6-v2`), VADER


* **Vector Search & ML:** FAISS, Scikit-learn (TF-IDF, PassiveAggressive Classifier)


* **Core Language:** Python

## 🛠️ Tech Stack & Dependencies

* **NLP & Embeddings:** Hugging Face Transformers (`bart-large-mnli`, `distilbert-base-uncased`), `sentence-transformers` (`all-MiniLM-L6-v2`), VADER


* **Vector Search & ML:** FAISS, Scikit-learn (TF-IDF, PassiveAggressive Classifier)


* **Core Language:** Python
