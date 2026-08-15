Comparative Sentiment Analysis, Advanced RAG, and Adversarial Fake Data Detection SystemAn end-to-end Natural Language Processing (NLP) framework featuring sentiment analysis benchmarking, an Advanced Retrieval-Augmented Generation (RAG) system with RLHF alignment, and an adversarial fake news detection pipeline. 

📌 Project OverviewDeveloped by Priyanka Bera (Feb 2026 – Jul 2026) within the Department of Natural Language Processing & Machine Learning, this project systematically addresses data scarcity in sentiment classification, optimizes context accuracy in retrieval systems, and detects adversarial misinformation at scale.  🚀 Key Modules & Performance Benchmarks1. Sentiment Analysis OptimizationBenchmarked zero-shot BART (facebook/bart-large-mnli) against minimal-sample few-shot DistilBERT models.  Zero-Shot Baseline: BART achieved ~35% accuracy (F1-score: 0.38).  Few-Shot Overfitting: DistilBERT trained on 16 samples yielded 10% accuracy due to severe data scarcity.  Supervised Fine-Tuning: Expanded training data on an 80-20 train-test split, elevating classification accuracy to 87% with 500 samples and 91% with 2,000 samples.  Pythonfrom transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)
2. Advanced RAG System & RLHFReplaced Naive RAG baselines with sentence-window retrieval and auto-merging architectures in Python.  MetricNaive RAG BaselineAdvanced RAGImprovementContext Relevance65%82%+17%  Groundedness70%88%+18%  Answer Relevance72%90%+18%  RLHF Alignment: Integrated Reinforcement Learning from Human Feedback, reaching ~89% accuracy while evaluating trade-offs from binary reward constraints.  Pythonimport faiss
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = embedder.encode(corpus_chunks, show_progress_bar=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
3. Adversarial Fake Data Detection SystemConstructed an end-to-end detection pipeline using 23,481 clean news articles (fake.csv).  Combined dense vector retrieval using FAISS indexed with all-MiniLM-L6-v2 embeddings alongside a TF-IDF PassiveAggressive Classifier.  Test Performance: Achieved 96.51% classification accuracy (Weighted Precision: 0.965, Recall: 0.965, F1-Score: 0.965) on an unseen holdout set of 4,697 articles.  
📊 Exploratory Data Analysis (EDA)Topic & Term Frequency AnalysisThe dataset spans multiple political and news categories, heavily concentrated in general news and politics.  Lexical frequency profiling highlights key political entities driving TF-IDF feature extraction:VADER Sentiment ProfilingSentiment analysis revealed strong bimodal polarization in fake news articles, with compound scores clustering heavily at extreme negative (-1.0) and extreme positive (+1.0) values.  🛠️ Tech Stack & DependenciesCore Language: Python  Transformers & Models: facebook/bart-large-mnli, distilbert-base-uncased, sentence-transformers (all-MiniLM-L6-v2)  Vector Search & ML: FAISS, Scikit-learn (TF-IDF, PassiveAggressive Classifier), VADER  

