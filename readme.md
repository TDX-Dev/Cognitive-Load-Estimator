# Cognitive Load Estimator

A machine learning–based system that estimates cognitive load in text using linguistic features across multiple levels:

- Morphological
- Syntactic
- Semantic
- Optional transformer-based surprisal

It produces a sentence-level heatmap and an overall cognitive load score on a 0–10 scale.

Built with:

- spaCy
- PyTorch
- scikit-learn
- HuggingFace Transformers
- Streamlit



## Overview

This project moves beyond traditional readability metrics by modeling structural and semantic complexity. Instead of relying only on sentence length or word length, it incorporates:

- Lexical density
- Type-token ratio
- Dependency structure
- Clause embedding
- Semantic ambiguity
- Ranking-based learning

The system is trained on aligned Elementary, Intermediate, and Advanced texts from the OneStopEnglish corpus.



## Features

### Sentence-Level Cognitive Load Scoring

Each sentence is scored from 0 to 10 using:

- Lexical density
- Sentence length
- Average word length
- Type-token ratio
- Average dependency distance
- Parse tree depth
- Clause count
- WordNet polysemy score

### Heatmap Visualization

- Sentences are color-coded based on difficulty.
- The overall score is computed as a weighted average based on sentence length.
- Output can be rendered directly in Streamlit or exported as HTML.

### Ranking-Based Learning

Instead of predicting absolute difficulty only, the ranking model learns ordered relationships:




## How It Works

### Feature Extraction

For each sentence:

Morphological:
- Lexical density
- Type-token ratio
- Average word length
- Sentence length

Syntactic:
- Average dependency distance
- Tree depth
- Clause count

Semantic:
- WordNet polysemy score

Optional:
- BERT-based surprisal score



## Model Types

### 1. Regression Model

File: `models/difficulty.pkl`

- Trained using RandomForestRegressor
- Predicts numeric difficulty directly

### 2. Ranking Model

Files:
- `models/ranking_model.pt`
- `models/ranking_scaler.pkl`

- Trained using MarginRankingLoss
- Learns relative ordering
- Outputs normalized 0–10 scores

The ranking model is generally more stable for relative cognitive load comparisons.



