# Named Entity Recognition (NER) Project 🔍

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📜 Project Description

This project focuses on building and evaluating a **Named Entity Recognition (NER)** model using the Hugging Face `transformers` library. NER is a core Natural Language Processing (NLP) technique that identifies and classifies named entities in text into predefined categories.

The goal of this notebook is to demonstrate a **full End-to-End MLOps pipeline**, covering data loading, complex preprocessing (token alignment), model fine-tuning, evaluation, and interactive deployment using **Gradio**.

## ✨ Features

* **Dataset Loading**: Utilizes the benchmark **CoNLL-2003** dataset.
* **Pre-trained Model**: Fine-tunes `distilbert-base-uncased` for token classification.
* **Smart Preprocessing**: Implements a custom **Token-Label Alignment** strategy to handle subword tokenization (a critical step for BERT-based models).
* **Efficient Training**: Uses the Hugging Face `Trainer` API with optimized hyperparameters.
* **Metrics**: Calculates Precision, Recall, and F1-score via `classification_report`.
* **Model Persistence**: Saves the fine-tuned model and tokenizer for reuse.
* **Interactive Demo**: Deploys a web-based UI using **Gradio** for real-time testing.

## 🛠️ Technologies Used

* **Python**
* **Hugging Face Transformers**: Model, Tokenizer, Trainer API
* **Datasets**: Data loading and management
* **PyTorch**: Deep Learning backend
* **Scikit-learn**: Evaluation metrics
* **Gradio**: Interactive Web UI
* **Matplotlib & Pandas**: Visualization

## 📂 Dataset

The project uses the **CoNLL-2003** dataset, a standard benchmark containing news wire text annotated with four entity types:

| Tag | Entity Type | Description |
| :--- | :--- | :--- |
| **PER** | Person | Names of people (e.g., "Elon Musk") |
| **ORG** | Organization | Companies, institutions (e.g., "Google", "UN") |
| **LOC** | Location | Cities, countries, rivers (e.g., "Paris", "Mt. Everest") |
| **MISC** | Miscellaneous | Nationalities, events, products |

## 🚀 How to Run

### 1. Installation
To run this project locally or in Google Colab, first install the required dependencies:

```bash
pip install datasets transformers tensorboard gradio scikit-learn
