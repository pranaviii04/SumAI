# SummAI - ML-Based Business Text Summarization

## 📜 Project Overview

SummAI is a state-of-the-art text summarization application designed to distill complex business documents into concise, actionable summaries. It leverages the power of **Transformer models (T5)** for high-quality abstractive summarization, while also providing a robust **Extractive fallback** using NLTK for speed and reliability.

Whether you need to summarize long reports, news articles, or technical documents, SummAI provides a unified interface via a modern **Web UI** and a comprehensive **REST API**.

---

## ✨ Key Features

-   **Dual Summarization Engine**:
    -   **Abstractive**: Uses fine-tuned **T5-small** models to generate human-like summaries.
    -   **Extractive**: Uses statistical NLTK text analysis for fast, reliable key-sentence extraction.
-   **Modern Web Interface**: A beautiful Neo-Pop / Brutalist styled UI with dark mode support.
-   **REST API**: Production-ready FastAPI endpoints for integration.
-   **Comprehensive Metrics**: Evaluate models using **ROUGE-1/2/L**, **Token Accuracy**, **F1-Scores**, and **Perplexity**.
-   **Customizable Training**: Easily fine-tune models on your own CSV datasets.
-   **PDF/TXT Support**: Drag-and-drop file upload for automatic text extraction.

---

## 📦 Project Structure

```
SummAI/
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI Application Entry Point
│   ├── config.py           # Hyperparameters & Configuration
│   ├── core/               # Core Modules
│   │   ├── summarizer.py   # NLTK Extractive Logic
│   │   ├── data_loader.py  # Data Loading (HuggingFace + CSV)
│   │   └── metrics.py      # Evaluation Metrics (ROUGE, F1, etc.)
│   └── ml/                 # Machine Learning Scripts
│       ├── train.py        # Model Training Script
│       ├── evaluate.py     # Evaluation Script
│       └── predict.py      # Inference/Prediction Script
├── scripts/
│   └── setup_nltk.py       # Setup Utility
├── static/                 # Frontend Assets (HTML/CSS/JS)
├── model/                  # Trained Model Artifacts
├── requirements.txt        # Dependencies
├── README.md               # Project Documentation
└── PROJECT_REPORT.md       # Project Report
```

---

## 🚀 Quick Start Guide

### 1. Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Web Application

Launch the server immediately with the default NLTK summarizer (no training required):

```bash
uvicorn app.main:app --reload
```

-   **Web UI**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000)
-   **API Docs**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧠 Machine Learning Workflow

To unlock the full power of Abstractive Summarization, follow these steps to train your own model.

### 1. Train the Model

Fine-tune the T5 model on the CNN/DailyMail dataset (or your custom data):

```bash
python -m app.ml.train
```

*Training typically takes 30-60 minutes on CPU. The model will be saved to `./model/best_model`.*

### 2. Evaluate Performance

Run a full evaluation on the test set to get ROUGE and F1 scores:

```bash
python -m app.ml.evaluate
```

### 3. Generate Predictions

Test the trained model via the command line:

```bash
# Interactive Mode
python -m app.ml.predict

# Direct Text
python -m app.ml.predict --text "Your long text goes here..."

# From File
python -m app.ml.predict --file path/to/document.txt
```

---

## 🔧 Configuration

All settings are centralized in `app/config.py`. You can customize:

-   **Model Architecture**: `MODEL_NAME = "t5-small"` (or `facebook/bart-base`)
-   **Training Params**: `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`
-   **Dataset**: `DATASET_NAME = "cnn_dailymail"` or use `CSV_DATA_PATH`
-   **Generation**: `NUM_BEAMS`, `MAX_TARGET_LENGTH`

---

## 📊 Evaluation Metrics Explained

We use industry-standard metrics to ensure quality:

| Metric | Description |
| :--- | :--- |
| **ROUGE-1** | Initial overlap of words (unigrams). Good for content coverage. |
| **ROUGE-2** | Overlap of bigrams. Measures phrasing match. |
| **ROUGE-L** | Longest Common Subsequence. Measures sentence structure similarity. |
| **Perplexity** | Model confidence (lower is better). |
| **Token F1** | Token-level classification accuracy. |

---

## 🎯 API Endpoints

### `POST /summarize`

Generates a summary for the provided text.

**Request:**
```json
{
  "text": "The global economy is facing...",
  "use_ml": true
}
```

**Response:**
```json
{
  "summary": "The global economy is facing challenges...",
  "model_used": "ML (Transformer)",
  "original_length": 500,
  "summary_length": 150
}
```

### `POST /extract-text`
Upload a PDF or TXT file to extract its raw text content.

### `GET /evaluate`
Returns the latest evaluation metrics from the trained model.

---

## 📝 License

This project is open-source and available under the MIT License.
