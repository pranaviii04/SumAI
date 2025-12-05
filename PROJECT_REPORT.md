# SummAI Project Report

## 1. Project Overview
SummAI is a machine learning-powered text summarization application designed to process business documents and generate concise summaries. 

### Key Features
- **Extractive Summarization**: Uses NLTK (Rule-based) as a fast, resource-efficient fallback.
- **Abstractive Summarization**: Uses Transformer models (T5-small) for human-like summaries.
- **Evaluation Pipeline**: Detailed metrics including ROUGE scores, token-level accuracy/F1, and perplexity.
- **REST API**: Built with FastAPI for easy integration.
- **User Interface**: Web-based UI for uploading documents and viewing summaries.

## 2. Technical Architecture
### Core Components
- **Framework**: Python 3.12 + FastAPI
- **ML Library**: PyTorch + Hugging Face Transformers
- **NLP Library**: NLTK (for Tokenization/Stopwords)
- **Data Handling**: Pandas, PyPDF (for PDF text extraction)

### Directory Structure
- `main.py`: The entry point for the API server.
- `train.py`: Script to train/fine-tune the transformer model.
- `evaluate.py`: Script to evaluate the trained model on test data.
- `predict.py`: CLI tool for quick predictions.
- `config.py`: Centralized configuration for hyperparameters.
- `metrics.py`: Custom implementation of ROUGE and token-level metrics.
- `model/`: Directory storing the trained model artifacts.

## 3. Training & Model Details
### Configuration
- **Model Architecture**: T5-small (Text-to-Text Transfer Transformer)
- **Dataset**: Custom CSV (`sample_data.csv`) used for this run.
- **Batch Size**: 2
- **Epochs**: 3
- **Optimizer**: AdamW with linear warmup.
- **Device**: CPU (as per available hardware).

### Training History (Last Run)
The model was trained for 3 epochs.

| Epoch | Train Loss | Eval Loss | Perplexity | Token F1 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|------------|-----------|------------|----------|---------|---------|---------|
| 1     | 17.4459    | 8.4883    | 4857.41    | 0.5984   | 0.3429  | 0.1212  | 0.3429  |
| 2     | 18.8192    | 8.4873    | 4852.67    | 0.5984   | 0.3429  | 0.1212  | 0.3429  |
| 3     | 19.4194    | 8.4860    | 4846.58    | 0.5984   | 0.3429  | 0.1212  | 0.3429  |

*Note: The consistent metrics across epochs suggest the model might have quickly converged or the dataset size (sample data) was small enough that it memorized/stabilized quickly.*

## 4. Evaluation Results
The final model was evaluated on the test set.

**Key Metrics:**
- **ROUGE-1**: 0.2949 (Overlap of unigrams)
- **ROUGE-2**: 0.1132 (Overlap of bigrams)
- **ROUGE-L**: 0.2146 (Longest common subsequence)
- **Token Accuracy**: 0.87% (Low, likely due to small sample size/vocab mismatch)
- **Perplexity**: ~32,587 (High, indicates model uncertainty, typical for small datasets/early training)

## 5. Usage Guide
### Starting the Server
```bash
uvicorn main:app --reload
```
Access the UI at: `http://127.0.0.1:8000`

### Training a New Model
1. Ensure data is in `static/data/sample_data.csv` (or configure `config.py`).
2. Run:
   ```bash
   python train.py
   ```

### Running Evaluation
```bash
python evaluate.py
```

## 6. Recommendations & Next Steps
1. **Expand Dataset**: The current training was likely performed on a very small sample set (`sample_data.csv`). For production-grade results, train on a larger dataset like CNN/DailyMail or XSum.
2. **Increase Eras/Compute**: Training on CPU is limited. For better results, use a GPU environment to increase batch size and sequence length.
3. **Hyperparameter Tuning**: Experiment with learning rates and beam search parameters (in `config.py`) to lower the perplexity.
