"""
SummAI Evaluation Metrics
Functions to compute accuracy, precision, recall, F1, and ROUGE scores
"""

import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Dict, Tuple
import torch


def compute_token_metrics(predictions: List[List[int]], references: List[List[int]]) -> Dict[str, float]:
    """
    Compute token-level classification metrics
    
    Treats summarization as a token prediction task where we compare
    predicted token IDs with reference token IDs.
    
    Args:
        predictions: List of predicted token ID sequences
        references: List of reference token ID sequences
    
    Returns:
        Dictionary with accuracy, precision, recall, and F1 scores
    """
    # Flatten all sequences for token-level comparison
    all_preds = []
    all_refs = []
    
    for pred, ref in zip(predictions, references):
        # Pad or truncate to same length
        max_len = max(len(pred), len(ref))
        pred_padded = pred + [0] * (max_len - len(pred))
        ref_padded = ref + [0] * (max_len - len(ref))
        
        all_preds.extend(pred_padded)
        all_refs.extend(ref_padded)
    
    # Compute metrics
    accuracy = accuracy_score(all_refs, all_preds)
    
    # Use weighted average for multi-class precision/recall/f1
    precision = precision_score(all_refs, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_refs, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_refs, all_preds, average='weighted', zero_division=0)
    
    return {
        "token_accuracy": accuracy,
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1
    }


def compute_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE scores for summarization
    
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
    the overlap between generated and reference summaries.
    
    Args:
        predictions: List of generated summaries
        references: List of reference summaries
    
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        "rouge1": np.mean(rouge1_scores),
        "rouge2": np.mean(rouge2_scores),
        "rougeL": np.mean(rougeL_scores)
    }


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from loss
    
    Perplexity measures how well the model predicts the sample.
    Lower is better.
    
    Args:
        loss: Cross-entropy loss
    
    Returns:
        Perplexity score
    """
    return np.exp(loss)


def evaluate_model(
    model,
    tokenizer,
    eval_dataloader,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Comprehensive evaluation of the summarization model
    
    Args:
        model: Trained summarization model
        tokenizer: Tokenizer
        eval_dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
    
    Returns:
        Dictionary with all evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_references = []
    all_pred_texts = []
    all_ref_texts = []
    total_loss = 0
    
    print("[Evaluation] Evaluating model...")
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Compute loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()
            
            # Generate predictions
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode predictions and references
            pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Store for metrics
            all_pred_texts.extend(pred_texts)
            all_ref_texts.extend(ref_texts)
            
            # Store token IDs for token-level metrics
            all_predictions.extend(generated_ids.cpu().tolist())
            all_references.extend(labels.cpu().tolist())
    
    # Compute average loss
    avg_loss = total_loss / len(eval_dataloader)
    
    # Compute all metrics
    token_metrics = compute_token_metrics(all_predictions, all_references)
    rouge_metrics = compute_rouge_scores(all_pred_texts, all_ref_texts)
    perplexity = compute_perplexity(avg_loss)
    
    # Combine all metrics
    metrics = {
        "eval_loss": avg_loss,
        "perplexity": perplexity,
        **token_metrics,
        **rouge_metrics
    }
    
    # Print results
    print("\n" + "="*60)
    print("[Results] EVALUATION RESULTS")
    print("="*60)
    print(f"Loss: {metrics['eval_loss']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.4f}")
    print(f"\nToken-Level Metrics:")
    print(f"  Accuracy:  {metrics['token_accuracy']:.4f}")
    print(f"  Precision: {metrics['token_precision']:.4f}")
    print(f"  Recall:    {metrics['token_recall']:.4f}")
    print(f"  F1-Score:  {metrics['token_f1']:.4f}")
    print(f"\nROUGE Scores:")
    print(f"  ROUGE-1: {metrics['rouge1']:.4f}")
    print(f"  ROUGE-2: {metrics['rouge2']:.4f}")
    print(f"  ROUGE-L: {metrics['rougeL']:.4f}")
    print("="*60 + "\n")
    
    return metrics


def print_sample_predictions(
    predictions: List[str],
    references: List[str],
    inputs: List[str],
    num_samples: int = 3
):
    """
    Print sample predictions for qualitative evaluation
    
    Args:
        predictions: Generated summaries
        references: Reference summaries
        inputs: Input texts
        num_samples: Number of samples to print
    """
    print("\n" + "="*60)
    print("[Samples] SAMPLE PREDICTIONS")
    print("="*60)
    
    for i in range(min(num_samples, len(predictions))):
        print(f"\n--- Sample {i+1} ---")
        print(f"\nINPUT:\n{inputs[i][:200]}...")
        print(f"\nREFERENCE SUMMARY:\n{references[i]}")
        print(f"\nGENERATED SUMMARY:\n{predictions[i]}")
        print("-" * 60)
