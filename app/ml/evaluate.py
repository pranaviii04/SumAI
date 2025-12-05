"""
SummAI Evaluation Script
Evaluate a trained summarization model

USAGE:
    python evaluate.py

This will:
1. Load the trained model from ./model/best_model
2. Evaluate on the test dataset
3. Display all metrics (Accuracy, Precision, Recall, F1, ROUGE)
4. Show sample predictions
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import os

from app.config import Config
from app.core.data_loader import prepare_datasets
from app.core.metrics import evaluate_model, print_sample_predictions


def evaluate_saved_model(
    model_path: str = Config.MODEL_SAVE_PATH,
    use_huggingface: bool = True,
    csv_path: str = None,
    num_samples: int = 5
):
    """
    Evaluate a saved model
    
    Args:
        model_path: Path to saved model
        use_huggingface: Whether to use HuggingFace dataset
        csv_path: Path to custom CSV (if use_huggingface=False)
        num_samples: Number of sample predictions to display
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please train a model first using: python train.py"
        )
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load model and tokenizer
    print(f"📦 Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load training config if available
    config_path = os.path.join(model_path, "training_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            training_config = json.load(f)
        print(f"\n📋 Model Training Config:")
        print(f"  Original Model: {training_config.get('model_name', 'N/A')}")
        print(f"  Epochs Trained: {training_config.get('num_epochs', 'N/A')}")
        print(f"  Learning Rate: {training_config.get('learning_rate', 'N/A')}")
    
    # Prepare evaluation dataset
    print("\n📊 Preparing evaluation dataset...")
    _, eval_dataset = prepare_datasets(
        tokenizer,
        use_huggingface=use_huggingface,
        csv_path=csv_path
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Perform evaluation
    metrics = evaluate_model(model, tokenizer, eval_dataloader, device)
    
    # Save evaluation results
    results_path = os.path.join(model_path, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Evaluation results saved to: {results_path}")
    
    # Generate and display sample predictions
    if num_samples > 0:
        print(f"\n🎯 Generating {num_samples} sample predictions...")
        
        # Get a few samples
        sample_indices = list(range(min(num_samples, len(eval_dataset))))
        sample_inputs = []
        sample_references = []
        sample_predictions = []
        
        for idx in sample_indices:
            sample = eval_dataset[idx]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            labels = sample["labels"]
            
            # Generate prediction
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_length=Config.MAX_TARGET_LENGTH,
                    num_beams=Config.NUM_BEAMS,
                    no_repeat_ngram_size=Config.NO_REPEAT_NGRAM_SIZE,
                    early_stopping=Config.EARLY_STOPPING
                )
            
            # Decode
            input_text = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
            reference = tokenizer.decode(labels, skip_special_tokens=True)
            prediction = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)
            
            sample_inputs.append(input_text)
            sample_references.append(reference)
            sample_predictions.append(prediction)
        
        # Print samples
        print_sample_predictions(
            sample_predictions,
            sample_references,
            sample_inputs,
            num_samples
        )
    
    return metrics


def compare_models(model_paths: list):
    """
    Compare multiple trained models
    
    Args:
        model_paths: List of paths to trained models
    """
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON")
    print("="*60)
    
    results = []
    
    for model_path in model_paths:
        print(f"\nEvaluating: {model_path}")
        try:
            metrics = evaluate_saved_model(model_path, num_samples=0)
            results.append({
                "model": model_path,
                **metrics
            })
        except Exception as e:
            print(f"❌ Error evaluating {model_path}: {str(e)}")
    
    # Print comparison table
    if results:
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(f"{'Model':<30} {'ROUGE-1':>10} {'ROUGE-L':>10} {'F1':>10}")
        print("-"*60)
        for r in results:
            model_name = os.path.basename(r['model'])
            print(f"{model_name:<30} {r['rouge1']:>10.4f} {r['rougeL']:>10.4f} {r['token_f1']:>10.4f}")
        print("="*60)


if __name__ == "__main__":
    """
    Main evaluation script
    
    USAGE:
        python evaluate.py
    
    This will evaluate the model saved in ./model/best_model
    """
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         SummAI - Model Evaluation Script                 ║
    ║  Comprehensive Metrics for Summarization Models          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        metrics = evaluate_saved_model(num_samples=5)
        
        print("\n✅ Evaluation complete!")
        print("\n📊 Key Metrics Summary:")
        print(f"  ROUGE-1: {metrics['rouge1']:.4f}")
        print(f"  ROUGE-2: {metrics['rouge2']:.4f}")
        print(f"  ROUGE-L: {metrics['rougeL']:.4f}")
        print(f"  Token F1: {metrics['token_f1']:.4f}")
        
    except FileNotFoundError as e:
        print(f"\n❌ {str(e)}")
        print("\n💡 Please train a model first:")
        print("   python train.py")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
