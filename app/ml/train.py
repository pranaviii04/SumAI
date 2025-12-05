"""
SummAI Training Script
Train a transformer-based summarization model

USAGE:
    python train.py

This will:
1. Load the dataset (CNN/DailyMail by default)
2. Initialize a T5 model
3. Train for specified epochs
4. Save the best model to ./model/best_model
5. Display training progress and metrics
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import json

from app.config import Config
from app.core.data_loader import prepare_datasets
from app.core.metrics import evaluate_model


def train_model(
    model_name: str = Config.MODEL_NAME,
    batch_size: int = Config.BATCH_SIZE,
    num_epochs: int = Config.NUM_EPOCHS,
    learning_rate: float = Config.LEARNING_RATE,
    use_huggingface: bool = True,
    csv_path: str = None
):
    """
    Train the summarization model
    
    Args:
        model_name: HuggingFace model name (t5-small, facebook/bart-base, etc.)
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        use_huggingface: Whether to use HuggingFace dataset
        csv_path: Path to custom CSV (if use_huggingface=False)
    """
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")
    
    # Load tokenizer and model
    print(f"[Loading] Loading {model_name} model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    
    # Prepare datasets
    print("[Data] Preparing datasets...")
    train_dataset, eval_dataset = prepare_datasets(
        tokenizer,
        use_huggingface=use_huggingface,
        csv_path=csv_path
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("\\n[Training] Starting training...")
    print(f"Total epochs: {num_epochs}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    print(f"Steps per epoch: {len(train_dataloader)}")
    print(f"Total training steps: {total_steps}\\n")
    
    best_eval_loss = float('inf')
    training_history = []
    
    for epoch in range(num_epochs):
        print(f"\\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_train_loss / (step + 1):.4f}'
            })
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Evaluation phase
        print("\\n[Evaluation] Running evaluation...")
        eval_metrics = evaluate_model(model, tokenizer, eval_dataloader, device)
        
        # Save metrics
        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            **eval_metrics
        }
        training_history.append(epoch_results)
        
        # Save best model
        if eval_metrics["eval_loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["eval_loss"]
            print(f"\\n[Checkpoint] New best model! Saving to {Config.MODEL_SAVE_PATH}")
            
            # Create directory if it doesn't exist
            os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(Config.MODEL_SAVE_PATH)
            tokenizer.save_pretrained(Config.MODEL_SAVE_PATH)
            
            # Save training config
            config_path = os.path.join(Config.MODEL_SAVE_PATH, "training_config.json")
            with open(config_path, 'w') as f:
                json.dump({
                    "model_name": model_name,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "best_eval_loss": best_eval_loss
                }, f, indent=2)
        
        # Print epoch summary
        print(f"\\n[Results] Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Eval Loss:  {eval_metrics['eval_loss']:.4f}")
        print(f"  ROUGE-1:    {eval_metrics['rouge1']:.4f}")
        print(f"  ROUGE-L:    {eval_metrics['rougeL']:.4f}")
    
    # Save training history
    history_path = os.path.join(Config.MODEL_SAVE_PATH, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("\\n" + "="*60)
    print("[COMPLETE] TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model saved to: {Config.MODEL_SAVE_PATH}")
    print(f"Best eval loss: {best_eval_loss:.4f}")
    print(f"Training history saved to: {history_path}")
    
    return model, tokenizer, training_history


if __name__ == "__main__":
    """
    Main training script
    
    To train with default settings:
        python train.py
    
    To modify settings, edit config.py or modify the train_model() call below
    """
    
    print("""
    ==============================================================
    ||           SummAI - ML Training Script                  ||
    ||  Transformer-based Text Summarization Training         ||
    ==============================================================
    """)
    
    # Train the model
    # Use local CSV by default since Config.DATASET_NAME is None
    model, tokenizer, history = train_model(
        use_huggingface=False,
        csv_path=Config.CSV_DATA_PATH
    )
    
    print("\\n[Ready] Ready to evaluate or make predictions!")
    print("Run: python evaluate.py  (to evaluate)")
    print("Run: python predict.py   (to test predictions)")
