"""
SummAI Prediction Script
Generate summaries for new business text using trained model

USAGE:
    # Interactive mode
    python predict.py
    
    # Command-line mode
    python predict.py --text "Your business text here..."
    
    # From file
    python predict.py --file path/to/document.txt
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import os

from app.config import Config


class Summarizer:
    """ML-based summarizer using trained transformer model"""
    
    def __init__(self, model_path: str = Config.MODEL_SAVE_PATH):
        """
        Initialize the summarizer
        
        Args:
            model_path: Path to trained model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please train a model first using: python train.py"
            )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Device] Loading model on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("[Success] Model loaded successfully!")
    
    def generate_summary(
        self,
        text: str,
        max_length: int = Config.MAX_TARGET_LENGTH,
        num_beams: int = Config.NUM_BEAMS,
        no_repeat_ngram_size: int = Config.NO_REPEAT_NGRAM_SIZE,
        early_stopping: bool = Config.EARLY_STOPPING
    ) -> str:
        """
        Generate a summary for the input text
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of generated summary
            num_beams: Number of beams for beam search
            no_repeat_ngram_size: Size of n-grams to avoid repeating
            early_stopping: Whether to stop when all beams finished
        
        Returns:
            Generated summary string
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            max_length=Config.MAX_INPUT_LENGTH,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Generate summary
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping
            )
        
        # Decode and return
        summary = self.tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)
        return summary


def interactive_mode(summarizer: Summarizer):
    """
    Run summarizer in interactive mode
    
    Args:
        summarizer: Initialized Summarizer instance
    """
    print("\n" + "="*60)
    print("[Interactive] INTERACTIVE SUMMARIZATION MODE")
    print("="*60)
    print("Enter business text to summarize (or 'quit' to exit)")
    print("For multi-line input, end with a line containing only '---'")
    print("="*60 + "\n")
    
    while True:
        print("\nEnter text (or 'quit' to exit):")
        lines = []
        
        while True:
            line = input()
            if line.lower() == 'quit':
                print("\n[Exit] Goodbye!")
                return
            if line == '---':
                break
            lines.append(line)
        
        text = ' '.join(lines).strip()
        
        if not text:
            print("[Warning] No text entered. Please try again.")
            continue
        
        print("\n[Processing] Generating summary...")
        summary = summarizer.generate_summary(text)
        
        print("\n" + "="*60)
        print("[Input] ORIGINAL TEXT:")
        print("-"*60)
        print(text[:500] + ("..." if len(text) > 500 else ""))
        print("\n" + "="*60)
        print("[Output] GENERATED SUMMARY:")
        print("-"*60)
        print(summary)
        print("="*60)
        print(f"\nCompression: {len(text)} → {len(summary)} characters ({len(summary)/len(text)*100:.1f}%)")


def main():
    """Main prediction function"""
    
    parser = argparse.ArgumentParser(
        description="SummAI - Generate summaries using trained ML model"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to summarize"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to text file to summarize"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=Config.MODEL_SAVE_PATH,
        help="Path to trained model"
    )
    
    args = parser.parse_args()
    
    print("""
    ==============================================================
    ||          SummAI - ML Prediction Script                   ||
    ||  Generate Summaries with Trained Transformer Model       ||
    ==============================================================
    """)
    
    # Initialize summarizer
    try:
        summarizer = Summarizer(args.model_path)
    except FileNotFoundError as e:
        print(f"\n[Error] {str(e)}")
        return
    
    # Process based on arguments
    if args.text:
        # Command-line text mode
        print("\n[Processing] Generating summary...")
        summary = summarizer.generate_summary(args.text)
        
        print("\n" + "="*60)
        print("[Output] GENERATED SUMMARY:")
        print("="*60)
        print(summary)
        print("="*60)
        
    elif args.file:
        # File mode
        if not os.path.exists(args.file):
            print(f"[Error] File not found: {args.file}")
            return
        
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"\n[File] Processing file: {args.file}")
        print(f"File length: {len(text)} characters")
        print("\n[Processing] Generating summary...")
        
        summary = summarizer.generate_summary(text)
        
        print("\n" + "="*60)
        print("[Output] GENERATED SUMMARY:")
        print("="*60)
        print(summary)
        print("="*60)
        print(f"\nCompression: {len(text)} → {len(summary)} characters ({len(summary)/len(text)*100:.1f}%)")
        
        # Optionally save summary
        output_file = args.file.rsplit('.', 1)[0] + "_summary.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"\n[Saved] Summary saved to: {output_file}")
        
    else:
        # Interactive mode
        interactive_mode(summarizer)


if __name__ == "__main__":
    main()


# Example usage in code:
"""
from predict import Summarizer

# Initialize
summarizer = Summarizer()

# Generate summary
text = "Your long business document here..."
summary = summarizer.generate_summary(text)
print(summary)
"""
