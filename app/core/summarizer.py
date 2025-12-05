"""
SummAI - Summarization Logic Module

This module contains the core extractive summarization logic using NLTK.
It analyzes text and extracts the most important sentences based on word frequency.
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from typing import Dict


def summarize_text(text: str) -> str:
    """
    Summarize business text using extractive summarization.
    
    Algorithm:
    1. Tokenize text into words and sentences
    2. Remove English stopwords
    3. Build word frequency table
    4. Score each sentence based on word frequencies
    5. Select sentences with score > 1.2 × average score
    6. Join selected sentences to form summary
    
    Args:
        text (str): The input text to summarize
        
    Returns:
        str: The summarized text or error message
    """
    
    # Handle empty or whitespace-only input
    if not text or text.strip() == "":
        return "No summary could be generated. Input text is empty."
    
    try:
        # Step 1: Tokenize text into words
        words = word_tokenize(text.lower())
        
        # Step 2: Remove English stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        
        # Handle case where no meaningful words exist
        if not filtered_words:
            return "No summary could be generated. No meaningful words found."
        
        # Step 3: Build word frequency table
        freq_table: Dict[str, int] = {}
        for word in filtered_words:
            freq_table[word] = freq_table.get(word, 0) + 1
        
        # Step 4: Tokenize text into sentences
        sentences = sent_tokenize(text)
        
        # Handle case where no sentences exist
        if not sentences:
            return "No summary could be generated. No valid sentences found."
        
        # Step 5: Score each sentence based on word frequencies
        sentence_scores: Dict[str, float] = {}
        
        for sentence in sentences:
            sentence_words = word_tokenize(sentence.lower())
            word_count = 0
            
            for word in sentence_words:
                if word in freq_table:
                    sentence_scores[sentence] = sentence_scores.get(sentence, 0) + freq_table[word]
                    word_count += 1
            
            # Normalize score by number of words in sentence (avoid long sentence bias)
            if word_count > 0 and sentence in sentence_scores:
                sentence_scores[sentence] = sentence_scores[sentence] / word_count
        
        # Handle case where no sentences were scored
        if not sentence_scores:
            return "No summary could be generated. Unable to score sentences."
        
        # Step 6: Compute average sentence score
        average_score = sum(sentence_scores.values()) / len(sentence_scores)
        
        # Step 7: Select sentences with score > 1.2 × average score
        threshold = 1.2 * average_score
        summary_sentences = [sentence for sentence in sentences 
                           if sentence in sentence_scores and sentence_scores[sentence] > threshold]
        
        # If no sentences meet threshold, return the top-scoring sentence
        if not summary_sentences:
            top_sentence = max(sentence_scores, key=sentence_scores.get)
            return top_sentence
        
        # Step 8: Return joined summary string
        summary = ' '.join(summary_sentences)
        return summary
    
    except Exception as e:
        return f"No summary could be generated. Error: {str(e)}"


def download_nltk_data():
    """
    Download required NLTK data packages if not already present.
    This function should be called once during application startup.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK 'stopwords' corpus...")
        nltk.download('stopwords')
    
    print("✓ NLTK data ready!")
