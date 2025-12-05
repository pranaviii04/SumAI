import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.summarizer import download_nltk_data
print("Downloading NLTK data...")
download_nltk_data()
print("Done.")
