"""
SummAI Configuration
Hyperparameters and settings for model training and evaluation
"""

class Config:
    """Configuration class for SummAI ML model"""
    
    # Model Settings
    MODEL_NAME = "t5-small"  
    MAX_INPUT_LENGTH = 256
    MAX_TARGET_LENGTH = 64
    
    # Training Settings
    BATCH_SIZE = 2
    NUM_EPOCHS = 3
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 100
    
    # Data Settings
    TRAIN_TEST_SPLIT = 0.2
    MAX_TRAIN_SAMPLES = None  # Set to a number to limit training samples (e.g., 1000)
    MAX_EVAL_SAMPLES = 100   # Set to a number to limit evaluation samples (e.g., 100)
    
    # IMPORTANT — USE YOUR CSV FILE
    USE_CSV = True
    CSV_DATA_PATH = "./static/data/sample_data.csv"
    
    
    # Remove HuggingFace dataset usage
    DATASET_NAME = "cnn_dailymail"
    DATASET_VERSION = "3.0.0"
    
    # Paths
    MODEL_SAVE_PATH = "./model/best_model"
    CHECKPOINT_DIR = "./model/checkpoints"
    DATA_DIR = "./static/data"
    
    # Evaluation
    ROUGE_METRICS = ["rouge1", "rouge2", "rougeL"]
    
    # Generation Settings
    NUM_BEAMS = 4
    NO_REPEAT_NGRAM_SIZE = 3
    EARLY_STOPPING = True
    
    # Device
    DEVICE = "cpu"  # use CPU because your laptop has no GPU (safe)
