"""
Global Configuration for LLM-Rec Project
Loads configuration from .env file and environment variables
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ===== DATASET CONFIGURATION =====
DATA_DIR = os.getenv("DATA_DIR", "movielens-1m")
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.dat")
MOVIES_FILE = os.path.join(DATA_DIR, "movies.dat")
MOVIES_DESC_FILE = "movies_desc.dat"

# Enhanced description files for different model types
# Type B: Basic descriptions only
MOVIES_ENHANCED_DESC_BASIC = "movies_enhanced_desc_basic.dat"

# Type C: Recommendation-driven descriptions only
MOVIES_ENHANCED_DESC_REC_DRIVEN = "movies_enhanced_desc_rec_driven.dat"

# Type D: Combined basic and recommendation-driven descriptions
MOVIES_ENHANCED_DESC_COMBINED = "movies_enhanced_desc_combined.dat"

# ===== EMBEDDING CONFIGURATION =====
EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDINGS_FILE = "movie_embeddings.json"
ENHANCED_EMBEDDINGS_FILE = "movies_with_embeddings.json"

# Embeddings files for different model types
EMBEDDINGS_BASIC = "embeddings_basic.json"
EMBEDDINGS_REC_DRIVEN = "embeddings_rec_driven.json"
EMBEDDINGS_COMBINED = "embeddings_combined.json"

# Map model type to enhanced descriptions and embeddings files
DESCRIPTIONS_FILE_MAP = {
    "A": MOVIES_DESC_FILE,  # Original descriptions from TMDB (movies_desc.dat)
    "B": MOVIES_ENHANCED_DESC_BASIC,
    "C": MOVIES_ENHANCED_DESC_REC_DRIVEN,
    "D": MOVIES_ENHANCED_DESC_COMBINED,
}

EMBEDDINGS_FILE_MAP = {
    "A": "movie_embeddings.json",
    "B": EMBEDDINGS_BASIC,
    "C": EMBEDDINGS_REC_DRIVEN,
    "D": EMBEDDINGS_COMBINED,
}

# ===== MODEL CONFIGURATION =====
NUM_USERS = 6040  # MovieLens-1M
NUM_MOVIES = 3952  # MovieLens-1M
ATTENTION_HEADS = 2
DROPOUT_RATE = 0.4
EMBEDDING_DIMENSION = 384

# Model Type Selection
# A: SimpleCF - Collaborative filtering without text embeddings
# B: LLMRec with basic descriptions only
# C: LLMRec with recommendation-driven descriptions only
# D: LLMRec with combined basic and recommendation-driven descriptions
MODEL_TYPE = os.getenv("MODEL_TYPE", "B").upper()

if MODEL_TYPE not in ["A", "B", "C", "D"]:
    raise ValueError(f"Invalid MODEL_TYPE '{MODEL_TYPE}'. Must be one of: A, B, C, D")

# Select descriptions and embeddings files based on model type
SELECTED_DESCRIPTIONS_FILE = DESCRIPTIONS_FILE_MAP[MODEL_TYPE]
SELECTED_EMBEDDINGS_FILE = EMBEDDINGS_FILE_MAP[MODEL_TYPE]

# ===== TRAINING CONFIGURATION =====
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "50"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
NUM_NEGATIVES = int(os.getenv("NUM_NEGATIVES", "9"))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", "10"))
WEIGHT_DECAY = 1e-5
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5

# ===== EVALUATION CONFIGURATION =====
TEST_SIZE = 0.2
VAL_SIZE = 0.2
EVALUATION_K = 10  # Top-k for evaluation metrics
POSITIVE_RATING_THRESHOLD = 2.0  # Ratings >= this are considered positive feedback

# ===== PURDUE GENAI CONFIGURATION =====
# Load from .env file
PURDUE_API_URL = os.getenv("PURDUE_API_URL")
PURDUE_API_KEY = os.getenv("PURDUE_GEN_AI_KEY")
LLM_MODEL = os.getenv("MODEL")

# Validate Purdue GenAI configuration
if not all([PURDUE_API_URL, PURDUE_API_KEY, LLM_MODEL]):
    raise ValueError(
        "Missing required Purdue GenAI configuration in .env file. "
        "Please set: PURDUE_API_URL, PURDUE_GEN_AI_KEY, and MODEL"
    )

# LLM Generation Parameters
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))
LLM_BATCH_SIZE = int(os.getenv("LLM_BATCH_SIZE", "1"))

# ===== OUTPUT FILES =====
BASIC_DESC_FILE = "basic_descriptions.txt"
REC_DRIVEN_DESC_FILE = "recommendation_driven_descriptions.txt"

# ===== TMDB API CONFIGURATION =====
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_INITIAL_DELAY = float(os.getenv("TMDB_INITIAL_DELAY", "0.25"))
TMDB_MAX_RETRIES = int(os.getenv("TMDB_MAX_RETRIES", "3"))
TMDB_INITIAL_BACKOFF = float(os.getenv("TMDB_INITIAL_BACKOFF", "1"))
TMDB_MAX_BACKOFF = float(os.getenv("TMDB_MAX_BACKOFF", "60"))

if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY not found in .env file")

# ===== REPRODUCIBILITY =====
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# ===== DEVICE CONFIGURATION =====
USE_CUDA = True
DEVICE_TYPE = os.getenv("DEVICE_TYPE", "cuda")  # "cuda" or "cpu"

# ===== LOGGING =====
LOG_FILE = "execution.log"
VERBOSE = os.getenv("VERBOSE", "False").lower() == "true"
