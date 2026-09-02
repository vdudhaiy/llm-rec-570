"""
Global Configuration for LLM-Rec Project
Loads configuration from .env file and environment variables
"""

import os
import warnings

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

# Type E: no new descriptions - reuses the per-view text of types A/B/C, but
# embeds each view separately instead of concatenating them into one string.

# ===== EMBEDDING CONFIGURATION =====
EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDINGS_FILE = "movie_embeddings.json"
ENHANCED_EMBEDDINGS_FILE = "movies_with_embeddings.json"

# Embeddings files for different model types
EMBEDDINGS_BASIC = "embeddings_basic.json"
EMBEDDINGS_REC_DRIVEN = "embeddings_rec_driven.json"
EMBEDDINGS_COMBINED = "embeddings_combined.json"

# Type E: one embedding per description *view*, stacked per movie.
# Shape on disk: {"views": [...], "embeddings": {movieId: [[dim], [dim], ...]}}
EMBEDDINGS_MULTIVIEW = "embeddings_multiview.json"

# Which single-view embedding files feed the multi-view stack, in order.
# Views whose file is missing are skipped, and the stack is built from the rest.
MULTIVIEW_SOURCES = [
    ("original", "movie_embeddings.json"),
    ("basic", EMBEDDINGS_BASIC),
    ("rec_driven", EMBEDDINGS_REC_DRIVEN),
]

# Map model type to enhanced descriptions and embeddings files
DESCRIPTIONS_FILE_MAP = {
    "A": MOVIES_DESC_FILE,  # Original descriptions from TMDB (movies_desc.dat)
    "B": MOVIES_ENHANCED_DESC_BASIC,
    "C": MOVIES_ENHANCED_DESC_REC_DRIVEN,
    "D": MOVIES_ENHANCED_DESC_COMBINED,
    "E": MOVIES_DESC_FILE,  # Type E reuses existing text; it adds no new .dat file
}

EMBEDDINGS_FILE_MAP = {
    "A": "movie_embeddings.json",
    "B": EMBEDDINGS_BASIC,
    "C": EMBEDDINGS_REC_DRIVEN,
    "D": EMBEDDINGS_COMBINED,
    "E": EMBEDDINGS_MULTIVIEW,
}

# ===== MODEL CONFIGURATION =====
NUM_USERS = 6040  # MovieLens-1M
# MovieLens-1M movie IDs run 1..3952 and are used directly as nn.Embedding indices,
# so the table needs 3953 rows (index 3952 must be valid).
NUM_MOVIES = 3953  # MovieLens-1M (max movieId 3952, +1 because IDs are 1-based)
ATTENTION_HEADS = 2
DROPOUT_RATE = 0.4
EMBEDDING_DIMENSION = 384

# Model Type Selection
# A: SimpleCF - Collaborative filtering without text embeddings
# B: LLMRec with basic descriptions only
# C: LLMRec with recommendation-driven descriptions only
# D: LLMRec with combined basic and recommendation-driven descriptions
# E: LLMRec with multi-view attention over each description separately
MODEL_TYPE = os.getenv("MODEL_TYPE", "B").upper()

VALID_MODEL_TYPES = ["A", "B", "C", "D", "E"]
if MODEL_TYPE not in VALID_MODEL_TYPES:
    raise ValueError(
        f"Invalid MODEL_TYPE '{MODEL_TYPE}'. Must be one of: {', '.join(VALID_MODEL_TYPES)}"
    )

# Select descriptions and embeddings files based on model type.
# EMBEDDINGS_FILE_OVERRIDE decouples the *architecture* from the *text* it is fed.
# That is what the paper's baseline comparison needs: the same LLMRec model run
# once on original-description embeddings and once on LLM-enhanced ones. Without
# it, switching text also switches architecture (Type A is SimpleCF), which
# measures something else entirely.
SELECTED_DESCRIPTIONS_FILE = DESCRIPTIONS_FILE_MAP[MODEL_TYPE]
SELECTED_EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE_OVERRIDE") or EMBEDDINGS_FILE_MAP[MODEL_TYPE]

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
# Number of negatives per positive at EVALUATION time. Must be > EVALUATION_K - 1,
# otherwise every candidate fits inside the top-k and HR@k/Precision@k are constant.
EVAL_NUM_NEGATIVES = int(os.getenv("EVAL_NUM_NEGATIVES", "99"))
EVAL_MAX_BATCHES = int(os.getenv("EVAL_MAX_BATCHES", "0"))  # 0 = evaluate the full split
POSITIVE_RATING_THRESHOLD = 2.0  # Ratings >= this are considered positive feedback

# ===== PURDUE GENAI CONFIGURATION =====
# Load from .env file
PURDUE_API_URL = os.getenv("PURDUE_API_URL")
PURDUE_API_KEY = os.getenv("PURDUE_GEN_AI_KEY")
LLM_MODEL = os.getenv("MODEL")

# Purdue GenAI credentials are only needed for the prompting step (Step 2).
# Training and evaluation must still work without them.
PURDUE_CONFIGURED = all([PURDUE_API_URL, PURDUE_API_KEY, LLM_MODEL])
if not PURDUE_CONFIGURED:
    warnings.warn(
        "Purdue GenAI configuration incomplete (need PURDUE_API_URL, PURDUE_GEN_AI_KEY, MODEL "
        "in .env). Description-generation steps will fail; training is unaffected."
    )

# LLM Generation Parameters
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))
LLM_BATCH_SIZE = int(os.getenv("LLM_BATCH_SIZE", "1"))

# ===== OUTPUT FILES =====
BASIC_DESC_FILE = "basic_descriptions.txt"
REC_DRIVEN_DESC_FILE = "recommendation_driven_descriptions.txt"

# ===== RESULTS DIRECTORY =====
# Everything a run *produces as a result* - metrics JSON, run logs - lands here.
# Intermediate data (descriptions, embeddings) stays at the project root, because
# those are inputs to the next step and the resume logic looks for them there.
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")


def output_path(filename):
    """
    Resolve a results filename into OUTPUT_DIR, creating the directory as needed.

    Absolute paths are honoured as given, so a caller can still write anywhere.
    Relative paths are placed under OUTPUT_DIR, and a path that already names
    OUTPUT_DIR is left alone rather than nested a second time.

    Args:
        filename: Desired results path, absolute or relative

    Returns:
        str: Path to write to
    """
    if os.path.isabs(filename):
        resolved = filename
    else:
        norm = os.path.normpath(filename)
        out_norm = os.path.normpath(OUTPUT_DIR)
        if norm == out_norm or norm.startswith(out_norm + os.sep):
            resolved = norm
        else:
            resolved = os.path.join(OUTPUT_DIR, norm)

    parent = os.path.dirname(resolved)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError:
            # e.g. a special path like /dev/null - let the caller's open() decide
            pass
    return resolved

# ===== TMDB API CONFIGURATION =====
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
# Optional path to a file holding the key, kept for backwards compatibility.
TMDB_API_KEY_FILE = os.getenv("TMDB_API_KEY_FILE")
TMDB_INITIAL_DELAY = float(os.getenv("TMDB_INITIAL_DELAY", "0.25"))
TMDB_MAX_RETRIES = int(os.getenv("TMDB_MAX_RETRIES", "3"))
TMDB_INITIAL_BACKOFF = float(os.getenv("TMDB_INITIAL_BACKOFF", "1"))
TMDB_MAX_BACKOFF = float(os.getenv("TMDB_MAX_BACKOFF", "60"))

# The TMDB key is only needed for Step 1 (fetching descriptions). Once
# movies_desc.dat exists nothing else in the pipeline touches TMDB.
if not TMDB_API_KEY and TMDB_API_KEY_FILE and os.path.exists(TMDB_API_KEY_FILE):
    with open(TMDB_API_KEY_FILE, "r") as _f:
        TMDB_API_KEY = _f.read().strip()

if not TMDB_API_KEY:
    warnings.warn(
        "TMDB_API_KEY not set in .env. Step 1 (fetching descriptions) is unavailable; "
        "all other steps work as long as movies_desc.dat already exists."
    )

# ===== TRIAL / SMOKE-RUN CONTROLS =====
# Fraction of the ratings file to keep (1.0 = full 1M). Useful for quick trial runs.
SUBSAMPLE_FRAC = float(os.getenv("SUBSAMPLE_FRAC", "1.0"))

# ===== REPRODUCIBILITY =====
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# ===== DEVICE CONFIGURATION =====
# DEVICE_TYPE: "auto" (default) picks CUDA when a GPU is visible, otherwise CPU.
# Force a device with DEVICE_TYPE=cuda or DEVICE_TYPE=cpu in .env.
DEVICE_TYPE = os.getenv("DEVICE_TYPE", "auto").lower()

# Mixed precision (bfloat16 autocast) on CUDA. Big speedup on RTX 30xx and newer.
USE_AMP = os.getenv("USE_AMP", "True").lower() == "true"

# TF32 matmuls: another free speedup on Ampere+ GPUs, negligible accuracy impact here.
USE_TF32 = os.getenv("USE_TF32", "True").lower() == "true"

# DataLoader workers. 0 is the safe default on Windows (spawn-based multiprocessing).
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))
PIN_MEMORY = os.getenv("PIN_MEMORY", "True").lower() == "true"


def get_device():
    """
    Resolve the torch device to run on, honouring DEVICE_TYPE.

    Returns:
        torch.device: cuda device when available/requested, else cpu.
    """
    import torch

    if DEVICE_TYPE == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if DEVICE_TYPE == "cuda":
        warnings.warn("DEVICE_TYPE=cuda requested but no CUDA GPU is visible. Falling back to CPU.")
    return torch.device("cpu")


def configure_backend(device):
    """
    Apply GPU performance settings (TF32) once a device has been chosen.

    Args:
        device: torch.device returned by get_device()
    """
    import torch

    if device.type == "cuda" and USE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def describe_device(device):
    """Human-readable one-line summary of the active device."""
    import torch

    if device.type != "cuda":
        return "CPU"
    idx = device.index or 0
    name = torch.cuda.get_device_name(idx)
    total = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
    major, minor = torch.cuda.get_device_capability(idx)
    return f"{name} (sm_{major}{minor}, {total:.1f} GB, CUDA {torch.version.cuda})"

# ===== LOGGING =====
LOG_FILE = "execution.log"
VERBOSE = os.getenv("VERBOSE", "False").lower() == "true"
