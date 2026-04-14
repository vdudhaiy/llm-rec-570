# LLM-Rec Reimplementation

## Project Overview
LLM-Rec is a movie recommendation system that combines large language models (LLMs) with deep learning-based collaborative filtering. The system uses:
- **Movie Embeddings**: Semantic embeddings generated from movie descriptions using SentenceTransformer
- **LLM-Enhanced Descriptions**: Movie descriptions refined via Purdue GenAI Studio (REST API)
- **Attention-Based Neural Model**: Custom neural models for ranking recommendations
- **Multiple Model Variants**: 4 configurable model types for different use cases

## Project Structure

```
llm-rec-570/
├── main.py                             # Pipeline orchestrator (MAIN ENTRY POINT)
├── model.py                            # Neural network models (SimpleCF, LLMRec)
├── metrics.py                          # Evaluation metrics (HR, MRR, NDCG, Precision)
├── training.py                         # Model training and testing
├── data_processing.py                  # Dataset creation and embedding generation
├── fetching_descriptions.py            # Fetch movie descriptions from TMDB API
├── prompting.py                        # LLM-based description enhancement
├── config.py                           # Centralized configuration (loads from .env)
├── .env                                # Environment variables (credentials)
├── requirements.txt                    # Python dependencies
├── ECE570Project.ipynb                 # Jupyter notebook reference
└── README.md                           # This file
```

## Modular Architecture

### New Structure (Refactored)
- **model.py**: Contains `SimpleCF` (Type A) and `LLMRec` (Types B-D) model definitions + `EarlyStopping`
- **metrics.py**: Standalone metric functions (HR@k, MRR@k, NDCG@k, Precision@k)
- **training.py**: Training loops, evaluation, and test result serialization
- **main.py**: Unified pipeline orchestrator with 5 steps
- **config.py**: Centralized configuration with MODEL_TYPE selection

## Pipeline Overview

### Unified Pipeline Execution (main.py)

The `main.py` script orchestrates all steps with auto-skipping:

**Step 1: Fetch Descriptions** → `movies_desc.dat`
- Fetches movie metadata from TMDB API
- Skips if file already exists

**Step 2: Generate Enhanced Descriptions**
- Creates basic descriptions (summaries)
- Creates recommendation-driven descriptions
- Auto-skips if both files already exist

**Step 3: Generate Embeddings**
- Generates embeddings for all 4 model types:
  - Type A: Original descriptions → `embeddings.json`
  - Type B: Basic descriptions → `embeddings_basic.json`
  - Type C: Recommendation-driven → `embeddings_rec_driven.json`
  - Type D: Combined descriptions → `embeddings_combined.json`
- Auto-skips existing files

**Step 4: Create and Validate Datasets**
- Loads ratings and movies
- Creates train/val/test splits
- Validates dataset disjointness
- Logs statistics

**Step 5: Train and Test Model**
- Trains selected model type
- Evaluates on test set
- Saves results to `test_results.json`

## Model Types

| Type | Name | Description | Use Case |
|------|------|-------------|----------|
| **A** | SimpleCF | Simple collaborative filtering (learnable embeddings, no LLM) | Baseline comparison |
| **B** | LLMRec Basic | Multi-head attention with basic LLM descriptions | Fast, focused content |
| **C** | LLMRec RecDriven | Multi-head attention with recommendation-driven descriptions | Marketing-oriented |
| **D** | LLMRec Combined | Multi-head attention with combined descriptions | Best quality, slowest |

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for faster training)
- Purdue GenAI Studio API access

### Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd llm-rec-570
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Linux/Mac
   # or
   venv\Scripts\Activate.ps1     # On Windows PowerShell
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables in `.env`:
   ```env
   # TMDB API Configuration
   TMDB_API_KEY=<your_tmdb_api_key>
   
   # Purdue GenAI Studio Configuration
   PURDUE_API_URL=https://genai.rcac.purdue.edu/api/chat/completions
   PURDUE_GEN_AI_KEY=<your_purdue_genai_key>
   MODEL=gemma3:27b
   
   # Model Selection (A, B, C, or D)
   MODEL_TYPE=B
   
   # Training Parameters (optional)
   TRAIN_EPOCHS=50
   LEARNING_RATE=1e-4
   BATCH_SIZE=64
   RANDOM_SEED=42
   VERBOSE=False
   ```

5. Download MovieLens-1M dataset:
   ```bash
   # Download from https://grouplens.org/datasets/movielens/1m/
   # Extract to ./movielens-1m/ directory
   ```

## Usage

### Simplified Entry Point: main.py

**Run full pipeline** (auto-skips completed steps):
```bash
python main.py
```

**Run with specific model type**:
```bash
python main.py --model-type A    # SimpleCF
python main.py --model-type B    # LLMRec + Basic (default)
python main.py --model-type C    # LLMRec + RecDriven
python main.py --model-type D    # LLMRec + Combined
```

**Run specific pipeline step**:
```bash
python main.py --step 1          # Fetch descriptions only
python main.py --step 2          # Generate descriptions only
python main.py --step 3          # Generate embeddings only
python main.py --step 4          # Create datasets only
python main.py --step 5          # Train and test only
python main.py --step 5 -m D     # Train model type D only
```

### Individual Module Usage (Advanced)

```python
# Step 1: Fetch descriptions
import fetching_descriptions    # Runs TMDB fetch

# Step 2: Generate enhanced descriptions
from prompting import read_movies_desc, generateBasicDescriptions, generateRecDrivenDescriptions
movies = read_movies_desc()
generateBasicDescriptions(movies)
generateRecDrivenDescriptions(movies)

# Step 3: Generate embeddings
from prompting import generateDescEmbeddings
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
generateDescEmbeddings(encoder, movies)

# Step 4: Create datasets
from data_processing import createDataset
train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = createDataset()

# Step 5: Train and test
from training import train, test
from model import LLMRec
model = LLMRec(num_users=6040, num_movies=3952, embedding_dim=384)
train(model, train_loader, val_loader, embeddings, list(embeddings.keys()), model_type='B')
metrics, loss = test(model, test_loader, criterion, list(embeddings.keys()), embeddings, model_type='B')
```

## Configuration

All parameters are centralized in `config.py` and can be overridden via `.env`:

### Model Type Selection
- Set `MODEL_TYPE=A|B|C|D` in `.env` or pass `--model-type` to CLI

### LLM/Prompting Configuration
- `PURDUE_API_URL`: API endpoint
- `PURDUE_GEN_AI_KEY`: Authentication token
- `MODEL`: Model name (e.g., gemma3:27b)
- `TEMPERATURE`: 0.7
- `MAX_NEW_TOKENS`: 200

### Training Parameters
- `TRAIN_EPOCHS`: 50
- `LEARNING_RATE`: 1e-4
- `BATCH_SIZE`: 64
- `NUM_NEGATIVES`: 9
- `EARLY_STOPPING_PATIENCE`: 10
- `WEIGHT_DECAY`: 1e-5

### Dataset Parameters
- `TEST_SIZE`: 0.2
- `VAL_SIZE`: 0.2
- `POSITIVE_RATING_THRESHOLD`: 2.0

### Reproducibility
- `RANDOM_SEED`: 42

## Evaluation Metrics

Models output test results to `test_results.json`:

```json
{
  "model_type": "B",
  "model_description": "LLMRec with basic descriptions",
  "test_metrics": {
    "HR@10": 0.4521,
    "MRR@10": 0.3214,
    "NDCG@10": 0.2987,
    "Precision@10": 0.0452
  },
  "average_test_loss": 0.5231,
  "timestamp": "2026-04-13 15:30:45.123456"
}
```

### Metric Definitions
- **HR@k**: Hit Rate - fraction of test queries where true item appears in top-k
- **MRR@k**: Mean Reciprocal Rank - average of 1/rank for correct items in top-k
- **NDCG@k**: Normalized Discounted Cumulative Gain - ranking quality with position decay
- **Precision@k**: Precision - fraction of top-k predictions that are correct

## File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | **Pipeline orchestrator** - run this for end-to-end execution |
| `model.py` | SimpleCF and LLMRec model definitions with EarlyStopping |
| `metrics.py` | Standalone evaluation metric functions |
| `training.py` | Training loops, validation, test evaluation, result serialization |
| `config.py` | Centralized config management with .env integration |
| `data_processing.py` | Dataset creation, train/val/test splitting, DataLoader setup |
| `fetching_descriptions.py` | TMDB API integration with exponential backoff retry logic |
| `prompting.py` | Purdue GenAI Studio REST API calls for description enhancement |

## Notes

- **Purdue GenAI Studio**: Uses REST API for LLM inference (no local model loading)
- **GPU Acceleration**: Descriptions enhanced on Purdue's GPU infrastructure
- **Rate Limiting**: TMDB API queries are rate-limited (250ms delay between requests)
- **Data Size**: MovieLens-1M is ~200MB; ensure sufficient disk space
- **Training Time**: ~2-5 hours per epoch on GPU (depending on batch size)
- **Security**: Never commit `.env` file or API keys to repository
- **Reproducibility**: Always use consistent RANDOM_SEED for experiments
