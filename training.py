"""
Training module for recommendation models
Supports training of different model types with configurable embeddings

Everything here is written to run on the GPU: negatives are sampled as tensors,
candidate scoring is done in one batched forward pass, and the embedding lookup
is a dense matrix that lives on the device for the whole run.
"""

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_processing import createDataset
from model import LLMRec, SimpleCF, EarlyStopping
from metrics import hr_at_k, mrr_at_k, ndcg_at_k
from config import (
    NUM_USERS, NUM_MOVIES, EMBEDDING_DIMENSION, TRAIN_EPOCHS, LEARNING_RATE,
    NUM_NEGATIVES, EARLY_STOPPING_PATIENCE, WEIGHT_DECAY, SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE, EVALUATION_K, EVAL_NUM_NEGATIVES, EVAL_MAX_BATCHES,
    RANDOM_SEED, SELECTED_EMBEDDINGS_FILE, EMBEDDINGS_FILE_MAP, VERBOSE, MODEL_TYPE,
    POSITIVE_RATING_THRESHOLD, USE_AMP, get_device, configure_backend, describe_device,
    output_path
)

# Set seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# Configure logging
logging.basicConfig(level=logging.INFO if VERBOSE else logging.WARNING)
logger = logging.getLogger(__name__)

MODEL_DESCRIPTIONS = {
    "A": "SimpleCF - Collaborative filtering",
    "B": "LLMRec with basic descriptions",
    "C": "LLMRec with recommendation-driven descriptions",
    "D": "LLMRec with combined descriptions",
    "E": "LLMRec with multi-view attention over each description separately",
}

METRIC_FNS = {"hr": hr_at_k, "mrr": mrr_at_k, "ndcg": ndcg_at_k}


def get_model(model_type):
    """
    Create and return the appropriate model based on model_type.

    Args:
        model_type: str, one of 'A', 'B', 'C', 'D', 'E'
                   A: SimpleCF - collaborative filtering
                   B-E: LLMRec with different embeddings

    Returns:
        model: Initialized PyTorch model
    """
    if model_type == "A":
        logger.info("Creating SimpleCF model (Type A - No embeddings)")
        return SimpleCF(num_users=NUM_USERS, num_movies=NUM_MOVIES, embedding_dim=EMBEDDING_DIMENSION)
    elif model_type in ["B", "C", "D", "E"]:
        logger.info(f"Creating LLMRec model (Type {model_type} - {MODEL_DESCRIPTIONS[model_type]})")
        return LLMRec(num_users=NUM_USERS, num_movies=NUM_MOVIES, embedding_dim=EMBEDDING_DIMENSION)
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be A, B, C, D, or E")


def get_embeddings_loader(model_type=None):
    """
    Get the embeddings file path for a model type.

    Resolution order:
      1. EMBEDDINGS_FILE_OVERRIDE, if set (swap the text, keep the architecture)
      2. the file mapped to `model_type`
      3. the file mapped to config.MODEL_TYPE

    Taking `model_type` as an argument matters: `run(model_type=...)` and the
    `--model-type` flag choose the architecture at call time, and without this the
    embeddings file would still come from whatever MODEL_TYPE is in .env - so
    `-m E` would build a multi-view model and then feed it a single-view file.

    Args:
        model_type: 'A'-'E', or None to fall back to config.MODEL_TYPE

    Returns:
        str: Path to embeddings file
    """
    override = os.getenv("EMBEDDINGS_FILE_OVERRIDE")
    if override:
        return override
    if model_type is None:
        return SELECTED_EMBEDDINGS_FILE
    return EMBEDDINGS_FILE_MAP[model_type]


# ---------------------------------------------------------------------------
# User/item bookkeeping
# ---------------------------------------------------------------------------

def build_seen_matrix(dataframe, num_users, num_movies, device,
                      rating_threshold=POSITIVE_RATING_THRESHOLD):
    """
    Build a dense boolean matrix marking which movies each user rated positively.

    This replaces the old dict-of-sets built by iterating the Dataset one sample
    at a time. For MovieLens-1M the matrix is ~24 MB, so it fits comfortably on
    the GPU and turns negative sampling into a single tensor lookup.

    Args:
        dataframe: Ratings DataFrame with userId/movieId/rating columns
        num_users: Number of user rows in the matrix
        num_movies: Number of movie columns in the matrix
        device: Device to place the matrix on
        rating_threshold: Minimum rating counted as positive feedback

    Returns:
        torch.BoolTensor: (num_users, num_movies), True where the user liked the movie
    """
    seen = torch.zeros(num_users, num_movies, dtype=torch.bool, device=device)
    positives = dataframe[dataframe['rating'] >= rating_threshold]
    users = torch.as_tensor(positives['userId'].to_numpy(), dtype=torch.long, device=device) - 1
    movies = torch.as_tensor(positives['movieId'].to_numpy(), dtype=torch.long, device=device)
    seen[users, movies] = True
    logger.info(f"Marked {len(positives)} positive interactions (threshold={rating_threshold})")
    return seen


def sample_negatives(user_ids, true_movie_ids, movie_id_pool, seen, num_negatives,
                     generator=None, max_resamples=4):
    """
    Draw negatives for each row, avoiding the true item and the user's known positives.

    Sampling is done with rejection: draw uniformly, then re-draw only the slots
    that collided. A handful of passes is enough because collisions are rare
    (a typical user has liked well under 10% of the catalogue).

    Args:
        user_ids: (batch,) user indices
        true_movie_ids: (batch,) the positive movie for each row
        movie_id_pool: (num_pool,) tensor of sampleable movie ids
        seen: (num_users, num_movies) bool matrix of known positives, or None
        num_negatives: Negatives to draw per row
        generator: Optional torch.Generator for reproducible sampling
        max_resamples: How many rejection passes to run

    Returns:
        torch.LongTensor: (batch, num_negatives) sampled movie ids
    """
    batch = user_ids.shape[0]
    device = user_ids.device
    pool_size = movie_id_pool.shape[0]

    idx = torch.randint(pool_size, (batch, num_negatives), device=device, generator=generator)
    candidates = movie_id_pool[idx]

    for _ in range(max_resamples):
        collision = candidates == true_movie_ids.unsqueeze(1)
        if seen is not None:
            collision |= seen[user_ids.unsqueeze(1).expand_as(candidates), candidates]
        if not bool(collision.any()):
            break
        redraw = torch.randint(pool_size, (batch, num_negatives), device=device, generator=generator)
        candidates = torch.where(collision, movie_id_pool[redraw], candidates)

    return candidates


def score_candidates(model, user_ids, candidate_ids, embedding_matrix, model_type, use_amp):
    """
    Score a (batch, num_candidates) grid of user/movie pairs in one forward pass.

    Args:
        model: The recommendation model
        user_ids: (batch,) user indices
        candidate_ids: (batch, num_candidates) movie ids to score
        embedding_matrix: (num_movies, num_views, dim) text embeddings indexed by movie id
        model_type: 'A' (SimpleCF, ignores text) or 'B'/'C'/'D' (LLMRec)
        use_amp: Whether to run the forward pass under bfloat16 autocast

    Returns:
        torch.Tensor: (batch, num_candidates) float32 logits
    """
    batch, num_candidates = candidate_ids.shape
    device = candidate_ids.device

    flat_users = user_ids.unsqueeze(1).expand(batch, num_candidates).reshape(-1, 1)
    flat_movies = candidate_ids.reshape(-1, 1)

    if model_type == "A":
        flat_embs = torch.zeros(batch * num_candidates, 1, EMBEDDING_DIMENSION, device=device)
    else:
        # (batch * candidates, num_views, dim) - num_views is 1 for B/C/D and
        # >1 for E, where the attention layer picks between description views.
        flat_embs = embedding_matrix[flat_movies.reshape(-1)]

    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)
    with amp_ctx:
        logits = model(flat_users, flat_movies, flat_embs)

    return logits.float().reshape(batch, num_candidates)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test(model, dataloader, criterion, movie_id_pool, embedding_matrix, model_type="B",
         k=EVALUATION_K, device=None, num_negatives=EVAL_NUM_NEGATIVES, seen=None,
         max_batches=EVAL_MAX_BATCHES):
    """
    Evaluate model on validation/test set.

    Each positive is ranked against `num_negatives` sampled negatives. That count
    must exceed k - 1, otherwise every candidate fits inside the top-k window and
    HR@k / Precision@k become constants that say nothing about the model.

    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for evaluation set
        criterion: Loss function
        movie_id_pool: Tensor of all sampleable movie ids
        embedding_matrix: (num_movies, num_views, dim) text embeddings indexed by movie id
        model_type: Model type ('A', 'B', 'C', or 'D')
        k: Cutoff for metrics computation
        device: Device to use (defaults to the configured device)
        num_negatives: Number of negative samples per positive sample
        seen: Optional (num_users, num_movies) matrix of known positives to avoid
        max_batches: Stop after this many batches (0 = evaluate everything)

    Returns:
        tuple: (metrics dict, average loss)
    """
    device = device or get_device()
    model.eval()
    metrics = {name: 0.0 for name in METRIC_FNS}
    total_loss = 0.0
    count = 0

    # Fixed generator so validation uses the same negatives every epoch and the
    # epoch-to-epoch numbers are actually comparable.
    generator = torch.Generator(device=device)
    generator.manual_seed(RANDOM_SEED)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            user_ids = batch['user_id'].to(device, non_blocking=True)
            true_movie_ids = batch['movie_id'].to(device, non_blocking=True)
            batch_size = user_ids.shape[0]

            negatives = sample_negatives(user_ids, true_movie_ids, movie_id_pool, seen,
                                         num_negatives, generator=generator)
            candidates = torch.cat([true_movie_ids.unsqueeze(1), negatives], dim=1)

            # Shuffle each row so the true item is not always at index 0 - topk
            # breaks ties by lowest index, which would silently inflate the metrics.
            perm = torch.argsort(torch.rand(candidates.shape, device=device, generator=generator), dim=1)
            candidates = torch.gather(candidates, 1, perm)
            labels = (perm == 0).float().argmax(dim=1)

            logits = score_candidates(model, user_ids, candidates, embedding_matrix,
                                      model_type, use_amp=False)

            for name, fn in METRIC_FNS.items():
                value, _ = fn(k, logits, labels)
                metrics[name] += value

            target = torch.zeros_like(logits)
            target[torch.arange(batch_size, device=device), labels] = 1.0
            total_loss += criterion(logits, target).item() * batch_size
            count += batch_size

    if count == 0:
        return {name: 0.0 for name in METRIC_FNS}, 0.0

    return {name: value / count for name, value in metrics.items()}, total_loss / count


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, criterion, movie_id_pool, embedding_matrix,
                    seen, model_type="B", device=None, num_negatives=NUM_NEGATIVES,
                    rating_threshold=POSITIVE_RATING_THRESHOLD):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model to train
        dataloader: DataLoader for training set
        optimizer: Optimizer (SGD, Adam, etc.)
        criterion: Loss function
        movie_id_pool: Tensor of all sampleable movie ids
        embedding_matrix: (num_movies, num_views, dim) text embeddings indexed by movie id
        seen: (num_users, num_movies) matrix of known positives
        model_type: Model type ('A', 'B', 'C', or 'D')
        device: Device to use (defaults to the configured device)
        num_negatives: Number of negative samples per positive sample
        rating_threshold: Ratings at or above this count as positive feedback

    Returns:
        float: Average training loss for the epoch
    """
    device = device or get_device()
    use_amp = USE_AMP and device.type == "cuda"
    model.train()
    total_loss = 0.0
    batches = 0

    for batch in dataloader:
        user_ids = batch['user_id'].to(device, non_blocking=True)
        pos_movie_ids = batch['movie_id'].to(device, non_blocking=True)
        ratings = batch['rating'].to(device, non_blocking=True)
        batch_size = user_ids.shape[0]

        negatives = sample_negatives(user_ids, pos_movie_ids, movie_id_pool, seen, num_negatives)
        candidates = torch.cat([pos_movie_ids.unsqueeze(1), negatives], dim=1)

        # Column 0 is the observed interaction: it only counts as a positive if the
        # user actually rated it at or above the threshold. The write-up defines
        # implicit feedback this way; labelling every observed row 1.0 would train
        # the model to recommend movies the user disliked.
        targets = torch.zeros(batch_size, candidates.shape[1], device=device)
        targets[:, 0] = (ratings >= rating_threshold).float()

        optimizer.zero_grad(set_to_none=True)
        logits = score_candidates(model, user_ids, candidates, embedding_matrix,
                                  model_type, use_amp=use_amp)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        batches += 1

    return total_loss / max(batches, 1)


def train(model, train_loader, val_loader, embedding_matrix, movie_id_pool, model_type="B",
          device=None, epochs=TRAIN_EPOCHS, lr=LEARNING_RATE, seen=None):
    """
    Train model with early stopping and learning rate scheduling.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        embedding_matrix: (num_movies, num_views, dim) text embeddings indexed by movie id
        movie_id_pool: Tensor of all sampleable movie ids
        model_type: Model type ('A', 'B', 'C', or 'D')
        device: Device to use (defaults to the configured device)
        epochs: Number of training epochs
        lr: Learning rate
        seen: Optional (num_users, num_movies) matrix of known positives

    Returns:
        dict: Training history with per-epoch losses and validation metrics
    """
    device = device or get_device()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR,
                                  patience=SCHEDULER_PATIENCE)
    criterion = nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    history = []

    logger.info(f"Starting training (Model Type {model_type}) for {epochs} epochs with lr={lr}")

    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, movie_id_pool,
                                     embedding_matrix, seen, model_type=model_type, device=device)
        val_metrics, val_loss = test(model, val_loader, criterion, movie_id_pool, embedding_matrix,
                                     model_type=model_type, device=device, seen=seen)
        scheduler.step(val_loss)
        elapsed = time.time() - epoch_start

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metrics": dict(val_metrics),
            "seconds": elapsed,
        })

        logger.info(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s):")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Val Metrics: HR@{EVALUATION_K}={val_metrics['hr']:.4f}, "
                    f"MRR@{EVALUATION_K}={val_metrics['mrr']:.4f}, "
                    f"NDCG@{EVALUATION_K}={val_metrics['ndcg']:.4f}")

        if not early_stopper.check_loss(val_loss):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    logger.info("Training complete!")
    if history:
        logger.info(f"Average Train Loss: {np.mean([h['train_loss'] for h in history]):.4f}")
        logger.info(f"Average Validation Loss: {np.mean([h['val_loss'] for h in history]):.4f}")

    return history


def load_embedding_matrix(embeddings_file, num_movies, embedding_dim, device):
    """
    Load embeddings from JSON into a dense (num_movies, num_views, dim) tensor.

    Two file layouts are accepted:
      * single view -> {movieId: [dim floats]}
      * multi view  -> {"views": [names], "embeddings": {movieId: [[dim], [dim], ...]}}

    Both are returned as 3D so the rest of the pipeline has one code path; a
    single-view file simply has num_views == 1.

    Args:
        embeddings_file: Path to the JSON embeddings file
        num_movies: Number of rows to allocate (max movie id + 1)
        embedding_dim: Expected embedding dimension
        device: Device to place the matrix on

    Returns:
        tuple: (embedding tensor, movie ids that have an embedding, view names)
    """
    try:
        with open(embeddings_file, "r") as f:
            payload = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Embeddings file not found: {embeddings_file}. "
            f"Run `python main.py --step 3` to generate it."
        )

    if isinstance(payload, dict) and "embeddings" in payload and "views" in payload:
        view_names = list(payload["views"])
        embeddings = payload["embeddings"]
    else:
        view_names = ["single"]
        embeddings = payload

    num_views = len(view_names)
    matrix = torch.zeros(num_movies, num_views, embedding_dim, dtype=torch.float)
    movie_ids = []
    nan_count = 0

    for key, vec in embeddings.items():
        mid = int(key)
        if mid >= num_movies:
            logger.warning(f"Skipping movie id {mid}: outside the embedding table ({num_movies} rows)")
            continue
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape != (num_views, embedding_dim):
            raise ValueError(
                f"Embedding shape mismatch for movie {mid}: "
                f"expected {(num_views, embedding_dim)}, got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            nan_count += 1
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        matrix[mid] = torch.from_numpy(arr)
        movie_ids.append(mid)

    if nan_count:
        logger.warning(f"Found and cleaned {nan_count} embeddings with NaN/inf values")

    logger.info(f"Loaded {len(movie_ids)} embeddings ({num_views} view(s): "
                f"{', '.join(view_names)}) onto {device}")
    return (matrix.to(device),
            torch.tensor(sorted(movie_ids), dtype=torch.long, device=device),
            view_names)


def mean_attention_weights(model, dataloader, embedding_matrix, view_names, device,
                           max_batches=50):
    """
    Average the model's per-view attention weights over real interactions.

    This is the payoff of multi-view attention: rather than inferring which
    prompting strategy helped by comparing final scores across separate training
    runs, the model reports directly how much weight it placed on each
    description type.

    Args:
        model: A trained LLMRec
        dataloader: DataLoader to sample interactions from
        embedding_matrix: (num_movies, num_views, dim) embeddings
        view_names: Names of the views, in matrix order
        device: Device to run on
        max_batches: How many batches to average over

    Returns:
        dict: view name -> mean attention weight, or None when there is nothing
              meaningful to report (fewer than two views, or no attention layer)
    """
    if embedding_matrix.shape[1] < 2:
        return None

    model.eval()
    totals = torch.zeros(len(view_names), device=device)
    count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            user_ids = batch['user_id'].to(device).unsqueeze(1)
            movie_ids = batch['movie_id'].to(device).unsqueeze(1)
            embs = embedding_matrix[movie_ids.reshape(-1)]

            out = model(user_ids, movie_ids, embs, return_attention=True)
            if not isinstance(out, tuple) or out[1] is None:
                return None
            totals += out[1].sum(dim=0)
            count += out[1].shape[0]

    if count == 0:
        return None
    return {name: float(v) for name, v in zip(view_names, (totals / count).tolist())}


def run(model_type=MODEL_TYPE, epochs=TRAIN_EPOCHS, results_file=None):
    """
    Full train + test run for one model type, writing metrics to a JSON file.

    Args:
        model_type: Model type to train ('A', 'B', 'C', or 'D')
        epochs: Number of training epochs
        results_file: Results filename. Relative paths land in OUTPUT_DIR;
                      defaults to test_results_<model_type>.json.

    Returns:
        dict: The results payload that was written to disk
    """
    results_file = output_path(results_file or f"test_results_{model_type}.json")
    device = get_device()
    configure_backend(device)
    embeddings_file = get_embeddings_loader(model_type)

    logger.info("=" * 60)
    logger.info("Training Configuration:")
    logger.info(f"  Device: {describe_device(device)}")
    logger.info(f"  Model Type: {model_type}")
    logger.info(f"  Description: {MODEL_DESCRIPTIONS.get(model_type)}")
    logger.info(f"  Embeddings File: {embeddings_file}")
    logger.info(f"  Epochs: {epochs}")
    logger.info("=" * 60)

    logger.info("Loading datasets...")
    train_dataset, _, _, train_loader, val_loader, test_loader = createDataset()

    model = get_model(model_type).to(device)

    embedding_matrix, movie_id_pool, view_names = load_embedding_matrix(
        embeddings_file, NUM_MOVIES, EMBEDDING_DIMENSION, device
    )

    seen = build_seen_matrix(train_dataset.data, NUM_USERS, NUM_MOVIES, device)

    logger.info("Starting model training...")
    start = time.time()
    history = train(model, train_loader, val_loader, embedding_matrix, movie_id_pool,
                    model_type=model_type, device=device, epochs=epochs, seen=seen)
    train_seconds = time.time() - start

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 60)
    testing, avg_test_loss = test(model, test_loader, nn.BCEWithLogitsLoss(), movie_id_pool,
                                  embedding_matrix, model_type=model_type, device=device, seen=seen)

    logger.info(f"\nTest Results for Model Type {model_type}:")
    for name, label in [("hr", "HR"), ("mrr", "MRR"), ("ndcg", "NDCG")]:
        logger.info(f"  {label}@{EVALUATION_K}: {testing[name]:.4f}")
    logger.info(f"  Average Test Loss: {avg_test_loss:.4f}")

    attention = mean_attention_weights(model, test_loader, embedding_matrix,
                                       view_names, device)
    if attention:
        logger.info("\n  Mean attention weight per description view:")
        for name, weight in sorted(attention.items(), key=lambda kv: -kv[1]):
            logger.info(f"    {name:<14} {weight:6.1%}")

    results = {
        "model_type": model_type,
        "model_description": MODEL_DESCRIPTIONS.get(model_type, "Unknown"),
        "embeddings_file": embeddings_file,
        "device": describe_device(device),
        "epochs_run": len(history),
        "train_seconds": round(train_seconds, 2),
        "eval_num_negatives": EVAL_NUM_NEGATIVES,
        "views": view_names,
        "mean_attention_weights": attention,
        "test_metrics": {
            f"HR@{EVALUATION_K}": float(testing['hr']),
            f"MRR@{EVALUATION_K}": float(testing['mrr']),
            f"NDCG@{EVALUATION_K}": float(testing['ndcg']),
        },
        "average_test_loss": float(avg_test_loss),
        "history": history,
        "timestamp": str(pd.Timestamp.now()),
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nTest results saved to {results_file}")
    logger.info("=" * 60)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a recommendation model")
    parser.add_argument('--model-type', '-m', choices=['A', 'B', 'C', 'D', 'E'], default=MODEL_TYPE)
    parser.add_argument('--epochs', '-e', type=int, default=TRAIN_EPOCHS)
    parser.add_argument('--results-file', default=None,
                        help='Results filename. Relative paths are saved under OUTPUT_DIR '
                             '(default: outputs/). Defaults to test_results_<TYPE>.json.')
    args = parser.parse_args()

    run(model_type=args.model_type, epochs=args.epochs, results_file=args.results_file)
