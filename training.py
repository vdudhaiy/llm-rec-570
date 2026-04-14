"""
Training module for recommendation models
Supports training of different model types with configurable embeddings
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import numpy as np
import pandas as pd
import random
import logging
from collections import defaultdict
from data_processing import createDataset
from model import LLMRec, SimpleCF, EarlyStopping
from metrics import hr_at_k, mrr_at_k, ndcg_at_k, precision_at_k
from config import (
    NUM_USERS, NUM_MOVIES, EMBEDDING_DIMENSION, TRAIN_EPOCHS, LEARNING_RATE,
    NUM_NEGATIVES, EARLY_STOPPING_PATIENCE, WEIGHT_DECAY, SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE, EVALUATION_K, RANDOM_SEED, DEVICE_TYPE, 
    SELECTED_EMBEDDINGS_FILE, VERBOSE, MODEL_TYPE
)

# Set seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Configure logging
logging.basicConfig(level=logging.INFO if VERBOSE else logging.WARNING)
logger = logging.getLogger(__name__)


def get_model(model_type):
    """
    Create and return the appropriate model based on model_type.
    
    Args:
        model_type: str, one of 'A', 'B', 'C', 'D'
                   A: SimpleCF - collaborative filtering
                   B-D: LLMRec with different embeddings
                   
    Returns:
        model: Initialized PyTorch model
    """
    if model_type == "A":
        logger.info(f"Creating SimpleCF model (Type A - No embeddings)")
        return SimpleCF(num_users=NUM_USERS, num_movies=NUM_MOVIES, embedding_dim=EMBEDDING_DIMENSION)
    elif model_type in ["B", "C", "D"]:
        type_names = {"B": "Basic descriptions", "C": "Recommendation-driven descriptions", 
                      "D": "Combined descriptions"}
        logger.info(f"Creating LLMRec model (Type {model_type} - {type_names.get(model_type)})")
        return LLMRec(num_users=NUM_USERS, num_movies=NUM_MOVIES, embedding_dim=EMBEDDING_DIMENSION)
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be A, B, C, or D")


def get_embeddings_loader():
    """
    Get the appropriate embeddings file path based on model type.
    
    Returns:
        str: Path to embeddings file
    """
    return SELECTED_EMBEDDINGS_FILE


# Helper functions for user-movie relationships
def build_user_positive_dict(dataset, rating_threshold=2.0):
    """
    Build a dictionary of positive (liked) items per user based on rating threshold.
    
    Args:
        dataset: List of data samples with 'user_id', 'movie_id', 'rating' keys
        rating_threshold: Minimum rating to consider as positive feedback
        
    Returns:
        dict: user_id -> set of positive movie_ids
    """
    user_pos = defaultdict(set)
    for data in dataset:
        uid = int(data['user_id']) 
        mid = int(data['movie_id'])  
        rating = float(data['rating'])
        if rating >= rating_threshold:
            user_pos[uid].add(mid)
    return user_pos


def build_user_negative_dict(all_movie_ids, user_positive_dict):
    """
    Build a dictionary of negative (not rated/disliked) items per user.
    
    Args:
        all_movie_ids: List of all movie IDs
        user_positive_dict: Dictionary of positive items per user
        
    Returns:
        dict: user_id -> list of negative movie_ids
    """
    user_neg = {}
    all_movies_set = set(all_movie_ids)
    for uid, pos_movies in user_positive_dict.items():
        user_neg[uid] = list(all_movies_set - pos_movies)
    return user_neg


# Evaluation Function
def test(model, dataloader, criterion, all_movie_ids, movie_embedding_lookup, model_type="B", 
         k=EVALUATION_K, device='cuda', num_negatives=NUM_NEGATIVES):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for evaluation set
        criterion: Loss function
        all_movie_ids: List of all movie IDs
        movie_embedding_lookup: Dictionary mapping movie_id to embeddings
        model_type: Model type ('A', 'B', 'C', or 'D')
        k: Cutoff for metrics computation
        device: Device to use (cuda/cpu)
        num_negatives: Number of negative samples per positive sample
        
    Returns:
        tuple: (metrics dict, average loss)
    """
    model.eval()
    metrics = {'hr': 0, 'mrr': 0, 'ndcg': 0, 'precision': 0}
    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            user_ids = batch['user_id'].to(device)
            true_movie_ids = batch['movie_id'].to(device)
            true_embeddings = batch['embedding'].to(device)

            batch_size = user_ids.size(0)

            for i in range(batch_size):
                uid = user_ids[i].item()
                true_mid = true_movie_ids[i].item()
                true_emb = true_embeddings[i]

                # Sample negatives for this user
                neg_ids = [m for m in all_movie_ids if m != true_mid]
                neg_sample = random.sample(neg_ids, k=num_negatives)

                candidate_ids = [true_mid] + neg_sample
                random.shuffle(candidate_ids)
                label = candidate_ids.index(true_mid)

                # Build inputs
                uids = torch.tensor([uid] * len(candidate_ids), dtype=torch.long, device=device)
                mids = torch.tensor(candidate_ids, dtype=torch.long, device=device)
                
                # Handle embeddings based on model type
                if model_type == "A":
                    # SimpleCF doesn't use embeddings, pass dummy
                    embs = torch.zeros(len(candidate_ids), 1, EMBEDDING_DIMENSION, device=device)
                else:
                    # LLMRec uses actual embeddings
                    embs = torch.stack([movie_embedding_lookup[mid].to(device) for mid in candidate_ids])

                logits = model(uids, mids, embs).squeeze()

                # Compute metrics
                for name, fn in zip(metrics.keys(), [hr_at_k, mrr_at_k, ndcg_at_k, precision_at_k]):
                    value, _ = fn(k, logits.unsqueeze(0), torch.tensor([label], device=device))
                    metrics[name] += value

                # Compute loss
                labels_tensor = torch.zeros(len(candidate_ids), device=device)
                labels_tensor[label] = 1.0
                loss = criterion(logits, labels_tensor)
                total_loss += loss.item()
                count += 1

    avg_loss = total_loss / count if count > 0 else 0
    return {k: v / count for k, v in metrics.items()}, avg_loss


# Training Functions
def train_one_epoch(model, dataloader, optimizer, criterion, all_movie_ids, movie_embedding_lookup, 
                    user_negative_dict, model_type="B", device="cuda", num_negatives=NUM_NEGATIVES):
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model to train
        dataloader: DataLoader for training set
        optimizer: Optimizer (SGD, Adam, etc.)
        criterion: Loss function
        all_movie_ids: List of all movie IDs
        movie_embedding_lookup: Dictionary mapping movie_id to embeddings
        user_negative_dict: Dictionary of negative items per user
        model_type: Model type ('A', 'B', 'C', or 'D')
        device: Device to use (cuda/cpu)
        num_negatives: Number of negative samples per positive sample
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0

    for batch in dataloader:
        user_ids = batch['user_id'].to(device)
        pos_movie_ids = batch['movie_id'].to(device)
        pos_embeddings = batch['embedding'].to(device)

        # Create lists to accumulate batch with negatives
        all_user_ids = []
        all_movie_ids_batch = []
        all_embeddings = []
        all_labels = []

        for i in range(len(user_ids)):
            uid = user_ids[i].item()
            pos_mid = pos_movie_ids[i]
            pos_emb = pos_embeddings[i]

            # Positive sample
            all_user_ids.append(uid)
            all_movie_ids_batch.append(pos_mid)
            all_embeddings.append(pos_emb)
            all_labels.append(1.0)

            # Negative samples with error handling
            if uid in user_negative_dict and user_negative_dict[uid]:
                neg_sampled_ids = random.sample(user_negative_dict[uid], 
                                                 k=min(num_negatives, len(user_negative_dict[uid])))
            else:
                neg_sampled_ids = random.sample(all_movie_ids, k=num_negatives)

            for neg_mid in neg_sampled_ids:
                neg_emb = movie_embedding_lookup[neg_mid].to(device)
                all_user_ids.append(uid)
                all_movie_ids_batch.append(neg_mid)
                all_embeddings.append(neg_emb)
                all_labels.append(0.0)

        # Convert to tensors
        user_tensor = torch.tensor(all_user_ids, dtype=torch.long, device=device).unsqueeze(1)
        movie_tensor = torch.tensor(all_movie_ids_batch, dtype=torch.long, device=device).unsqueeze(1)
        embedding_tensor = torch.stack(all_embeddings).to(device).unsqueeze(1)
        label_tensor = torch.tensor(all_labels, dtype=torch.float, device=device)

        optimizer.zero_grad()
        
        # Forward pass based on model type
        if model_type == "A":
            # SimpleCF: pass None for embeddings (uses learned embeddings)
            outputs = model(user_tensor, movie_tensor, None).squeeze()
        else:
            # LLMRec: pass actual embeddings
            outputs = model(user_tensor, movie_tensor, embedding_tensor).squeeze()

        loss = criterion(outputs, label_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(model, train_loader, val_loader, movie_embedding_lookup, all_movie_ids, model_type="B",
          device="cuda", epochs=TRAIN_EPOCHS, lr=LEARNING_RATE):
    """
    Train model with early stopping and learning rate scheduling.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        movie_embedding_lookup: Dictionary mapping movie_id to embeddings
        all_movie_ids: List of all movie IDs
        model_type: Model type ('A', 'B', 'C', or 'D')
        device: Device to use (cuda/cpu)
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        None (logs results)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, 
                                   patience=SCHEDULER_PATIENCE, verbose=False)
    criterion = nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    total_train_loss = 0
    total_val_loss = 0
    count = 0
    user_positive_dict = build_user_positive_dict(train_loader.dataset)
    user_negative_dict = build_user_negative_dict(all_movie_ids, user_positive_dict)

    logger.info(f"Starting training (Model Type {model_type}) for {epochs} epochs with lr={lr}")

    for epoch in range(epochs):
        count += 1
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, all_movie_ids, 
                                     movie_embedding_lookup, user_negative_dict, model_type=model_type, 
                                     device=device)
        total_train_loss += train_loss
        val_metrics, val_loss = test(model, val_loader, criterion, all_movie_ids, 
                                     movie_embedding_lookup, model_type=model_type, device=device)
        scheduler.step(val_loss)
        total_val_loss += val_loss

        logger.info(f"Epoch {epoch+1}/{epochs}:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Val Metrics: HR@{EVALUATION_K}={val_metrics['hr']:.4f}, MRR@{EVALUATION_K}={val_metrics['mrr']:.4f}, " 
                   f"NDCG@{EVALUATION_K}={val_metrics['ndcg']:.4f}, Precision@{EVALUATION_K}={val_metrics['precision']:.4f}")

        if not early_stopper.check_loss(val_loss):
            break

    avg_train_loss = total_train_loss / count
    avg_val_loss = total_val_loss / count

    logger.info(f"Training complete!")
    logger.info(f"Average Train Loss: {avg_train_loss:.4f}")
    logger.info(f"Average Validation Loss: {avg_val_loss:.4f}")

    return


if __name__ == "__main__":
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU (training will be slow)")

    logger.info(f"=" * 60)
    logger.info(f"Training Configuration:")
    logger.info(f"  Model Type: {MODEL_TYPE}")
    model_descriptions = {
        "A": "SimpleCF - Collaborative filtering",
        "B": "LLMRec with basic descriptions",
        "C": "LLMRec with recommendation-driven descriptions",
        "D": "LLMRec with combined descriptions"
    }
    logger.info(f"  Description: {model_descriptions.get(MODEL_TYPE)}")
    logger.info(f"  Embeddings File: {get_embeddings_loader()}")
    logger.info(f"=" * 60)

    logger.info("Loading datasets...")
    _, _, _, train_loader, val_loader, test_loader = createDataset()

    # Create model based on type
    model = get_model(MODEL_TYPE)
    model.to(device)

    # Load embeddings
    embeddings_file = get_embeddings_loader()
    logger.info(f"Loading embeddings from {embeddings_file}...")
    try:
        with open(embeddings_file, "r") as f:
            embeddings = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}. "
                                f"Please run prompting.py to generate embeddings.")
    
    # Validate and convert embeddings to tensors
    embeddings_tensors = {}
    nan_count = 0
    for k, v in embeddings.items():
        v_array = np.array(v)
        if any(not np.isfinite(x) for x in v_array):
            nan_count += 1
            v_array = np.nan_to_num(v_array, nan=0.0, posinf=0.0, neginf=0.0)
        embeddings_tensors[int(k)] = torch.tensor(v_array, dtype=torch.float).to(device)
    
    if nan_count > 0:
        logger.warning(f"Found and cleaned {nan_count} embeddings with NaN/inf values")
    
    embeddings = embeddings_tensors
    logger.info(f"Loaded {len(embeddings)} embeddings onto {device}")

    # Train model
    logger.info("Starting model training...")
    train(model, train_loader, val_loader, embeddings, list(embeddings.keys()), 
          model_type=MODEL_TYPE, device=device)

    # Test Model
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 60)
    testing, avg_test_loss = test(model, test_loader, nn.BCEWithLogitsLoss(), 
                                   list(embeddings.keys()), embeddings, model_type=MODEL_TYPE, device=device)
    
    logger.info(f"\nTest Results for Model Type {MODEL_TYPE}:")
    logger.info(f"  HR@{EVALUATION_K}: {testing['hr']:.4f}")
    logger.info(f"  MRR@{EVALUATION_K}: {testing['mrr']:.4f}")
    logger.info(f"  NDCG@{EVALUATION_K}: {testing['ndcg']:.4f}")
    logger.info(f"  Precision@{EVALUATION_K}: {testing['precision']:.4f}")
    logger.info(f"  Average Test Loss: {avg_test_loss:.4f}")
    
    # Save test results to JSON file
    results = {
        "model_type": MODEL_TYPE,
        "model_description": {
            "A": "SimpleCF - Collaborative filtering",
            "B": "LLMRec with basic descriptions",
            "C": "LLMRec with recommendation-driven descriptions",
            "D": "LLMRec with combined descriptions"
        }.get(MODEL_TYPE, "Unknown"),
        "test_metrics": {
            f"HR@{EVALUATION_K}": float(testing['hr']),
            f"MRR@{EVALUATION_K}": float(testing['mrr']),
            f"NDCG@{EVALUATION_K}": float(testing['ndcg']),
            f"Precision@{EVALUATION_K}": float(testing['precision']),
        },
        "average_test_loss": float(avg_test_loss),
        "timestamp": str(pd.Timestamp.now())
    }
    
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Test results saved to test_results.json")
    logger.info("=" * 60)
