"""
Evaluation metrics for recommendation models
Computes ranking-based metrics: HR@k, MRR@k, NDCG@k, Precision@k
"""

import torch


def hr_at_k(k, logits, labels):
    """
    Hit Rate @ k: Proportion of queries where the true item is in top-k predictions.
    
    Args:
        k: Cutoff rank for evaluation
        logits: Predicted scores (batch_size, num_candidates)
        labels: True item indices (batch_size,) - index of the true item in candidates
        
    Returns:
        tuple: (number of hits, total batch size)
    """
    k = min(k, logits.shape[1])  # safeguard against k > num_candidates
    _, indices = torch.topk(logits, k=k)
    is_correct_pred = indices == labels.unsqueeze(1)
    is_correct_pred = is_correct_pred.to(torch.float32)
    return int(is_correct_pred.sum()), labels.shape[0]


def mrr_at_k(k, logits, labels):
    """
    Mean Reciprocal Rank @ k: Average of 1/rank for correctly ranked items in top-k.
    
    Args:
        k: Cutoff rank for evaluation
        logits: Predicted scores (batch_size, num_candidates)
        labels: True item indices (batch_size,)
        
    Returns:
        tuple: (sum of reciprocal ranks, total batch size)
    """
    k = min(k, logits.shape[1])  # safeguard against k > num_candidates
    _, indices = torch.topk(logits, k=k)
    is_correct_pred = indices == labels.unsqueeze(1)
    return int((1 / (is_correct_pred.nonzero()[:, 1] + 1).type(torch.float)).sum()), labels.shape[0]


def ndcg_at_k(k, logits, labels):
    """
    Normalized Discounted Cumulative Gain @ k: Measures ranking quality with position-based discount.
    
    Args:
        k: Cutoff rank for evaluation
        logits: Predicted scores (batch_size, num_candidates)
        labels: True item indices (batch_size,)
        
    Returns:
        tuple: (sum of NDCG scores, total batch size)
    """
    k = min(k, logits.shape[1])  # safeguard against k > num_candidates
    _, indices = torch.topk(logits, k=k)
    is_correct_pred = indices == labels.unsqueeze(1)
    is_correct_pred = is_correct_pred.to(torch.float32)
    discount = torch.log2(torch.arange(2, k + 2, device=logits.device))
    return int((is_correct_pred / discount).sum()), labels.shape[0]


def precision_at_k(k, logits, labels):
    """
    Precision @ k: Proportion of top-k predictions that are correct.
    
    Args:
        k: Cutoff rank for evaluation
        logits: Predicted scores (batch_size, num_candidates)
        labels: True item indices (batch_size,)
        
    Returns:
        tuple: (average precision score, total batch size)
    """
    k = min(k, logits.shape[1])  # safeguard against k > num_candidates
    _, indices = torch.topk(logits, k=k)
    is_correct_pred = indices == labels.unsqueeze(1)
    precision = is_correct_pred.to(torch.float32).sum() / k
    return float(precision), labels.shape[0]
