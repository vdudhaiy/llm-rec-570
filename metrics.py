"""
Evaluation metrics for recommendation models
Computes ranking-based metrics: HR@k, MRR@k, NDCG@k

Every function takes a batch of candidate scores and the index of the true item
within those candidates, and returns (sum over the batch, batch size) so the
caller can accumulate across batches and divide once at the end.

Note: these are *float* sums. Rounding them to integers (as an earlier version
did) collapses MRR and NDCG onto HR@1, because a single-positive reciprocal rank
or discounted gain is always in (0, 1].

Precision@k is deliberately absent. Each query here has exactly one relevant
item, so Precision@k == HR@k / k identically - it is a rescaling of HR@k, not an
independent measurement. Derive it from HR@k if you need it to line up with a
table that reports it.
"""

import torch


def _hit_matrix(k, logits, labels):
    """
    Shared helper: rank candidates and mark where the true item landed.

    Args:
        k: Cutoff rank for evaluation
        logits: Predicted scores (batch_size, num_candidates)
        labels: True item indices (batch_size,) - index of the true item in candidates

    Returns:
        tuple: (k actually used, bool tensor (batch_size, k) that is True at the
               position where the true item was ranked)
    """
    k = min(k, logits.shape[1])  # safeguard against k > num_candidates
    _, indices = torch.topk(logits, k=k, dim=1)
    return k, indices == labels.unsqueeze(1)


def hr_at_k(k, logits, labels):
    """
    Hit Rate @ k: Proportion of queries where the true item is in top-k predictions.

    Args:
        k: Cutoff rank for evaluation
        logits: Predicted scores (batch_size, num_candidates)
        labels: True item indices (batch_size,)

    Returns:
        tuple: (number of hits, total batch size)
    """
    _, is_correct_pred = _hit_matrix(k, logits, labels)
    return float(is_correct_pred.sum().item()), labels.shape[0]


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
    k, is_correct_pred = _hit_matrix(k, logits, labels)
    ranks = torch.arange(1, k + 1, device=logits.device, dtype=torch.float32)
    reciprocal = is_correct_pred.to(torch.float32) / ranks
    return float(reciprocal.sum().item()), labels.shape[0]


def ndcg_at_k(k, logits, labels):
    """
    Normalized Discounted Cumulative Gain @ k: Measures ranking quality with position-based discount.

    With exactly one relevant item per query the ideal DCG is 1, so DCG and NDCG
    coincide and no extra normalisation term is needed.

    Args:
        k: Cutoff rank for evaluation
        logits: Predicted scores (batch_size, num_candidates)
        labels: True item indices (batch_size,)

    Returns:
        tuple: (sum of NDCG scores, total batch size)
    """
    k, is_correct_pred = _hit_matrix(k, logits, labels)
    discount = torch.log2(torch.arange(2, k + 2, device=logits.device, dtype=torch.float32))
    gains = is_correct_pred.to(torch.float32) / discount
    return float(gains.sum().item()), labels.shape[0]
