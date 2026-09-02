"""
Neural network models for recommendation systems
- SimpleCF: Simple collaborative filtering without embeddings
- LLMRec: Multi-head attention based model with text embeddings
"""

import torch
import torch.nn as nn
import logging

from config import ATTENTION_HEADS, DROPOUT_RATE

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping mechanism to prevent overfitting during training."""
    
    def __init__(self, patience):
        """
        Args:
            patience: Number of epochs with no improvement after which training stops
        """
        self.best_loss = float("inf")
        self.counter = 0
        self.patience = patience
        self.improved = False

    def check_loss(self, loss):
        """
        Check if loss has improved. Returns True to continue training, False to stop.
        
        Args:
            loss: Current validation loss
            
        Returns:
            bool: True if training should continue, False if should stop
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            self.improved = True
            logger.debug(f"New best loss: {loss:.4f}")
            return True
        else:
            self.improved = False
            self.counter += 1
            logger.debug(f"No improvement. Patience: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
                return False
        return True


class SimpleCF(nn.Module):
    """
    Simple Collaborative Filtering model without embeddings (Model Type A).
    Uses user and movie embeddings learned through matrix factorization.
    """
    
    def __init__(self, num_users, num_movies, embedding_dim):
        """
        Args:
            num_users: Number of unique users
            num_movies: Number of unique movies
            embedding_dim: Dimension of user/movie embeddings
        """
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_dim = embedding_dim
        
        # Learnable user and movie embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # Biases
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        
        # Global bias
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Output layer
        self.linear = nn.Linear(embedding_dim * 2, 1)
        
    def forward(self, user_id, movie_id, mov_embedding=None, return_attention=False):
        """
        Forward pass computing interaction scores.
        
        Args:
            user_id: User IDs (batch_size,) or (batch_size, 1)
            movie_id: Movie IDs (batch_size,) or (batch_size, 1)
            mov_embedding: Unused (for compatibility with LLMRec interface)
            return_attention: Unused; SimpleCF has no attention layer
            
        Returns:
            Predicted interaction scores, always shape (batch_size,)
        """
        # reshape(-1) rather than squeeze(): squeeze() would also drop the batch
        # dimension when the batch happens to contain a single example.
        user_id = user_id.reshape(-1)
        movie_id = movie_id.reshape(-1)

        user_emb = self.user_embedding(user_id)
        movie_emb = self.movie_embedding(movie_id)

        user_bias = self.user_bias(user_id).reshape(-1)
        movie_bias = self.movie_bias(movie_id).reshape(-1)

        # Concatenate embeddings
        x = torch.cat([user_emb, movie_emb], dim=-1)

        # Output
        output = self.linear(x).reshape(-1) + user_bias + movie_bias + self.global_bias

        if return_attention:
            return output, None
        return output


class LLMRec(nn.Module):
    """
    LLMRec: Neural Recommendation Model with Multi-Head Attention and Text Embeddings
    Uses attention mechanism to combine user embeddings with movie text embeddings.
    
    Model Types:
    - Type B: LLMRec with basic text descriptions
    - Type C: LLMRec with recommendation-driven descriptions
    - Type D: LLMRec with combined basic and recommendation-driven descriptions
    """
    
    def __init__(self, num_users, num_movies, embedding_dim):
        """
        Args:
            num_users: Number of unique users
            num_movies: Number of unique movies
            embedding_dim: Dimension of embeddings (must match movie embeddings dimension)
        """
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_dim = embedding_dim

        # Attention layer - attends from user to movie embeddings
        # The user embedding is the query; the movie's description views are the
        # keys/values. With one view (types B/C/D) the softmax is over a sequence
        # of length 1, so it is always 1.0 and this degenerates to a learned
        # linear projection. With several views (type E) the softmax is a real
        # choice: it learns, per user, which kind of description to trust.
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=ATTENTION_HEADS,
            batch_first=True
        )

        # Movie embedding projection
        self.project_movie = nn.Linear(embedding_dim, embedding_dim)
        
        # Learnable user embedding
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        # Feedforward network
        self.linear1 = nn.Linear(embedding_dim * 2, 256)
        self.ln1 = nn.LayerNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)

        self.linear2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)

        self.linear3 = nn.Linear(128, 1)

    def forward(self, user_id, movie_id, mov_embedding, return_attention=False):
        """
        Forward pass computing interaction scores with attention.

        Args:
            user_id: User IDs (batch_size, 1)
            movie_id: Movie IDs (batch_size, 1) - unused but kept for interface compatibility
            mov_embedding: Movie text embeddings (batch_size, num_views, embedding_dim).
                           num_views is 1 for types B/C/D and >1 for type E.
            return_attention: If True, also return the per-view attention weights

        Returns:
            Predicted interaction scores, shape (batch_size,).
            If return_attention, a tuple (scores, weights) where weights has shape
            (batch_size, num_views) and each row sums to 1.
        """
        if user_id.dim() == 1:
            user_id = user_id.unsqueeze(1)
        if mov_embedding.dim() == 2:
            mov_embedding = mov_embedding.unsqueeze(1)
        user_emb = self.user_embedding(user_id)  # (batch_size, 1, embedding_dim)

        # Project movie embeddings
        mov_proj = self.project_movie(mov_embedding)  # (batch_size, 1, embedding_dim)

        # Apply attention: the user embedding attends over the movie's views
        attn_output, attn_weights = self.attention(user_emb, mov_proj, mov_proj)
        # attn_weights: (batch_size, 1 query, num_views) -> (batch_size, num_views)
        attn_weights = attn_weights.squeeze(1)
        attn_output = attn_output.squeeze(1)  # (batch_size, embedding_dim)
        user_emb = user_emb.squeeze(1)  # (batch_size, embedding_dim)

        # Concatenate user and attended movie embedding
        x = torch.cat([user_emb, attn_output], dim=-1)

        # Feedforward pass with LayerNorm and dropout
        x = self.linear1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear3(x).reshape(-1)

        if return_attention:
            return x, attn_weights
        return x
