"""
Neural network models for recommendation systems
- SimpleCF: Simple collaborative filtering without embeddings
- LLMRec: Multi-head attention based model with text embeddings
"""

import torch
import torch.nn as nn
import logging

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
            logger.debug(f"New best loss: {loss:.4f}")
            return True
        else:
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
        
    def forward(self, user_id, movie_id, mov_embedding=None):
        """
        Forward pass computing interaction scores.
        
        Args:
            user_id: User IDs (batch_size,) or (batch_size, 1)
            movie_id: Movie IDs (batch_size,) or (batch_size, 1)
            mov_embedding: Unused (for compatibility with LLMRec interface)
            
        Returns:
            Predicted interaction scores (batch_size,) or (batch_size, 1)
        """
        # Ensure tensors are 1D for embedding lookup
        user_id = user_id.squeeze()
        movie_id = movie_id.squeeze()
        
        user_emb = self.user_embedding(user_id)
        movie_emb = self.movie_embedding(movie_id)
        
        user_bias = self.user_bias(user_id).squeeze()
        movie_bias = self.movie_bias(movie_id).squeeze()
        
        # Concatenate embeddings
        x = torch.cat([user_emb, movie_emb], dim=-1)
        
        # Output
        output = self.linear(x).squeeze() + user_bias + movie_bias + self.global_bias
        
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
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=2, 
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
        self.dropout = nn.Dropout(0.4)

        self.linear2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)

        self.linear3 = nn.Linear(128, 1)

    def forward(self, user_id, movie_id, mov_embedding):
        """
        Forward pass computing interaction scores with attention.
        
        Args:
            user_id: User IDs (batch_size, 1)
            movie_id: Movie IDs (batch_size, 1) - unused but kept for interface compatibility
            mov_embedding: Movie text embeddings (batch_size, 1, embedding_dim)
            
        Returns:
            Predicted interaction scores (batch_size,) or (batch_size, 1)
        """
        user_emb = self.user_embedding(user_id)  # (batch_size, 1, embedding_dim)

        # Project movie embeddings
        mov_proj = self.project_movie(mov_embedding)  # (batch_size, 1, embedding_dim)

        # Apply attention: user embedding attends to movie embeddings
        attn_output, attn_weights = self.attention(user_emb, mov_proj, mov_proj)
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

        x = self.linear3(x)

        return x
