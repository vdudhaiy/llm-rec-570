import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import json
import pandas as pd
from collections import defaultdict
import numpy as np
import logging
from config import (
    RATINGS_FILE, MOVIES_DESC_FILE, EMBEDDINGS_FILE,
    EMBEDDING_MODEL, EMBEDDING_DIM, TEST_SIZE, VAL_SIZE,
    POSITIVE_RATING_THRESHOLD, RANDOM_SEED, VERBOSE, SELECTED_EMBEDDINGS_FILE,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, SUBSAMPLE_FRAC, get_device
)

# Set seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Configure logging
logging.basicConfig(level=logging.INFO if VERBOSE else logging.WARNING)
logger = logging.getLogger(__name__)

# Custom MovieLens Dataset
class MovieLens(Dataset):
    def __init__(self, data, movies, embeddings, expected_embedding_dim=EMBEDDING_DIM):
        self.data = data
        self.movies = movies
        self.embeddings = embeddings
        self.expected_embedding_dim = expected_embedding_dim
        self._validate_embeddings()

        # Pull the three columns we need out of pandas once, up front.
        # The previous version called data.iloc[index] three times per sample,
        # which dominated the runtime of every epoch.
        self.user_ids = torch.as_tensor(data['userId'].to_numpy(), dtype=torch.long) - 1
        self.movie_ids = torch.as_tensor(data['movieId'].to_numpy(), dtype=torch.long)
        self.ratings = torch.as_tensor(data['rating'].to_numpy(), dtype=torch.float)

        # Dense embedding matrix indexed by movie id, so lookup is a tensor index
        # instead of a Python dict hit per sample. Unknown movies stay all-zero.
        max_movie_id = int(self.movie_ids.max().item())
        if embeddings:
            max_movie_id = max(max_movie_id, max(int(k) for k in embeddings))
        self.embedding_matrix = torch.zeros(max_movie_id + 1, expected_embedding_dim, dtype=torch.float)
        for mid, vec in embeddings.items():
            self.embedding_matrix[int(mid)] = torch.as_tensor(vec, dtype=torch.float)

        known = torch.as_tensor(sorted(int(k) for k in embeddings), dtype=torch.long)
        self.has_embedding = torch.zeros(max_movie_id + 1, dtype=torch.bool)
        self.has_embedding[known] = True
        missing = int((~self.has_embedding[self.movie_ids]).sum().item())
        if missing:
            logger.warning(
                f"{missing}/{len(self.movie_ids)} interactions reference movies with no "
                f"embedding; those fall back to a zero vector"
            )

    def _validate_embeddings(self):
        """Validate that embeddings have correct dimension"""
        if not self.embeddings:
            raise ValueError("Embeddings dictionary is empty")
        
        # Check first embedding
        first_emb = next(iter(self.embeddings.values()))
        if isinstance(first_emb, list):
            emb_dim = len(first_emb)
        else:
            emb_dim = first_emb.shape[0] if hasattr(first_emb, 'shape') else len(first_emb)
        
        if emb_dim != self.expected_embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.expected_embedding_dim}, got {emb_dim}"
            )
        logger.info(f"Validated embeddings: {len(self.embeddings)} embeddings with dimension {emb_dim}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        movie_id = self.movie_ids[index]

        return {
            'user_id': self.user_ids[index],
            'movie_id': movie_id,
            'embedding': self.embedding_matrix[movie_id],
            'rating': self.ratings[index]
        }

def createDataset(loaded=True):
    """Create PyTorch datasets and dataloaders with validation"""
    try:
        ratings = pd.read_csv(RATINGS_FILE, delimiter='::', names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')
        movies = pd.read_csv(MOVIES_DESC_FILE, delimiter='::', names=['movieId', 'title', 'year', 'genres','description'], engine='python')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file not found: {e}")
    
    if 0 < SUBSAMPLE_FRAC < 1.0:
        ratings = ratings.sample(frac=SUBSAMPLE_FRAC, random_state=RANDOM_SEED).reset_index(drop=True)
        logger.info(f"Subsampled ratings to {SUBSAMPLE_FRAC:.3f} of the file: {len(ratings)} rows")

    logger.info(f"Loaded {len(movies)} movies and {len(ratings)} ratings")
    logger.info(f"Movies sample:\n{movies.head()}")
    logger.info(f"Ratings sample:\n{ratings.head()}")

    if not loaded:
        logger.info(f"Generating embeddings using {EMBEDDING_MODEL}...")
        encoder = SentenceTransformer(EMBEDDING_MODEL, device=str(get_device()))
        embeddings = getAllEmbeddings(encoder, movies.to_dict('records'))
    else:
        logger.info(f"Loading pre-computed embeddings from {SELECTED_EMBEDDINGS_FILE}...")
        try:
            with open(SELECTED_EMBEDDINGS_FILE, "r") as f:
                embeddings = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Embeddings file not found: {SELECTED_EMBEDDINGS_FILE}. Set loaded=False to generate them.")

    logger.info("Embeddings loaded successfully")

    # Validate and clean embeddings
    embeddings_clean = {}
    for k, v in embeddings.items():
        if isinstance(v, list):
            v_array = np.array(v)
        else:
            v_array = np.array(v) if hasattr(v, '__iter__') else v
        
        if any(not np.isfinite(x) for x in v_array):
            logger.warning(f"Found NaN/inf in embedding for movie {k}, replacing with zeros")
            v_array = np.nan_to_num(v_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        embeddings_clean[int(k)] = torch.tensor(v_array, dtype=torch.float)
    
    embeddings = embeddings_clean
    logger.info(f"Cleaned embeddings: {len(embeddings)} valid embeddings")

    # Train/val/test split with validation
    train_data, test_data = train_test_split(ratings, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_data, val_data = train_test_split(train_data, test_size=VAL_SIZE, random_state=RANDOM_SEED)

    # Verify splits are disjoint
    train_set = set(train_data.index)
    val_set = set(val_data.index)
    test_set = set(test_data.index)
    
    assert len(train_set & val_set) == 0, "Train/val sets overlap!"
    assert len(train_set & test_set) == 0, "Train/test sets overlap!"
    assert len(val_set & test_set) == 0, "Val/test sets overlap!"
    
    logger.info(f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    train_dataset = MovieLens(train_data, movies, embeddings)
    val_dataset = MovieLens(val_data, movies, embeddings)
    test_dataset = MovieLens(test_data, movies, embeddings)

    logger.info("Datasets created successfully")

    pin = PIN_MEMORY and get_device().type == "cuda"
    loader_kwargs = {"batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS, "pin_memory": pin}
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    logger.info("DataLoaders created successfully")

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader

# Generate Movie Embeddings to set up MovieLens dataset
def getAllEmbeddings(encoder, movies, output_file=EMBEDDINGS_FILE, batch_size=256):
  """Encode every movie's text in one batched GPU pass instead of one call per movie."""
  texts = []
  movie_ids = []
  for mov in movies:
    parts = (
        f"Movie title: {mov['title']} ({mov['year']}).",
        f"Genres: {(mov['genres'])}. ",
        f"Description: {mov['description']}"
    )
    texts.append('. '.join([part for part in parts if part]).strip())
    movie_ids.append(int(mov["movieId"]))

  encoded = encoder.encode(texts, batch_size=batch_size, show_progress_bar=VERBOSE,
                           convert_to_numpy=True)
  embeddings = {mid: vec.tolist() for mid, vec in zip(movie_ids, encoded)}

  with open(output_file, "w") as f:
    json.dump(embeddings, f)

  logger.info(f"Embeddings saved to {output_file}")

  return embeddings

def build_user_positive_dict(dataset, rating_threshold=POSITIVE_RATING_THRESHOLD):
    """Build dictionary of positive items per user based on rating threshold"""
    user_pos = defaultdict(set)
    for data in dataset:
        uid = int(data['user_id']) 
        mid = int(data['movie_id'])  
        rating = float(data['rating'])
        if rating >= rating_threshold:
            user_pos[uid].add(mid)
    logger.info(f"Built positive dictionary for {len(user_pos)} users (threshold={rating_threshold})")
    return user_pos

def build_user_negative_dict(all_movie_ids, user_positive_dict):
    user_neg = {}
    all_movies_set = set(all_movie_ids)
    for uid, pos_movies in user_positive_dict.items():
        user_neg[uid] = list(all_movies_set - pos_movies)
    logger.info(f"Built negative dictionary for {len(user_neg)} users")
    return user_neg
