import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import json
import pandas as pd
from collections import defaultdict

# Custom MovieLens Dataset
class MovieLens(Dataset):
    def __init__(self, data, movies, embeddings):
        self.data = data
        self.movies = movies
        self.embeddings = embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user_id = self.data.iloc[index]['userId'] - 1
        movie_id = self.data.iloc[index]['movieId']
        embedding = self.embeddings.get(movie_id, torch.zeros(384))
        rating = self.data.iloc[index]['rating']

        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'movie_id': torch.tensor(movie_id, dtype=torch.long),
            'embedding': torch.tensor(embedding, dtype=torch.float),
            'rating': torch.tensor(rating, dtype=torch.float)
        }

def createDataset(loaded = True):
    ratings = pd.read_csv('ratings.dat', delimiter='::', names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')
    movies = pd.read_csv('movies_desc.dat', delimiter='::', names=['movieId', 'title', 'year', 'genres','description'], engine='python')

    print(len(movies))
    print(movies.head())
    print(len(ratings))
    print(ratings.head())
    print("Raw Data Read")


    if not loaded:
      encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
      embeddings = getAllEmbeddings(encoder, movies.to_dict('records'))
    else:
      with open("movie_embeddings.json", "r") as f:
        embeddings = json.load(f)

    print("Embeddings generated")

    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    print("Data has been split")

    train_dataset = MovieLens(train_data, movies, embeddings)
    val_dataset = MovieLens(val_data, movies, embeddings)
    test_dataset = MovieLens(test_data, movies, embeddings)

    print("Datasets created")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("DataLoaders created")

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader

# Generate Movie Embeddings to set up MovieLens dataset
def getAllEmbeddings(encoder, movies, filename):
  embeddings = {}
  for mov in movies:
    parts = (
        f"Movie title: {mov['title']} ({mov['year']}).",
        f"Genres: {(mov['genres'])}. ",
        f"Description: {mov['description']}"
    )
    movie_text = '. '.join([part for part in parts if part]).strip()

    embedding = encoder.encode(movie_text)
    embeddings[int(mov["movieId"])] = embedding.tolist()

  with open(f"{filename}.json", "w") as f:
    json.dump(embeddings, f)

  print(f"Embeddings saved to {filename}.json")

  return embeddings

def build_user_positive_dict(dataset):
    user_pos = defaultdict(set)
    for data in dataset:
        uid = int(data['user_id']) 
        mid = int(data['movie_id'])  
        rating = float(data['rating'])
        if rating >= 2:
            user_pos[uid].add(mid)  #add to set instead of overwriting
    return user_pos

def build_user_negative_dict(all_movie_ids, user_positive_dict):
    user_neg = {}
    all_movies_set = set(all_movie_ids)
    for uid, pos_movies in user_positive_dict.items():
        user_neg[uid] = list(all_movies_set - pos_movies)
    return user_neg
