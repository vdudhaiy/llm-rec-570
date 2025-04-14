import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import numpy as np
import random
from data_processing import *

# Early Stopping Mechanism
class EarlyStopping:
    def __init__(self, patience):
        self.loss = float("inf")
        self.counter = 0
        self.patience = patience

    def check_loss(self, loss):
        if loss < self.loss:
            self.loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return False
        return True

# Custom LLMRec Model
class LLMRec(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim):
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_dim = embedding_dim

        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=2, batch_first=True)

        self.project_movie = nn.Linear(embedding_dim, embedding_dim)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        self.linear1 = nn.Linear(embedding_dim * 2, 256)
        self.ln1 = nn.LayerNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

        self.linear2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)

        self.linear3 = nn.Linear(128, 1)

    def forward(self, user_id, movie_id, mov_embedding):
        user_emb = self.user_embedding(user_id)  # (batch_size, 1, embedding_dim)

        # Project movie embeddings
        mov_proj = self.project_movie(mov_embedding)  # (batch_size, top_k, embedding_dim)

        # Expand user embedding for attention query
        query = user_emb
        
        # Apply attention: query attends to movie embeddings
        attn_output, attn_weights = self.attention(query, mov_proj, mov_proj)
        attn_output = attn_output.squeeze(1)  # (batch_size, embedding_dim)
        user_emb = user_emb.squeeze(1)  # (batch_size, embedding_dim)

        # Concatenate user and attended movie embedding
        x = torch.cat([user_emb, attn_output], dim=-1)

        # Feedforward pass with LayerNorm
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
    
# Evaluation Metrics
def hr_at_k(k, logits, labels):
    k = min(k, logits.shape[1])  # safeguard
    # probabilities = torch.sigmoid(logits)  # Apply sigmoid to logits
    _, indices = torch.topk(logits, k=k)
    is_correct_pred = indices == labels.unsqueeze(1)
    is_correct_pred = is_correct_pred.to(torch.float32)
    return int(is_correct_pred.sum()), labels.shape[0]

def mrr_at_k(k, logits, labels):
    k = min(k, logits.shape[1])  # safeguard
    # probabilities = torch.sigmoid(logits)  # Apply sigmoid to logits
    _, indices = torch.topk(logits, k=k)
    is_correct_pred = indices == labels.unsqueeze(1)
    return int((1 / (is_correct_pred.nonzero()[:, 1] + 1).type(torch.float)).sum()), labels.shape[0]

def ndcg_at_k(k, logits, labels):
    k = min(k, logits.shape[1])  # safeguard
    # probabilities = torch.sigmoid(logits)  # Apply sigmoid to logits
    _, indices = torch.topk(logits, k=k)
    is_correct_pred = indices == labels.unsqueeze(1)
    is_correct_pred = is_correct_pred.to(torch.float32)
    discount = torch.log2(torch.arange(2, k + 2, device=logits.device))
    return int((is_correct_pred / discount).sum()), labels.shape[0]

def precision_at_k(k, logits, labels):
    k = min(k, logits.shape[1])  # safeguard
    # probabilities = torch.sigmoid(logits)  # Apply sigmoid to logits
    _, indices = torch.topk(logits, k=k)
    is_correct_pred = indices == labels.unsqueeze(1)
    precision = is_correct_pred.to(torch.float32).sum() / k
    return float(precision), labels.shape[0]


# Evaluation Function
def test(model, dataloader, criterion, all_movie_ids, movie_embedding_lookup, k=10, device='cuda', num_negatives=9):
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
                embs = torch.stack([movie_embedding_lookup[mid].to(device) for mid in candidate_ids])

                logits = model(uids, mids, embs).squeeze()  # shape [num_candidates]

                # Compute metrics
                for name, fn in zip(metrics.keys(), [hr_at_k, mrr_at_k, ndcg_at_k, precision_at_k]):
                    value, _ = fn(k, logits.unsqueeze(0), torch.tensor([label], device=device))
                    metrics[name] += value

                # Updated loss computation: use full candidate list
                labels_tensor = torch.zeros(len(candidate_ids), device=device)
                labels_tensor[label] = 1.0  # Only true item is 1
                loss = criterion(logits, labels_tensor)
                total_loss += loss.item()
                count += 1

    avg_loss = total_loss / count
    return {k: v / count for k, v in metrics.items()}, avg_loss

# Training Functions
def train_one_epoch(model, dataloader, optimizer, criterion, all_movie_ids,  movie_embedding_lookup, user_negative_dict, device="cuda", num_negatives=9):
    model.train()
    total_loss = 0

    for batch in dataloader:
        user_ids = batch['user_id'].to(device)
        pos_movie_ids = batch['movie_id'].to(device)
        pos_embeddings = batch['embedding'].to(device)

        # Create lists to accumulate new batch with negatives
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
            all_labels.append(1.0)  # positive label

            #  Negative samples with error handling
            if uid in user_negative_dict and user_negative_dict[uid]:
                neg_sampled_ids = random.sample(user_negative_dict[uid], k=min(num_negatives, len(user_negative_dict[uid])))
            else:
                # Handle cases where no negative samples are available
                neg_sampled_ids = random.sample(all_movie_ids, k=num_negatives)

            for neg_mid in neg_sampled_ids:
                neg_emb = movie_embedding_lookup[neg_mid].to(device)
                all_user_ids.append(uid)
                all_movie_ids_batch.append(neg_mid)
                all_embeddings.append(neg_emb)
                all_labels.append(0.0)  # negative label

        # Convert everything to tensors
        user_tensor = torch.tensor(all_user_ids, dtype=torch.long, device=device).unsqueeze(1)
        movie_tensor = torch.tensor(all_movie_ids_batch, dtype=torch.long, device=device).unsqueeze(1)
        embedding_tensor = torch.stack(all_embeddings).to(device).unsqueeze(1)
        label_tensor = torch.tensor(all_labels, dtype=torch.float, device=device)

        optimizer.zero_grad()
        outputs = model(user_tensor, movie_tensor, embedding_tensor).squeeze()

        # print("outputs:", outputs[:10])
        # print("labels:", label_tensor[:10])
        # print("Output shape:", outputs.shape)
        # print("Label shape:", label_tensor.shape)

        loss = criterion(outputs, label_tensor)
        # print("Loss:", loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(model, train_loader, val_loader, movie_embedding_lookup, all_movie_ids, device="cuda", epochs=50, lr=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopping(patience=10)
    total_train_loss = 0
    total_val_loss = 0
    count = 0
    user_positive_dict = build_user_positive_dict(train_loader.dataset)
    user_negative_dict = build_user_negative_dict(all_movie_ids, user_positive_dict)

    for epoch in range(epochs):
        count += 1
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, all_movie_ids,  movie_embedding_lookup, user_negative_dict, device=device)
        total_train_loss += train_loss
        val_metrics, val_loss = test(model, val_loader, criterion, all_movie_ids, movie_embedding_lookup, device=device)
        scheduler.step(val_loss)
        total_val_loss += val_loss

        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Metrics: HR@10={val_metrics['hr']:.4f}, MRR@10={val_metrics['mrr']:.4f}, "
              f"NDCG@10={val_metrics['ndcg']:.4f}, Precision@10={val_metrics['precision']:.4f}")

        if not early_stopper.check_loss(val_loss):
            print("Early stopping triggered!")
            break

    avg_train_loss = total_train_loss / count
    avg_val_loss = total_val_loss / count

    print(f"Average Train Loss: {avg_train_loss:.4f}")
    print(f"Average Validation Loss: {avg_val_loss:.4f}")

    return

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, _, train_loader, val_loader, test_loader = createDataset()

    model = LLMRec(num_users=6040, num_movies=3952, embedding_dim=384)
    model.to(device)

    embedding_file = "movie_embeddings.json" # If using enhanced embeddings, change file name

    with open(embedding_file, "r") as f:
        embeddings = json.load(f)
    for k, v in embeddings.items():
        if any(not np.isfinite(x) for x in v):  # If any values are NaN or inf
            print(f"Warning: Found NaN or inf in embedding for movie ID {k}")
        embeddings[k] = [0.0 if not np.isfinite(x) else x for x in v]  # Replace with 0.0
    embeddings = {int(k): torch.tensor(v, dtype=torch.float) for k, v in embeddings.items()}

    # Train Custom LLMRec Model
    train(model, train_loader, val_loader, embeddings, list(embeddings.keys()), device, epochs=50, lr=1e-4)

    # Test Model
    testing, avg_test_loss = test(model, test_loader, nn.BCEWithLogitsLoss(), list(embeddings.keys()), embeddings)
    print(f"  Test Metrics: HR@10={testing['hr']:.4f}, MRR@10={testing['mrr']:.4f}, "
                f"NDCG@10={testing['ndcg']:.4f}, Precision@10={testing['precision']:.4f}")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
