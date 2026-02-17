import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_pickle("data/ranking_dataset.pkl")

# Convert dict features to numpy
X_E_np = np.array([list(x["E"].values()) for x in data])
X_I_np = np.array([list(x["I"].values()) for x in data])
X_A_np = np.array([list(x["A"].values()) for x in data])

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
scaler.fit(np.vstack([X_E_np, X_I_np, X_A_np]))

X_E_np = scaler.transform(X_E_np)
X_I_np = scaler.transform(X_I_np)
X_A_np = scaler.transform(X_A_np)

# Convert to tensors
X_E = torch.tensor(X_E_np, dtype=torch.float32)
X_I = torch.tensor(X_I_np, dtype=torch.float32)
X_A = torch.tensor(X_A_np, dtype=torch.float32)

input_dim = X_E.shape[1]

# -----------------------------
# Stronger Model
# -----------------------------
class RankModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = RankModel(input_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MarginRankingLoss(margin=0.5)

# -----------------------------
# Training
# -----------------------------
epochs = 1000
batch_size = 64
n = len(X_E)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    permutation = torch.randperm(n)

    for i in tqdm(range(0, n, batch_size), desc=f"Epoch {epoch+1}"):
        indices = permutation[i:i+batch_size]

        e = X_E[indices]
        i_ = X_I[indices]
        a = X_A[indices]

        optimizer.zero_grad()

        s_e = model(e)
        s_i = model(i_)
        s_a = model(a)

        target = torch.ones(len(indices), 1)

        loss = 0
        loss += criterion(s_i, s_e, target)
        loss += criterion(s_a, s_i, target)
        loss += criterion(s_a, s_e, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# -----------------------------
# Ranking Accuracy
# -----------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for i in range(n):
        s_e = model(X_E[i])
        s_i = model(X_I[i])
        s_a = model(X_A[i])

        if s_i > s_e:
            correct += 1
        if s_a > s_i:
            correct += 1
        if s_a > s_e:
            correct += 1

        total += 3

ranking_accuracy = correct / total
print(f"Ranking accuracy: {ranking_accuracy:.4f}")

# -----------------------------
# Compute Score Range
# -----------------------------
all_scores = []

with torch.no_grad():
    for i in range(n):
        all_scores.append(model(X_E[i]).item())
        all_scores.append(model(X_I[i]).item())
        all_scores.append(model(X_A[i]).item())

min_score = min(all_scores)
max_score = max(all_scores)

print(f"Score range: {min_score:.4f} to {max_score:.4f}")

# -----------------------------
# Save Everything
# -----------------------------
torch.save({
    "model_state": model.state_dict(),
    "min_score": min_score,
    "max_score": max_score
}, "models/ranking_model.pt")

joblib.dump(scaler, "models/ranking_scaler.pkl")

print("Ranking model + scaler saved successfully.")
