import torch
import torch.nn as nn


class SimpleFeatureProjector(nn.Module):
    def __init__(self, n_input_features: int, n_pooling_features: int, n_classes: int):
        super().__init__()

        self.projection1 = nn.Linear(n_input_features, n_pooling_features)
        self.projection2 = nn.Linear(n_pooling_features, n_classes)

    def forward(self, X: torch.Tensor):
        X = self.projection1(X)
        X_pooled, _indices = X.max(dim=-2)
        y = self.projection2(X_pooled)

        return y


class AttentionPooler(nn.Module):
    def __init__(self, n_input_features: int, n_pooling_features: int, n_classes: int):
        super().__init__()

        self.qk_dim = 256
        self.v_dim = 256

        self.projection1 = nn.Linear(n_input_features, n_pooling_features)
        self.kv = nn.Linear(n_pooling_features, self.qk_dim + self.v_dim)
        self.query = nn.Embedding(1, self.qk_dim)
        self.projection2 = nn.Linear(n_pooling_features, n_classes)

    def forward(self, X: torch.Tensor):
        X = self.projection1(X)
        k, v = torch.split(self.kv(X), [self.qk_dim, self.v_dim], dim=-1)
        scores = torch.softmax((k @ self.query(torch.tensor(0))), dim=0)
        X_pooled = v.T @ scores
        y = self.projection2(X_pooled)

        return y
