import torch
import torch.nn as nn


class SimpleFeatureProjector(nn.Module):
    def __init__(self, n_input_features: int, n_pooling_features: int, n_classes: int):
        super().__init__()

        self.projection1 = nn.Linear(n_input_features, n_pooling_features)
        self.projection2 = nn.Linear(n_pooling_features, n_classes)

    def forward(self, X: torch.Tensor):
        X = self.projection1(X)
        X_pooled = X.max(dim=-2)
        y = self.projection1(X_pooled)

        return y
