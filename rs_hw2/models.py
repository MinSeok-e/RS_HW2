"""Original implementations of the published EASE and NeuMF model definitions."""
import numpy as np
import torch
from torch import nn


def ease_scores(x, regularization):
    """Exact EASE in user space (Woodbury), without materializing item x item B.

    P = (X.T X + lambda I)^-1, B = I - P / diag(P).
    A = (X X.T + lambda I)^-1; X P = A X.
    Therefore X B = X - (A X) / diag(P).
    """
    if regularization <= 0:
        raise ValueError('EASE regularization must be positive')
    x = np.asarray(x, dtype=np.float64)
    gram = x @ x.T
    gram.flat[::len(gram) + 1] += regularization
    ax = np.linalg.solve(gram, x)
    diagonal = (1.0 - np.sum(x * ax, axis=0)) / regularization
    if not (diagonal > 0).all():
        raise FloatingPointError('Nonpositive inverse diagonal')
    scores = x - ax / diagonal[None, :]
    return scores, {'x': x, 'ax': ax, 'inverse_diagonal': diagonal,
                    'regularization': np.asarray(regularization)}


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, mf_dim=16, mlp_dim=32, hidden=(32, 16, 8)):
        super().__init__()
        self.gmf_user = nn.Embedding(n_users, mf_dim)
        self.gmf_item = nn.Embedding(n_items, mf_dim)
        self.mlp_user = nn.Embedding(n_users, mlp_dim)
        self.mlp_item = nn.Embedding(n_items, mlp_dim)
        layers = []
        previous = 2 * mlp_dim
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(mf_dim + previous, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Embedding):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, users, items):
        gmf = self.gmf_user(users) * self.gmf_item(items)
        mlp = self.mlp(torch.cat([self.mlp_user(users), self.mlp_item(items)], dim=-1))
        return self.output(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)


@torch.inference_mode()
def neumf_scores(model, n_users, n_items, device, chunk_size=65536):
    model.eval()
    output = np.empty(n_users * n_items, dtype=np.float32)
    for start in range(0, len(output), chunk_size):
        stop = min(start + chunk_size, len(output))
        index = torch.arange(start, stop, device=device)
        output[start:stop] = model(index // n_items, index % n_items).cpu().numpy()
    return output.reshape(n_users, n_items)


def sample_epoch(positive_users, positive_items, seen, negatives, rng):
    """Uniform negatives from fit-active items, excluding only fit positives.

    Held-out interactions are deliberately NOT inspected for negative sampling.
    Negative samples are independent draws with replacement.
    """
    if negatives < 1:
        raise ValueError('At least one negative per positive is required')
    active = np.flatnonzero(seen.any(axis=0))
    if (seen.sum(axis=1) >= len(active)).any():
        raise ValueError('Some user has no negative candidate')
    users_neg = np.repeat(positive_users, negatives)
    items_neg = active[rng.integers(len(active), size=len(users_neg))]
    invalid = seen[users_neg, items_neg]
    while invalid.any():
        items_neg[invalid] = active[rng.integers(len(active), size=int(invalid.sum()))]
        invalid = seen[users_neg, items_neg]
    users = np.concatenate([positive_users, users_neg])
    items = np.concatenate([positive_items, items_neg])
    labels = np.concatenate([np.ones(len(positive_users)), np.zeros(len(users_neg))]).astype(np.float32)
    order = rng.permutation(len(users))
    return users[order], items[order], labels[order]
