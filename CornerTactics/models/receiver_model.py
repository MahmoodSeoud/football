#!/usr/bin/env python3
"""
Receiver prediction model (Stage 1 of two-stage pipeline).

Predicts which of the 22 players (or no-receiver, class 22) receives
the ball after a corner kick delivery.

Models:
- ReceiverGAT: 4 GATv2 layers with edge features -> per-node MLP -> 23-class softmax
- ReceiverDeepSets: Phi network -> aggregation -> per-node scores
- Baselines: nearest-player-to-ball, majority-class

Node features (13-dim TacticAI-style):
  [x, y, vx, vy, tired_factor, ball_possession, role_GK, role_DEF, role_MID, role_FWD,
   is_attacker, is_defender, is_ball]

Edge features (2-dim):
  [is_same_team, is_opponent]

Usage:
    python models/receiver_model.py \
        --data_dir grf/data/gnn_receiver \
        --output_dir results_11v11/receiver
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("Warning: torch_geometric not installed. GAT model unavailable.")


# Default feature dimensions (can be overridden by metadata)
DEFAULT_NUM_NODE_FEATURES = 13
DEFAULT_NUM_EDGE_FEATURES = 2


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ReceiverDataset(Dataset):
    """PyTorch Dataset for receiver prediction with edge features."""

    def __init__(self, data_path):
        data = np.load(data_path)
        self.node_features = torch.tensor(data['node_features'], dtype=torch.float32)  # (N, 23, 13)
        self.edge_index = torch.tensor(data['edge_index'], dtype=torch.long)            # (2, 506)
        self.receiver_labels = torch.tensor(data['receiver_labels'], dtype=torch.long)  # (N,)
        self.shot_labels = torch.tensor(data['shot_labels'], dtype=torch.float32)       # (N,)

        # Load edge attributes if available (for TacticAI-style edge features)
        if 'edge_attr' in data:
            self.edge_attr = torch.tensor(data['edge_attr'], dtype=torch.float32)       # (506, 2)
        else:
            # Backward compatibility: no edge features
            self.edge_attr = None

    def __len__(self):
        return len(self.receiver_labels)

    def __getitem__(self, idx):
        item = {
            'x': self.node_features[idx],          # (23, 13)
            'edge_index': self.edge_index,          # (2, 506)
            'receiver': self.receiver_labels[idx],  # scalar (0-22)
            'shot': self.shot_labels[idx],          # scalar (0 or 1)
        }
        if self.edge_attr is not None:
            item['edge_attr'] = self.edge_attr      # (506, 2)
        return item


def collate_receiver_graphs(batch):
    """Custom collate for PyG batching with receiver labels and edge features."""
    if HAS_TORCH_GEOMETRIC:
        data_list = []
        has_edge_attr = 'edge_attr' in batch[0]
        for item in batch:
            data = Data(
                x=item['x'],
                edge_index=item['edge_index'],
                y=item['receiver'].unsqueeze(0),
            )
            if has_edge_attr:
                data.edge_attr = item['edge_attr']
            data_list.append(data)
        pyg_batch = Batch.from_data_list(data_list)
        # Attach shot labels for potential multi-task use
        pyg_batch.shot = torch.stack([item['shot'] for item in batch])
        return pyg_batch
    else:
        result = {
            'x': torch.stack([item['x'] for item in batch]),
            'edge_index': batch[0]['edge_index'],
            'receiver': torch.stack([item['receiver'] for item in batch]),
            'shot': torch.stack([item['shot'] for item in batch]),
        }
        if 'edge_attr' in batch[0]:
            result['edge_attr'] = batch[0]['edge_attr']
        return result


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ReceiverGAT(nn.Module):
    """
    GATv2-based receiver prediction model (TacticAI-style architecture).

    Architecture:
    - 4 GATv2 layers with LayerNorm and edge features (TacticAI uses 4 layers, 8 heads)
    - Per-node MLP scores each player node (0-21) as potential receiver
    - Global pooling + MLP scores no-receiver class (22)
    - Concatenate to get 23-class logits

    Edge features encode teammate/opponent relationships for better graph reasoning.
    """

    def __init__(self, in_channels=13, hidden_channels=64, embed_dim=32,
                 heads=8, dropout=0.3, edge_dim=2):
        super().__init__()

        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric required for ReceiverGAT")

        self.edge_dim = edge_dim

        # GATv2 backbone (4 layers as per TacticAI) with edge features
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout,
                              edge_dim=edge_dim if edge_dim > 0 else None)
        self.norm1 = nn.LayerNorm(hidden_channels * heads)

        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout,
                              edge_dim=edge_dim if edge_dim > 0 else None)
        self.norm2 = nn.LayerNorm(hidden_channels * heads)

        self.gat3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout,
                              edge_dim=edge_dim if edge_dim > 0 else None)
        self.norm3 = nn.LayerNorm(hidden_channels * heads)

        self.gat4 = GATv2Conv(hidden_channels * heads, embed_dim, heads=1, concat=False, dropout=dropout,
                              edge_dim=edge_dim if edge_dim > 0 else None)
        self.norm4 = nn.LayerNorm(embed_dim)

        # Per-node scoring: each player node -> scalar score
        self.node_scorer = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # No-receiver scoring from global graph representation
        self.no_receiver_scorer = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self.dropout = dropout
        self.embed_dim = embed_dim
        self.heads = heads

    def get_node_embeddings(self, data):
        """Extract per-node embeddings from the GATv2 backbone (for Stage 2)."""
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = self.norm1(x)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = self.norm2(x)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat3(x, edge_index, edge_attr=edge_attr)
        x = self.norm3(x)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat4(x, edge_index, edge_attr=edge_attr)
        x = self.norm4(x)
        x = F.relu(x)

        return x  # (total_nodes_in_batch, embed_dim)

    def forward(self, data):
        batch = data.batch
        node_embeds = self.get_node_embeddings(data)  # (total_nodes, embed_dim)

        # Score each of the 22 player nodes (exclude ball node 22)
        node_scores = self.node_scorer(node_embeds).squeeze(-1)  # (total_nodes,)

        # Reshape to (batch_size, 23_nodes) then take first 22 player scores
        batch_size = batch.max().item() + 1
        node_scores_2d = node_scores.view(batch_size, 23)
        player_scores = node_scores_2d[:, :22]  # (batch_size, 22)

        # No-receiver score from global pooling
        graph_embed = global_mean_pool(node_embeds, batch)  # (batch_size, embed_dim)
        no_recv_score = self.no_receiver_scorer(graph_embed)  # (batch_size, 1)

        # Concatenate: 22 player scores + 1 no-receiver score = 23 classes
        logits = torch.cat([player_scores, no_recv_score], dim=1)  # (batch_size, 23)
        return logits


class ReceiverDeepSets(nn.Module):
    """
    DeepSets-inspired receiver prediction.

    Phi network processes each node independently,
    then per-node scores + global aggregation for no-receiver.
    """

    def __init__(self, in_channels=13, hidden_channels=64, embed_dim=32, dropout=0.2):
        super().__init__()

        # Phi: per-node feature extraction
        self.phi = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, embed_dim),
        )

        # Per-node scoring
        self.node_scorer = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # No-receiver scoring from aggregated representation
        self.no_receiver_scorer = nn.Sequential(
            nn.Linear(embed_dim * 3, 32),  # mean + max + sum
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, data):
        if HAS_TORCH_GEOMETRIC and hasattr(data, 'x'):
            x, batch = data.x, data.batch
            batch_size = batch.max().item() + 1
        else:
            x = data['x']  # (batch_size, 23, 8)
            batch_size = x.shape[0]
            batch = None

        # Per-node embeddings
        node_embeds = self.phi(x)  # (total_nodes, embed_dim) or (B, 23, embed_dim)

        if batch is not None:
            # PyG format
            node_scores = self.node_scorer(node_embeds).squeeze(-1)
            node_scores_2d = node_scores.view(batch_size, 23)
            player_scores = node_scores_2d[:, :22]

            x_mean = global_mean_pool(node_embeds, batch)
            from torch_geometric.nn import global_max_pool, global_add_pool
            x_max = global_max_pool(node_embeds, batch)
            x_sum = global_add_pool(node_embeds, batch)
            agg = torch.cat([x_mean, x_max, x_sum], dim=1)
        else:
            # Batched tensor format
            node_scores = self.node_scorer(node_embeds).squeeze(-1)  # (B, 23)
            player_scores = node_scores[:, :22]

            x_mean = node_embeds.mean(dim=1)
            x_max = node_embeds.max(dim=1).values
            x_sum = node_embeds.sum(dim=1)
            agg = torch.cat([x_mean, x_max, x_sum], dim=1)

        no_recv_score = self.no_receiver_scorer(agg)  # (B, 1)
        logits = torch.cat([player_scores, no_recv_score], dim=1)  # (B, 23)
        return logits


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def nearest_player_baseline(node_features_batch):
    """
    Baseline: predict receiver as the player nearest to the ball.

    Args:
        node_features_batch: (N, 23, 8) array. Nodes 0-21 are players, 22 is ball.
                             Features: [x, y, vx, vy, ...].

    Returns:
        predictions: (N,) array of predicted receiver node indices (0-21).
    """
    ball_pos = node_features_batch[:, 22, :2]  # (N, 2)
    player_pos = node_features_batch[:, :22, :2]  # (N, 22, 2)
    dists = np.linalg.norm(player_pos - ball_pos[:, np.newaxis, :], axis=2)  # (N, 22)
    return np.argmin(dists, axis=1)


def majority_class_baseline(train_labels, n_samples):
    """
    Baseline: always predict the most common receiver class.

    Args:
        train_labels: Training set receiver labels.
        n_samples: Number of predictions to make.

    Returns:
        predictions: (n_samples,) array of majority class.
    """
    majority = np.bincount(train_labels).argmax()
    return np.full(n_samples, majority)


def frequency_weighted_baseline(train_labels, n_samples, seed=42):
    """
    Baseline: random predictions weighted by training class frequency.

    Args:
        train_labels: Training set receiver labels (0-22).
        n_samples: Number of predictions to generate.
        seed: Random seed for reproducibility.

    Returns:
        predictions: (n_samples,) array of predicted class indices.
    """
    np.random.seed(seed)
    counts = np.bincount(train_labels, minlength=23).astype(float)
    probs = counts / counts.sum()
    return np.random.choice(23, size=n_samples, p=probs)


def baseline_topk_accuracy(train_labels, test_labels, k=3):
    """
    Top-k accuracy for a baseline that predicts the top-k most frequent classes.

    Args:
        train_labels: Training set receiver labels.
        test_labels: Test set receiver labels.
        k: Number of top classes to consider as correct.

    Returns:
        float: Fraction of test samples whose label is in the top-k classes.
    """
    counts = np.bincount(train_labels, minlength=23)
    top_k_classes = set(np.argsort(counts)[-k:])
    return float(np.mean([label in top_k_classes for label in test_labels]))


def evaluate_all_baselines(train_labels, test_labels, test_node_features, seed=42):
    """
    Evaluate all baseline methods and return comprehensive metrics.

    Args:
        train_labels: Training set receiver labels (0-22).
        test_labels: Test set receiver labels (0-22).
        test_node_features: (N, 23, features) array of test set node features.
        seed: Random seed for frequency-weighted baseline.

    Returns:
        Dict with metrics for each baseline:
        - majority_class: top1, top3, top5
        - freq_weighted: top1, top3, top5
        - nearest_player: top1
        - uniform_random: top1, top3, top5 (theoretical)
    """
    n_test = len(test_labels)

    # Majority class baseline
    majority_preds = majority_class_baseline(train_labels, n_test)
    majority_top1 = float((majority_preds == test_labels).mean())
    majority_top3 = baseline_topk_accuracy(train_labels, test_labels, k=3)
    majority_top5 = baseline_topk_accuracy(train_labels, test_labels, k=5)

    # Frequency-weighted baseline
    freq_preds = frequency_weighted_baseline(train_labels, n_test, seed=seed)
    freq_top1 = float((freq_preds == test_labels).mean())
    freq_top3 = baseline_topk_accuracy(train_labels, test_labels, k=3)
    freq_top5 = baseline_topk_accuracy(train_labels, test_labels, k=5)

    # Nearest player baseline
    nearest_preds = nearest_player_baseline(test_node_features)
    nearest_top1 = float((nearest_preds == test_labels).mean())

    # Uniform random baseline (theoretical values)
    n_classes = 23
    uniform_top1 = 1.0 / n_classes
    uniform_top3 = 3.0 / n_classes
    uniform_top5 = 5.0 / n_classes

    return {
        'majority_class': {
            'top1': majority_top1,
            'top3': majority_top3,
            'top5': majority_top5,
        },
        'freq_weighted': {
            'top1': freq_top1,
            'top3': freq_top3,
            'top5': freq_top5,
        },
        'nearest_player': {
            'top1': nearest_top1,
        },
        'uniform_random': {
            'top1': uniform_top1,
            'top3': uniform_top3,
            'top5': uniform_top5,
        },
    }


def print_baseline_table(baselines):
    """
    Print a formatted table of baseline results.

    Args:
        baselines: Dict from evaluate_all_baselines().
    """
    print("\n--- Baselines ---")
    print(f"{'':18s} {'Top-1':>7s} {'Top-3':>7s} {'Top-5':>7s}")
    print("-" * 42)

    # Majority class
    mc = baselines['majority_class']
    print(f"{'Majority class':18s} {mc['top1']:7.3f} {mc['top3']:7.3f} {mc['top5']:7.3f}")

    # Frequency-weighted
    fw = baselines['freq_weighted']
    print(f"{'Freq-weighted':18s} {fw['top1']:7.3f} {fw['top3']:7.3f} {fw['top5']:7.3f}")

    # Nearest player (only top-1)
    np_base = baselines['nearest_player']
    print(f"{'Nearest player':18s} {np_base['top1']:7.3f} {'N/A':>7s} {'N/A':>7s}")

    # Uniform random
    ur = baselines['uniform_random']
    print(f"{'Uniform random':18s} {ur['top1']:7.3f} {ur['top3']:7.3f} {ur['top5']:7.3f}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        labels = batch.y.squeeze()

        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def evaluate_receiver(model, loader, device):
    """Evaluate receiver prediction model."""
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            labels = batch.y.squeeze()
            logits = model(batch)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)  # (N, 23)
    all_labels = torch.cat(all_labels, dim=0)  # (N,)

    proba = torch.softmax(all_logits, dim=1).numpy()  # (N, 23)
    labels_np = all_labels.numpy()
    preds = all_logits.argmax(dim=1).numpy()

    return compute_receiver_metrics(labels_np, preds, proba)


def compute_receiver_metrics(y_true, y_pred, y_pred_proba, k_values=(1, 3, 5)):
    """
    Compute comprehensive receiver prediction metrics.

    Args:
        y_true: (N,) ground truth receiver labels (0-22).
        y_pred: (N,) predicted receiver labels.
        y_pred_proba: (N, 23) predicted probabilities.
        k_values: Tuple of k values for top-k accuracy.

    Returns:
        Dict of metrics.
    """
    n = len(y_true)
    top1_acc = float((y_pred == y_true).mean())

    # Top-k accuracy
    topk_accs = {}
    for k in k_values:
        topk_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]  # (N, k)
        topk_correct = np.array([y_true[i] in topk_preds[i] for i in range(n)])
        topk_accs[f'top{k}_acc'] = float(topk_correct.mean())

    # Per-group breakdown
    attacker_mask = y_true < 11
    defender_mask = (y_true >= 11) & (y_true < 22)
    no_recv_mask = y_true == 22

    group_accs = {}
    for name, mask in [('attacker', attacker_mask), ('defender', defender_mask), ('no_receiver', no_recv_mask)]:
        if mask.sum() > 0:
            group_accs[f'{name}_acc'] = float((y_pred[mask] == y_true[mask]).mean())
            group_accs[f'{name}_count'] = int(mask.sum())
        else:
            group_accs[f'{name}_acc'] = 0.0
            group_accs[f'{name}_count'] = 0

    # Confusion between attacker/defender/no-receiver (3-class)
    def to_group(labels):
        groups = np.zeros_like(labels)
        groups[labels < 11] = 0       # attacker
        groups[(labels >= 11) & (labels < 22)] = 1  # defender
        groups[labels == 22] = 2      # no-receiver
        return groups

    group_true = to_group(y_true)
    group_pred = to_group(y_pred)
    group_cm = confusion_matrix(group_true, group_pred, labels=[0, 1, 2])

    metrics = {
        'top1_acc': top1_acc,
        **topk_accs,
        **group_accs,
        'group_confusion_matrix': group_cm.tolist(),
        'n_samples': n,
    }

    return metrics


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train_receiver(data_dir, output_dir, model_type='gat', epochs=150,
                   batch_size=64, lr=1e-3, patience=20, device=None):
    """
    Train receiver prediction model with early stopping.

    Args:
        data_dir: Directory with train.npz, val.npz, test.npz (with receiver_labels).
        output_dir: Directory to save model and results.
        model_type: 'gat' or 'deepsets'.
        epochs: Maximum training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        patience: Early stopping patience.
        device: Torch device.

    Returns:
        Dict with training results.
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    train_dataset = ReceiverDataset(data_path / 'train.npz')
    val_dataset = ReceiverDataset(data_path / 'val.npz')
    test_dataset = ReceiverDataset(data_path / 'test.npz')

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Load metadata for class weights and feature dimensions
    meta_path = data_path / 'metadata.json'
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        class_weights = torch.tensor(metadata['receiver_class_weights'], dtype=torch.float32)
        in_channels = metadata.get('num_node_features', DEFAULT_NUM_NODE_FEATURES)
        edge_dim = metadata.get('num_edge_features', DEFAULT_NUM_EDGE_FEATURES)
    else:
        # Compute from training labels
        train_labels = train_dataset.receiver_labels.numpy()
        counts = np.bincount(train_labels, minlength=23).astype(np.float32)
        counts[counts == 0] = 1.0
        class_weights = torch.tensor(len(train_labels) / (23.0 * counts))
        in_channels = DEFAULT_NUM_NODE_FEATURES
        edge_dim = DEFAULT_NUM_EDGE_FEATURES

    # Check if dataset has edge attributes
    has_edge_attr = train_dataset.edge_attr is not None
    if not has_edge_attr:
        edge_dim = 0  # Disable edge features if not available

    class_weights = class_weights.to(device)
    print(f"Class weights range: [{class_weights.min():.2f}, {class_weights.max():.2f}]")
    print(f"Feature dimensions: in_channels={in_channels}, edge_dim={edge_dim}")

    # Data loaders
    use_pyg = HAS_TORCH_GEOMETRIC and model_type == 'gat'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_receiver_graphs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_receiver_graphs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_receiver_graphs)

    # Create model with correct feature dimensions
    if model_type == 'gat':
        model = ReceiverGAT(in_channels=in_channels, hidden_channels=64, embed_dim=32,
                            heads=8, dropout=0.3, edge_dim=edge_dim)
    elif model_type == 'deepsets':
        model = ReceiverDeepSets(in_channels=in_channels, hidden_channels=64, embed_dim=32,
                                 dropout=0.2)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__} ({n_params:,} parameters)")

    # Loss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7
    )

    # Training loop
    best_val_acc = 0
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_top1': [], 'val_top3': []}

    print("\nTraining...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                            criterion, device)
        val_metrics = evaluate_receiver(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_top1'].append(val_metrics['top1_acc'])
        history['val_top3'].append(val_metrics['top3_acc'])

        scheduler.step(val_metrics['top1_acc'])

        if val_metrics['top1_acc'] > best_val_acc:
            best_val_acc = val_metrics['top1_acc']
            best_epoch = epoch
            torch.save(model.state_dict(), output_path / 'best_model.pt')

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                  f"val_top1={val_metrics['top1_acc']:.3f}, val_top3={val_metrics['top3_acc']:.3f}")

        if epoch - best_epoch >= patience:
            print(f"\nEarly stopping at epoch {epoch} (best: {best_epoch})")
            break

    # Load best model and evaluate on test set
    print("\nLoading best model...")
    model.load_state_dict(torch.load(output_path / 'best_model.pt', weights_only=True))

    test_metrics = evaluate_receiver(model, test_loader, device)

    print(f"\nTest Results:")
    print(f"  Top-1 accuracy: {test_metrics['top1_acc']:.3f}")
    print(f"  Top-3 accuracy: {test_metrics['top3_acc']:.3f}")
    print(f"  Top-5 accuracy: {test_metrics['top5_acc']:.3f}")
    print(f"  Attacker accuracy: {test_metrics['attacker_acc']:.3f} (n={test_metrics['attacker_count']})")
    print(f"  Defender accuracy: {test_metrics['defender_acc']:.3f} (n={test_metrics['defender_count']})")
    print(f"  No-receiver accuracy: {test_metrics['no_receiver_acc']:.3f} (n={test_metrics['no_receiver_count']})")
    print(f"\n  Group confusion matrix (attacker/defender/no-receiver):")
    cm = test_metrics['group_confusion_matrix']
    print(f"    Pred:   Att   Def   NoR")
    for i, label in enumerate(['Att', 'Def', 'NoR']):
        print(f"    {label}: {cm[i][0]:5d} {cm[i][1]:5d} {cm[i][2]:5d}")

    # Run baselines on test set
    test_data = np.load(data_path / 'test.npz')
    test_node_features = test_data['node_features']
    test_receiver_labels = test_data['receiver_labels']
    train_data = np.load(data_path / 'train.npz')
    train_receiver_labels = train_data['receiver_labels']

    baselines = evaluate_all_baselines(
        train_receiver_labels, test_receiver_labels, test_node_features
    )
    print_baseline_table(baselines)

    # Save results
    results = {
        'model': model.__class__.__name__,
        'model_type': model_type,
        'best_epoch': best_epoch,
        'best_val_top1': best_val_acc,
        'test_metrics': test_metrics,
        'baselines': baselines,
        'history': history,
        'hyperparameters': {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'patience': patience,
            'weight_decay': 1e-4,
        },
    }

    with open(output_path / 'receiver_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path / 'receiver_results.json'}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train receiver prediction model (Stage 1)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory with receiver-labeled GNN data')
    parser.add_argument('--output_dir', type=str, default='./results_11v11/receiver',
                        help='Output directory for model and results')
    parser.add_argument('--model', type=str, default='gat',
                        choices=['gat', 'deepsets'],
                        help='Model type')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None

    train_receiver(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=device,
    )


if __name__ == '__main__':
    main()
