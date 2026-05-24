from __future__ import annotations

import torch


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu()
    labels = labels.detach().cpu().long()
    preds = (probs >= 0.5).long()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    total = max(len(labels), 1)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    balanced_acc = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
    return {
        "accuracy": (tp + tn) / total,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc(probs, labels),
    }


def auroc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    positives = probs[labels == 1]
    negatives = probs[labels == 0]
    if positives.numel() == 0 or negatives.numel() == 0:
        return float("nan")
    scores = torch.cat([positives, negatives])
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float32)
    pos_ranks = ranks[: positives.numel()].sum()
    auc = (pos_ranks - positives.numel() * (positives.numel() + 1) / 2) / (positives.numel() * negatives.numel())
    return float(auc)
