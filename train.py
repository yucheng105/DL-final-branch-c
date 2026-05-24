from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.data import PairedTransform, build_dataset, split_train_val
from src.engine import evaluate, save_checkpoint, train_one_epoch
from src.model import PatchForensicBranch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Branch C: Patch-Level Forensic Branch")
    parser.add_argument("--dataset-root", default="dataset", help="Root directory containing cifake/ and tiny-genimage/.")
    parser.add_argument("--dataset", choices=["cifake", "tiny-genimage"], default="cifake")
    parser.add_argument("--generators", nargs="*", default=None, help="Tiny-GenImage generator names to include.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--semantic-size", type=int, default=224)
    parser.add_argument("--forensic-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Useful for quick smoke tests.")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="runs/branch_c")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_transform = PairedTransform(
        semantic_size=args.semantic_size,
        forensic_size=args.forensic_size,
        train=True,
        augment=True,
    )
    val_transform = PairedTransform(
        semantic_size=args.semantic_size,
        forensic_size=args.forensic_size,
        train=False,
        augment=False,
    )

    train_full = build_dataset(args.dataset_root, args.dataset, "train", train_transform, generators=args.generators)
    val_full_for_split = build_dataset(args.dataset_root, args.dataset, "train", val_transform, generators=args.generators)
    train_subset, val_subset = split_train_val(train_full, args.val_fraction, args.seed)
    _, val_indices = train_subset.indices, val_subset.indices
    val_dataset = Subset(val_full_for_split, val_indices)

    if args.max_train_samples:
        train_subset = Subset(train_subset, range(min(args.max_train_samples, len(train_subset))))
    if args.max_val_samples:
        val_dataset = Subset(val_dataset, range(min(args.max_val_samples, len(val_dataset))))

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = PatchForensicBranch(
        patch_size=args.patch_size,
        stride=args.stride,
        top_k=args.top_k,
        feature_dim=args.feature_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_auroc = -1.0
    history: list[dict[str, object]] = []

    print(f"Training on {len(train_subset)} images, validating on {len(val_dataset)} images.")
    print(f"Device: {device}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, indent=2))

        val_score = val_metrics["auroc"]
        if val_score != val_score:
            val_score = val_metrics["balanced_accuracy"]
        if val_score > best_auroc:
            best_auroc = val_score
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, val_metrics)

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_checkpoint(output_dir / "last.pt", model, optimizer, args.epochs, history[-1]["val"])


if __name__ == "__main__":
    main()
