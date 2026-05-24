from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader, Subset

from src.data import PairedTransform, build_dataset
from src.engine import evaluate, evaluate_by_generator, load_model_weights
from src.model import PatchForensicBranch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Branch C checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--dataset", choices=["cifake", "tiny-genimage"], default="cifake")
    parser.add_argument("--split", choices=["train", "test", "val"], default="test")
    parser.add_argument("--generators", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--semantic-size", type=int, default=224)
    parser.add_argument("--forensic-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    transform = PairedTransform(
        semantic_size=args.semantic_size,
        forensic_size=args.forensic_size,
        train=False,
        augment=False,
    )
    dataset = build_dataset(args.dataset_root, args.dataset, args.split, transform, generators=args.generators)
    if args.max_samples:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))

    loader = DataLoader(
        dataset,
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
    checkpoint = load_model_weights(args.checkpoint, model, device)
    metrics = evaluate(model, loader, device)
    result = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "overall": metrics,
        "by_generator": evaluate_by_generator(model, loader, device),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
