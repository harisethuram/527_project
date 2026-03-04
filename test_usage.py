# save as: debug_chexpert_throughput.py
#
# Purpose:
# 1) Measure dataloader fetch time (CPU-side) vs GPU compute time.
# 2) Run a synthetic (no dataloader) GPU benchmark to rule out "system capacity" issues.
#
# Usage:
#   python debug_chexpert_throughput.py --model cnn --batch_size 32 --steps 200
#
# Notes:
# - This script does NOT train for epochs; it just benchmarks throughput.
# - It prints p50/p90/p99 for:
#     fetch time (next(batch)), H2D transfer time, GPU compute time, total iter time.
# - It also prints a synthetic baseline (random tensors) to show expected GPU speed.

import argparse
import os
import time
from statistics import mean

import torch
from torch.utils.data import DataLoader

from src.data.chexpert import ChexpertDataset
from src.model.CNN import ChexpertCNN
from src.model.CLIP import ChexpertCLIP
from src.model.transformer import ChexpertTransformer


def pct(arr, p):
    if not arr:
        return float("nan")
    s = sorted(arr)
    idx = int(p * (len(s) - 1))
    return s[idx]


def summarize(name, arr, unit="s"):
    if not arr:
        print(f"{name}: no samples")
        return
    print(f"{name}: mean={mean(arr):.4f}{unit}  p50={pct(arr,0.50):.4f}{unit}  p90={pct(arr,0.90):.4f}{unit}  p99={pct(arr,0.99):.4f}{unit}")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def move_to_device(batch, device):
    # Handles (tensor), (tuple/list), (dict)
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(x, device) for x in batch)
    return batch


def masked_bce_loss(outputs, labels):
    valid_mask = (labels != -1.0).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs, labels.clamp(min=0.0), reduction="none"
    )
    return (loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)


def build_model(name):
    if name == "cnn":
        return ChexpertCNN()
    if name == "clip":
        return ChexpertCLIP()
    if name == "transformer":
        return ChexpertTransformer()
    raise ValueError(f"Unknown model: {name}")


def infer_input_shape(train_dataset, batch_size, device):
    # Take a single sample to infer H/W/C (no GPU needed)
    x0, y0 = train_dataset[0]
    # x0 could be [C,H,W] or [H,W] depending on dataset; assume it returns tensor
    if not torch.is_tensor(x0):
        raise ValueError("Dataset did not return a tensor for images; cannot infer shape.")
    if x0.dim() == 2:
        # grayscale HxW -> add channel
        x0 = x0.unsqueeze(0)
    if x0.dim() != 3:
        raise ValueError(f"Unexpected image tensor shape: {tuple(x0.shape)}")
    C, H, W = x0.shape
    # CheXpert is multi-label (often 14); infer label dim
    if torch.is_tensor(y0) and y0.dim() == 1:
        num_classes = y0.numel()
    else:
        num_classes = 14
    return C, H, W, num_classes


@torch.no_grad()
def benchmark_dataloader_only(loader, steps, warmup):
    it = iter(loader)
    fetch = []
    for i in range(steps):
        t0 = time.perf_counter()
        _ = next(it)
        t1 = time.perf_counter()
        if i >= warmup:
            fetch.append(t1 - t0)
    print("\n=== DATALOADER-ONLY (CPU fetch) ===")
    summarize("next(batch)", fetch, unit="s")
    return fetch


def benchmark_full_iter(model, loader, optimizer, device, steps, warmup, use_amp=False):
    model.train()

    it = iter(loader)
    fetch_s, h2d_s, total_s = [], [], []
    gpu_ms = []

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    for i in range(steps):
        t_iter0 = time.perf_counter()

        # (a) fetch
        t0 = time.perf_counter()
        batch = next(it)
        t1 = time.perf_counter()

        # (b) H2D
        t2 = time.perf_counter()
        batch = move_to_device(batch, device)
        _sync()
        t3 = time.perf_counter()

        # unpack
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images, labels = batch[0], batch[1]
        elif isinstance(batch, dict):
            images = batch.get("image", batch.get("images", batch.get("x")))
            labels = batch.get("label", batch.get("labels", batch.get("y")))
            if images is None or labels is None:
                raise ValueError("Couldn't infer images/labels from dict batch. Edit unpacking logic.")
        else:
            raise ValueError("Unknown batch structure. Edit unpacking logic.")

        optimizer.zero_grad(set_to_none=True)

        starter.record()
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = masked_bce_loss(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ender.record()
        _sync()
        t_iter1 = time.perf_counter()

        if i >= warmup:
            fetch_s.append(t1 - t0)
            h2d_s.append(t3 - t2)
            gpu_ms.append(starter.elapsed_time(ender))
            total_s.append(t_iter1 - t_iter0)

    print("\n=== FULL ITERATION SPLIT ===")
    summarize("fetch next(batch)", fetch_s, unit="s")
    summarize("H2D transfer", h2d_s, unit="s")
    summarize("GPU compute (fwd+bwd+step)", gpu_ms, unit="ms")
    summarize("total iter wall", total_s, unit="s")

    return fetch_s, h2d_s, gpu_ms, total_s


def benchmark_synthetic(model, optimizer, device, batch_size, C, H, W, num_classes, steps, warmup, use_amp=False):
    model.train()

    # Synthetic inputs: eliminates dataloader, disk, decode, transforms entirely.
    x = torch.randn(batch_size, C, H, W, device=device)
    y = torch.randint(0, 2, (batch_size, num_classes), device=device).float()

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    gpu_ms, total_s = [], []

    for i in range(steps):
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        starter.record()
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(x)
            loss = masked_bce_loss(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ender.record()
        _sync()
        t1 = time.perf_counter()

        if i >= warmup:
            gpu_ms.append(starter.elapsed_time(ender))
            total_s.append(t1 - t0)

    print("\n=== SYNTHETIC (NO DATALOADER) ===")
    summarize("GPU compute (fwd+bwd+step)", gpu_ms, unit="ms")
    summarize("total iter wall", total_s, unit="s")

    return gpu_ms, total_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "clip", "transformer"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)

    # Dataloader knobs to test
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)

    # AMP toggle (helps compute, not I/O)
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    train_dataset = ChexpertDataset(split="train")

    # Infer shapes for synthetic benchmark
    C, H, W, num_classes = infer_input_shape(train_dataset, args.batch_size, device)
    print(f"Inferred input shape: C={C}, H={H}, W={W}, num_classes={num_classes}")

    # DataLoader
    # Note: persistent_workers requires num_workers>0
    persistent = args.persistent_workers and args.num_workers > 0
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=persistent,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # Model
    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Benchmarks
    # 1) Dataloader-only
    _ = benchmark_dataloader_only(loader, steps=args.steps, warmup=args.warmup)

    # 2) Full iteration split (real data)
    _ = benchmark_full_iter(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=device,
        steps=args.steps,
        warmup=args.warmup,
        use_amp=args.amp,
    )

    # 3) Synthetic compute baseline
    # Re-init optimizer to avoid weirdness from prior steps (not strictly necessary)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    _ = benchmark_synthetic(
        model=model,
        optimizer=optimizer,
        device=device,
        batch_size=args.batch_size,
        C=C, H=H, W=W,
        num_classes=num_classes,
        steps=args.steps,
        warmup=args.warmup,
        use_amp=args.amp,
    )

    print("\nInterpretation:")
    print("- If dataloader-only next(batch) is large (e.g., >0.2s, especially seconds), you're input-bound.")
    print("- If synthetic is fast but real is slow, it's not GPU capacity; it's I/O/decode/transforms/loader.")
    print("- If synthetic is also slow, then it's compute/model/config (AMP/channels-last/compile/etc.).")


if __name__ == "__main__":
    main()