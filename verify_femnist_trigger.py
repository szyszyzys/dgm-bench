"""
verify_femnist_trigger.py — Visually confirm the FEMNIST backdoor trigger.

Loads a few FEMNIST samples and applies the same trigger that
BackdoorImageGenerator applies at runtime for 28x28 inputs:
    CHECKERBOARD pattern, 4x4, alpha=0.8, BOTTOM_RIGHT.

Saves a PNG grid so you can eyeball whether the trigger is clearly visible.

This script deliberately re-implements the trigger math instead of importing
BackdoorImageGenerator, to avoid a circular-import chain in src.marketplace.*
that blocks loading the generator as a standalone tool.

Usage:
    python verify_femnist_trigger.py
    python verify_femnist_trigger.py --n 8 --out trigger_check.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset as hf_load_dataset


# -----------------------------------------------------------------------------
# Trigger logic — must match src/attacks/gradient_market/poison_attack/attack_utils.py
# -----------------------------------------------------------------------------
FEMNIST_SPATIAL = 28
TRIGGER_SIZE = (4, 4)
BLEND_ALPHA = 0.8


def make_checkerboard_trigger(h: int, w: int, channels: int) -> torch.Tensor:
    """Same formula as BackdoorImageGenerator._generate_trigger_pattern."""
    coords = torch.arange(h).unsqueeze(1) + torch.arange(w).unsqueeze(0)
    cb = (coords % 2 == 0).float()
    return cb.unsqueeze(0).repeat(channels, 1, 1)


def apply_trigger_bottom_right(image: torch.Tensor, trigger: torch.Tensor,
                               alpha: float) -> torch.Tensor:
    """Blend `trigger` into the bottom-right corner of `image`. Pure function."""
    image = image.clone().float()
    _, H, W = image.shape
    _, h, w = trigger.shape
    y, x = H - h, W - w
    region = image[:, y:y + h, x:x + w]
    blended = (1.0 - alpha) * region + alpha * trigger
    image[:, y:y + h, x:x + w] = torch.clamp(blended, 0.0, 1.0)
    return image


# -----------------------------------------------------------------------------
# Dataset loading — direct HF, no project imports
# -----------------------------------------------------------------------------
def load_femnist_samples(n: int) -> list[torch.Tensor]:
    """Return n clean FEMNIST samples as [3, 28, 28] float tensors in [0, 1]."""
    print("  Loading flwrlabs/femnist from HuggingFace...")
    hf_ds = hf_load_dataset("flwrlabs/femnist")
    split = hf_ds["train"] if "train" in hf_ds else list(hf_ds.values())[0]
    # Pull first n images via column access (no row iteration)
    imgs_pil = split["image"][:n]
    samples = []
    for img_pil in imgs_pil:
        arr = np.asarray(img_pil.convert("L"), dtype=np.uint8)  # [28, 28]
        t = torch.from_numpy(arr).unsqueeze(0).float() / 255.0  # [1, 28, 28]
        t = t.expand(3, -1, -1).contiguous()                    # [3, 28, 28]
        samples.append(t)
    return samples


# -----------------------------------------------------------------------------
# PNG grid writer — only needs Pillow
# -----------------------------------------------------------------------------
def save_grid(clean: list[torch.Tensor], triggered: list[torch.Tensor],
              out_path: Path):
    try:
        from PIL import Image
    except ImportError:
        print("  Pillow (PIL) not available; skipping PNG grid.")
        return

    n = len(clean)
    H, W = 28, 28
    scale = 6              # upscale so the pattern is clearly visible
    pad = 4
    cell_w = W * scale + pad
    cell_h = H * scale + pad
    grid_w = 2 * cell_w + pad
    grid_h = n * cell_h + pad

    canvas = np.full((grid_h, grid_w), 255, dtype=np.uint8)

    def to_gray_u8_upscaled(t: torch.Tensor) -> np.ndarray:
        arr = t[0].detach().cpu().clamp(0, 1).numpy()
        arr_u8 = (arr * 255).astype(np.uint8)
        # nearest-neighbour upscale
        return np.kron(arr_u8, np.ones((scale, scale), dtype=np.uint8))

    for i, (c, t) in enumerate(zip(clean, triggered)):
        y0 = pad + i * cell_h
        canvas[y0:y0 + H * scale, pad:pad + W * scale] = to_gray_u8_upscaled(c)
        canvas[y0:y0 + H * scale,
               pad + cell_w:pad + cell_w + W * scale] = to_gray_u8_upscaled(t)

    Image.fromarray(canvas).save(out_path)
    print(f"  Saved: {out_path}  ({grid_w}x{grid_h} px)")
    print(f"  Left column = clean  |  Right column = triggered")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--n", type=int, default=6, help="Number of samples to visualize")
    ap.add_argument("--out", default="femnist_trigger_check.png", help="Output PNG path")
    args = ap.parse_args()

    print("[1/3] Loading FEMNIST samples...")
    clean = load_femnist_samples(args.n)
    for i, t in enumerate(clean):
        print(f"  sample {i}: shape={tuple(t.shape)} dtype={t.dtype} "
              f"min={t.min():.3f} max={t.max():.3f} mean={t.mean():.3f}")

    print("\n[2/3] Building checkerboard trigger and applying...")
    trigger = make_checkerboard_trigger(TRIGGER_SIZE[0], TRIGGER_SIZE[1], channels=3)
    print(f"  trigger pattern ({TRIGGER_SIZE[0]}x{TRIGGER_SIZE[1]}, channel 0):")
    for row in trigger[0].int().tolist():
        print("   ", row)

    triggered = []
    for i, img in enumerate(clean):
        poisoned = apply_trigger_bottom_right(img, trigger, BLEND_ALPHA)
        triggered.append(poisoned)
        diff = (poisoned - img).abs()
        n_changed = (diff > 0.01).sum().item()
        max_delta = diff.max().item()
        br = poisoned[0, -4:, -4:]
        print(f"  sample {i}: pixels_changed={n_changed:3d}  max_delta={max_delta:.3f}"
              f"  br_min={br.min():.2f}  br_max={br.max():.2f}")

    print("\n[3/3] Saving visualization grid...")
    save_grid(clean, triggered, Path(args.out))

    # Final verdict (count per sample across all 3 channels)
    changed_per_sample = [
        ((t - c).abs() > 0.01).sum().item()
        for c, t in zip(clean, triggered)
    ]
    avg_changed = sum(changed_per_sample) / max(len(clean), 1)
    # Expected: 8 dark dots × 3 channels = 24 (if bottom-right is all white)
    print()
    print(f"Avg pixels visibly changed per sample: {avg_changed:.1f}")
    print(f"Expected on pure white background:    24.0  (8 dots × 3 channels)")

    if avg_changed >= 18:
        print("✅ TRIGGER IS CLEARLY VISIBLE — expect ASR to rise well above the 5% baseline.")
    elif avg_changed >= 10:
        print("⚠  Trigger is weaker than expected — some FEMNIST samples may have "
              "non-white content in the bottom-right. Still probably OK.")
    else:
        print("❌ TRIGGER IS NOT VISIBLE — something is wrong. Do NOT launch step 10.")


if __name__ == "__main__":
    main()
