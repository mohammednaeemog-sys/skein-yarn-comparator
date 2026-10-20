"""Train a yarn-core segmentation model on expert-reviewed masks.

Dataset layout (same stem in the images/ and masks/ folders):
  data/yarn_masks/images/train/*.jpg|png
  data/yarn_masks/masks/train/*.png
  data/yarn_masks/images/val/*.jpg|png
  data/yarn_masks/masks/val/*.png

Mask convention: grayscale mask where WHITE (>=128) means the yarn body/core
and BLACK means background. Annotate the body boundary consistently; do not
sometimes include long flyaway hairs and sometimes exclude them.

Split by independent physical sample/capture session before making these folders.
Do not put crops of the same physical strand in both train and val. Keep a separate
held-out test folder for model selection and final reporting.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def pair_files(root: Path, split: str):
    image_dir = root / "images" / split
    mask_dir = root / "masks" / split
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected both {image_dir} and {mask_dir}")
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in exts}
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in exts}
    if not images:
        raise ValueError(f"No images found in {image_dir}")
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(f"Image/mask filenames must match. Missing masks={missing_masks[:5]}, missing images={missing_images[:5]}")
    return [(images[k], masks[k]) for k in sorted(images)]


def read_pair(image_path: Path, mask_path: Path):
    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"))
    with Image.open(mask_path) as im:
        mask = np.asarray(im.convert("L"))
    if rgb.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Image/mask size mismatch: {image_path.name}: {rgb.shape[:2]} vs {mask.shape[:2]}")
    return rgb, (mask >= 128).astype(np.uint8)


def crop_padded(rgb, mask, y, x, size):
    h, w = mask.shape
    y0 = max(0, min(int(y), max(0, h - size)))
    x0 = max(0, min(int(x), max(0, w - size)))
    image_patch = rgb[y0:min(y0 + size, h), x0:min(x0 + size, w)]
    mask_patch = mask[y0:min(y0 + size, h), x0:min(x0 + size, w)]
    ph, pw = mask_patch.shape
    if ph < size or pw < size:
        pad_h, pad_w = size - ph, size - pw
        rgb_mode = "reflect" if ph > 1 and pw > 1 else "edge"
        image_patch = np.pad(image_patch, ((0, pad_h), (0, pad_w), (0, 0)), mode=rgb_mode)
        mask_patch = np.pad(mask_patch, ((0, pad_h), (0, pad_w)), mode="constant")
    return image_patch, mask_patch


class PatchDataset:
    """Dynamic native-resolution patch sampler, prioritizing patches with yarn."""
    def __init__(self, pairs, patch_size: int, patches_per_image: int, training: bool):
        self.pairs = pairs
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.training = training

    def __len__(self):
        return len(self.pairs) * self.patches_per_image

    def __getitem__(self, index):
        import torch
        from torch.utils.data import Dataset
        image_path, mask_path = self.pairs[index % len(self.pairs)]
        rgb, mask = read_pair(image_path, mask_path)
        h, w = mask.shape
        size = self.patch_size
        positive = np.argwhere(mask > 0)
        if self.training and len(positive) and random.random() < 0.85:
            cy, cx = positive[random.randrange(len(positive))]
            y = int(cy) - random.randint(size // 4, 3 * size // 4)
            x = int(cx) - random.randint(size // 4, 3 * size // 4)
        else:
            y = random.randint(0, max(0, h - size)) if self.training else max(0, (h - size) // 2)
            x = random.randint(0, max(0, w - size)) if self.training else max(0, (w - size) // 2)
        ip, mp = crop_padded(rgb, mask, y, x, size)
        if self.training:
            if random.random() < 0.5:
                ip, mp = np.fliplr(ip), np.fliplr(mp)
            if random.random() < 0.5:
                ip, mp = np.flipud(ip), np.flipud(mp)
            if random.random() < 0.5:
                k = random.randint(1, 3)
                ip, mp = np.rot90(ip, k), np.rot90(mp, k)
            if random.random() < 0.6:
                f = random.uniform(0.85, 1.15)
                ip = np.clip(ip.astype(np.float32) * f, 0, 255).astype(np.uint8)
            if random.random() < 0.3:
                hsv = cv2.cvtColor(ip, cv2.COLOR_RGB2HSV)
                hsv[..., 2] = np.clip(hsv[..., 2].astype(np.float32) * random.uniform(0.88, 1.12), 0, 255)
                ip = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        ip = np.ascontiguousarray(ip).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        ip = (ip - mean) / std
        x_tensor = torch.from_numpy(ip.transpose(2, 0, 1).copy()).float()
        y_tensor = torch.from_numpy(np.ascontiguousarray(mp)[None].astype(np.float32))
        return x_tensor, y_tensor


def dice_from_logits(logits, target):
    import torch
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).float()
    dims = (1, 2, 3)
    intersection = (pred * target).sum(dims)
    denom = pred.sum(dims) + target.sum(dims)
    return ((2 * intersection + 1e-6) / (denom + 1e-6)).mean()


def dice_loss(logits, target):
    import torch
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    return 1 - ((2 * intersection + 1e-6) / (denom + 1e-6)).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/yarn_masks"))
    parser.add_argument("--architecture", choices=["unetplusplus", "deeplabv3plus"], default="unetplusplus")
    parser.add_argument("--encoder-weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patches-per-image", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.patch_size < 128 or args.patch_size % 32:
        raise SystemExit("--patch-size must be >=128 and divisible by 32 (suggested: 512)")
    if args.epochs < 1 or args.batch_size < 1 or args.patches_per_image < 1:
        raise SystemExit("epochs, batch size, and patches per image must be positive")

    try:
        import torch
        import segmentation_models_pytorch as smp
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Install optional ML dependencies using the instructions in requirements-ml.txt") from exc

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_pairs = pair_files(args.data_dir, "train")
    val_pairs = pair_files(args.data_dir, "val")
    print(f"Device: {device}; train specimens: {len(train_pairs)}; validation specimens: {len(val_pairs)}")

    model_builder = smp.UnetPlusPlus if args.architecture == "unetplusplus" else smp.DeepLabV3Plus
    encoder_weights = None if args.encoder_weights == "none" else "imagenet"
    model = model_builder(encoder_name="resnet34", encoder_weights=encoder_weights, in_channels=3, classes=1, activation=None).to(device)
    train_ds = PatchDataset(train_pairs, args.patch_size, args.patches_per_image, training=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=4, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_dice = -1.0
    output_path = args.output or (Path("models") / f"yarn_{args.architecture}_resnet34_best.pth")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(x)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        # Validate on full original images tiled at native resolution. Validation masks
        # must belong to physical specimens/capture sessions not used in training.
        model.eval()
        val_dices = []
        with torch.inference_mode():
            for image_path, mask_path in val_pairs:
                rgb, true_mask = read_pair(image_path, mask_path)
                # Local import avoids making torch mandatory for classical app use.
                from yarn_ml import predict_mask_tiled
                predicted, _ = predict_mask_tiled(rgb, model, tile_size=args.patch_size, overlap=args.patch_size // 4)
                p = predicted > 0
                t = true_mask > 0
                d = (2.0 * np.logical_and(p, t).sum() + 1e-6) / (p.sum() + t.sum() + 1e-6)
                val_dices.append(float(d))
        val_dice = float(np.mean(val_dices)) if val_dices else 0.0
        scheduler.step(val_dice)
        print(f"Epoch {epoch:03d}/{args.epochs} | loss={np.mean(losses):.4f} | val Dice={val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "architecture": args.architecture,
                "encoder": "resnet34",
                "state_dict": model.state_dict(),
                "validation_dice": best_dice,
                "patch_size": args.patch_size,
                "epoch": epoch,
                "seed": 42,
            }, output_path)
            print(f"  Saved new best checkpoint: {output_path}")
    print(f"Best validation Dice={best_dice:.4f}; checkpoint={output_path}")
    print("Do not select this model as the final model until a separate held-out test set and diameter-reference validation have been evaluated.")


if __name__ == "__main__":
    main()
