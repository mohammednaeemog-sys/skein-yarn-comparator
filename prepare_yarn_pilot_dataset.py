#!/usr/bin/env python3
"""Prepare a tiny, group-separated Skein training smoke-test dataset.

Uses only manually annotated masks. It resolves each mask to an image from the
provided image roots, deduplicates byte-identical image copies, groups images by
name prefix before '.rf.' where possible, and creates a train/val/test layout.

WARNING: With only a handful of source groups this is a pipeline smoke test,
not a meaningful performance evaluation or evidence of 0.017 mm accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized_stem(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_mask"):
        stem = stem[:-5]
    if stem.endswith(" - Copy"):
        stem = stem[:-7]
    return stem


def group_from_stem(stem: str) -> str:
    # Roboflow-export variants sharing e.g. '2_bmp' are kept together.
    if ".rf." in stem:
        return stem.split(".rf.", 1)[0]
    return stem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-dir", type=Path, default=Path("data/yarn_masks/manual/masks"))
    parser.add_argument(
        "--image-root", type=Path, action="append",
        help="Image directory to search; may be supplied multiple times. Defaults to known project folders.",
    )
    parser.add_argument("--out", type=Path, default=Path("data/yarn_masks/pilot_v1"))
    parser.add_argument("--force", action="store_true", help="Delete and recreate output folder if it exists.")
    args = parser.parse_args()

    roots = args.image_root or [
        Path("data/annotation_pilot_5"),
        Path("data/annotation_pilot_more"),
        Path("data/annotation_candidate"),
        Path("data/raw_photos"),
    ]
    roots = [p for p in roots if p.is_dir()]
    if not roots:
        raise SystemExit("No image roots found. Run from the inner project directory or pass --image-root.")
    if not args.manual_dir.is_dir():
        raise SystemExit(f"Manual mask folder not found: {args.manual_dir}")

    masks = sorted(args.manual_dir.glob("*_mask.png"))
    if not masks:
        raise SystemExit(f"No manual *_mask.png files found in {args.manual_dir}")

    images: list[Path] = []
    for root in roots:
        images.extend(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    # Build exact-stem candidates and byte-hash aliases. Do not allow auto-review overlays into source roots.
    by_stem: dict[str, list[Path]] = {}
    hash_to_paths: dict[str, list[Path]] = {}
    for im in images:
        if "overlay" in im.stem.lower() or "preview" in im.stem.lower() or "contact_sheet" in im.stem.lower():
            continue
        by_stem.setdefault(normalized_stem(im), []).append(im)
        try:
            hash_to_paths.setdefault(sha256(im), []).append(im)
        except OSError:
            pass

    records: list[dict[str, object]] = []
    missing: list[str] = []
    for mask_path in masks:
        mask_key = normalized_stem(mask_path)
        matches = by_stem.get(mask_key, [])
        if not matches:
            missing.append(mask_path.name)
            continue
        # Prefer the input image path with the exact stem; copies with same bytes resolve to one canonical image.
        image_path = sorted(matches, key=lambda p: (" - Copy" in p.name, len(str(p)), str(p)))[0]
        img_hash = sha256(image_path)
        aliases = hash_to_paths.get(img_hash, [image_path])
        # If a renamed practice copy is byte-identical to its Roboflow original, recover the Roboflow source group.
        canonical = sorted(
            aliases,
            key=lambda p: (".rf." not in p.stem, " - Copy" in p.name, len(str(p)), str(p)),
        )[0]
        source_group = group_from_stem(canonical.stem)
        records.append({
            "key": mask_key,
            "image": image_path,
            "mask": mask_path,
            "group": source_group,
            "hash": img_hash,
        })

    if missing:
        print("WARNING: masks without matching images were omitted:")
        for name in missing:
            print("  ", name)
    if not records:
        raise SystemExit("No image/mask pairs could be resolved.")

    unique_groups = sorted({str(r["group"]) for r in records})
    if len(unique_groups) < 3:
        raise SystemExit(f"Need at least 3 source groups for train/val/test smoke-test split; found {len(unique_groups)}")

    # Deterministic grouping: first 70% train, next 15% val, remainder test.
    # The ordering is lexical and recorded so the split is reproducible and auditable.
    n_groups = len(unique_groups)
    n_train = max(1, int(n_groups * 0.70))
    n_val = max(1, int(n_groups * 0.15))
    if n_train + n_val >= n_groups:
        n_train = max(1, n_groups - 2)
        n_val = 1
    split_for_group: dict[str, str] = {}
    for i, group in enumerate(unique_groups):
        if i < n_train:
            split_for_group[group] = "train"
        elif i < n_train + n_val:
            split_for_group[group] = "val"
        else:
            split_for_group[group] = "test"

    if args.out.exists():
        if not args.force:
            raise SystemExit(f"Output already exists: {args.out}. Use --force to recreate it, or choose another --out.")
        shutil.rmtree(args.out)

    counts = {"train": 0, "val": 0, "test": 0}
    manifest_lines = [
        "pair_key\tsource_group\tsplit\timage_source\tmanual_mask\timage_sha256",
    ]
    for rec in sorted(records, key=lambda r: str(r["key"])):
        key = str(rec["key"])
        group = str(rec["group"])
        split = split_for_group[group]
        image_path = Path(rec["image"])
        mask_path = Path(rec["mask"])
        out_image_dir = args.out / "images" / split
        out_mask_dir = args.out / "masks" / split
        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)
        out_image = out_image_dir / f"{key}{image_path.suffix.lower()}"
        out_mask = out_mask_dir / f"{key}.png"
        shutil.copy2(image_path, out_image)
        shutil.copy2(mask_path, out_mask)
        counts[split] += 1
        manifest_lines.append(
            f"{key}\t{group}\t{split}\t{image_path.as_posix()}\t{mask_path.as_posix()}\t{rec['hash']}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "split_manifest.tsv").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    (args.out / "SMOKE_TEST_WARNING.txt").write_text(
        "This dataset is a tiny exploratory training-pipeline smoke test. It contains manually annotated masks only.\n"
        "It is not large enough for meaningful model selection, validation statistics, or claims of 0.017 mm accuracy.\n"
        "Source groups were kept in a single split using filename prefix before '.rf.' where available.\n",
        encoding="utf-8",
    )

    print(f"Manual masks found: {len(masks)}")
    print(f"Image-mask pairs prepared: {len(records)}")
    print(f"Source groups: {len(unique_groups)}")
    print("Source-group split:")
    for group in unique_groups:
        print(f"  {group}: {split_for_group[group]}")
    print("Pair counts:")
    for split, count in counts.items():
        print(f"  {split}: {count}")
    print(f"Output: {args.out}")
    print("WARNING: split metrics from this tiny dataset are not evidence of generalization or physical measurement accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
