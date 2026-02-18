#!/usr/bin/env python3
"""Download C4 (en) train subset and full validation split, save as Arrow on disk.

Train: streams ~23M documents (~10B tokens) from allenai/c4 "en" config.
       Covers 30M (3B), 50M (5B), and 100M (10B) models at 100N token budget.
Validation: full C4-en validation split (~365K docs).

Estimated disk usage: 10-15GB Arrow format.
Destination: /home/aaron/c4/{train,validation}/
"""

import os

from datasets import DownloadConfig, load_dataset

DEST = "/home/aaron/c4"
TRAIN_DOCS = 23_000_000  # ~10B tokens at ~430 tokens/doc average
NUM_PROC = min(os.cpu_count() or 1, 64)
DOWNLOAD_CONFIG = DownloadConfig(num_proc=NUM_PROC)


def download_train():
    train_path = os.path.join(DEST, "train")
    if os.path.isdir(train_path):
        print(f"Train split already exists at {train_path}, skipping.")
        return

    print(f"Streaming allenai/c4 (en) train split, taking {TRAIN_DOCS:,} documents...")
    ds = load_dataset(
        "allenai/c4",
        name="en",
        split=f"train[:{TRAIN_DOCS}]",
        num_proc=NUM_PROC,
        download_config=DOWNLOAD_CONFIG,
    )
    print(f"Loaded {len(ds):,} documents. Saving to {train_path}...")
    ds.save_to_disk(train_path, num_proc=NUM_PROC)
    print(f"Train split saved.")


def download_validation():
    val_path = os.path.join(DEST, "validation")
    if os.path.isdir(val_path):
        print(f"Validation split already exists at {val_path}, skipping.")
        return

    print("Downloading allenai/c4 (en) validation split...")
    ds = load_dataset(
        "allenai/c4",
        name="en",
        split="validation",
        num_proc=NUM_PROC,
        download_config=DOWNLOAD_CONFIG,
    )
    print(f"Loaded {len(ds):,} documents. Saving to {val_path}...")
    ds.save_to_disk(val_path, num_proc=NUM_PROC)
    print(f"Validation split saved.")


if __name__ == "__main__":
    os.makedirs(DEST, exist_ok=True)
    download_train()
    download_validation()
    print("Done.")
