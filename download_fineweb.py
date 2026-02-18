#!/usr/bin/env python3
"""Download fineweb-edu sample-10BT and save as a HF Arrow dataset on disk."""

import os

from datasets import DownloadConfig, load_dataset

DEST = "/mnt/skraid0/fineweb"
NUM_PROC = min(os.cpu_count() or 1, 64)

print("Downloading HuggingFaceFW/fineweb-edu (sample-10BT)...")
ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    num_proc=NUM_PROC,
    download_config=DownloadConfig(num_proc=NUM_PROC),
)
print(f"Loaded {len(ds)} examples. Saving to {DEST}...")
ds.save_to_disk(DEST, num_proc=NUM_PROC)
print(f"Done.")
