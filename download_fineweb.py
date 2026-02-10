#!/usr/bin/env python3
"""Download fineweb-edu sample-10BT and save as a HF Arrow dataset on disk."""

from datasets import load_dataset

DEST = "/mnt/skraid0/fineweb"

print("Downloading HuggingFaceFW/fineweb-edu (sample-10BT)...")
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
print(f"Loaded {len(ds)} examples. Saving to {DEST}...")
ds.save_to_disk(DEST)
print(f"Done.")
