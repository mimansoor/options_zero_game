#!/usr/bin/env python3

import os
import glob
import shutil

# Pattern to search for ckpt_best.pth.tar files
source_pattern = './options_zero_game_muzero_final_agent_v1.3_ns100_upc1000_bs256*/ckpt/ckpt_best.pth.tar'
target_dir = './best_ckpt/'
target_path = os.path.join(target_dir, 'ckpt_best.pth.tar')

# Find all matching checkpoint files
ckpt_files = glob.glob(source_pattern)
if not ckpt_files:
    print("No checkpoint files found.")
    exit(1)

# Get the latest one based on modification time
latest_ckpt = max(ckpt_files, key=os.path.getmtime)

# Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Copy the latest checkpoint to the target location
shutil.copy2(latest_ckpt, target_path)

print(f"Copied latest checkpoint:\n{latest_ckpt} → {target_path}")

