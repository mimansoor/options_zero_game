#!/bin/bash

SRC="/home/immansoo/options_zero_game/experiments/options_zero_game_muzero_agent_v2.2-Final_ns50_upc2000_bs256/ckpt/ckpt_best.pth.tar"
DEST="gdrive:/options_zero_game/"

while true
do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Uploading to Google Drive..."
    rclone copy "$SRC" "$DEST" --progress

    echo "Waiting 1 hour..."
    sleep 3600
done

