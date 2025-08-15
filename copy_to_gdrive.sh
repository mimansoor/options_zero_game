#!/bin/bash

SRC="/home/immansoo/options_zero_game/experiments/options_zero_game_muzero_agent_v2.2-Final_ns35_upc500_bs256/ckpt/ckpt_best.pth.tar"
DEST="/home/immansoo/gdrive/options_zero_game"

while true
do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Syncing..."
    rsync -u "$SRC" "$DEST"

    echo "Waiting 30 minutes..."
    sleep 1800
done

