#!/bin/bash

SRC="/home/immansoo/options_zero_game/experiments/options_zero_game_muzero_agent_v2.2-Final_ns50_upc1000_bs512/ckpt/ckpt_best.pth.tar"
DEST="/home/immansoo/gdrive/options_zero_game"

while true
do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Copying file to Google Drive..."
    cp "$SRC" "$DEST"
    
    if [ $? -eq 0 ]; then
        echo "Copy successful."
    else
        echo "Copy failed!"
    fi
    
    echo "Waiting 30 minutes..."
    sleep 1800
done

