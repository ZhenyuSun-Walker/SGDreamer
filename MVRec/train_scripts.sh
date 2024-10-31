#!/bin/bash

# Define the base directory containing the scenes
SCENE_DIR="data/GPT_dataset_1024_old/indoor"

# Iterate over each subdirectory (scene) in the SCENE_DIR
for scene in "$SCENE_DIR"/*; do
    if [ -d "$scene" ]; then
        # Extract the scene name from the directory path
        scene_name=$(basename "$scene")
        
        # Run the train.py script with the scene
        CUDA_VISIBLE_DEVICES=7 python train.py -s "$SCENE_DIR/$scene_name/colmap/$scene_name" --name "$scene_name" -o 
    fi
done

            
