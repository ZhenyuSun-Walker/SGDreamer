#!/bin/bash

# Specify the path to the indoor folder
INDOOR_PATH="rendered_images/GPT_dataset/indoor"

# Initialize total score and count for calculating the average across all scenes
total_score=0
total_count=0

# Loop over each scene folder within the indoor folder
for scene in "$INDOOR_PATH"/*/; do
  # Path to the iteration_4000 folder in the current scene
  iteration_path="${scene}iteration_4000"

  # Initialize variables to calculate the average score for the current scene
  scene_score=0
  scene_count=0

  # Check if the iteration_4000 folder exists
  if [ -d "$iteration_path" ]; then
    # Loop over each image in the iteration_4000 folder
    for image in "$iteration_path"/*.png; do
      # Calculate CLIP score using the Python script
      clip_score=$(python calculate_clip_score.py "$image")
      
      # Add the score to the scene's total score and increment the count
      scene_score=$(echo "$scene_score + $clip_score" | bc)
      scene_count=$((scene_count + 1))
    done

    # Calculate and print the average score for the current scene
    if [ "$scene_count" -ne 0 ]; then
      scene_avg=$(echo "$scene_score / $scene_count" | bc -l)
      echo "Average CLIP score for scene $(basename "$scene"): $scene_avg"
    fi

    # Add the scene's score and count to the total score and count for all scenes
    total_score=$(echo "$total_score + $scene_score" | bc)
    total_count=$((total_count + scene_count))
  else
    echo "No iteration_4000 folder found for scene $(basename "$scene")"
  fi
done

# Calculate and print the average score across all scenes
if [ "$total_count" -ne 0 ]; then
  overall_avg=$(echo "$total_score / $total_count" | bc -l)
  echo "Overall average CLIP score for all scenes in indoor: $overall_avg"
fi
