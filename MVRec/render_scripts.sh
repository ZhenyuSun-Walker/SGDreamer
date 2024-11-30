#!/bin/bash

# Set the base directory for GPT_dataset
BASE_DIR="rendered_images/GPT_dataset/indoor"

# Iterate over each scene folder in the indoor directory
for SCENE_DIR in "$BASE_DIR"/*; do
    # Check if it's a directory
    if [ -d "$SCENE_DIR" ]; then
        # Define the iteration_4000 directory containing images
        IMG_DIR="$SCENE_DIR/iteration_4000"

        # Check if iteration_4000 exists and contains images
        if [ -d "$IMG_DIR" ] && [ "$(ls -A "$IMG_DIR"/*.png 2>/dev/null)" ]; then
            # Define the output video file
            OUTPUT_VIDEO="$SCENE_DIR/scene_video.mp4"

            # Use ffmpeg to create a video from the images
            # Here, scale the images to have even height and width and adjust the frame rate
            ffmpeg -framerate 5 -pattern_type glob -i "$IMG_DIR/*.png" \
                -vf "fps=24,setpts=N/FRAME_RATE/TB,scale=trunc(iw/2)*2:trunc(ih/2)*2" \
                -c:v libx264 -pix_fmt yuv420p "$OUTPUT_VIDEO"

            echo "Rendered video saved to $OUTPUT_VIDEO"
        else
            echo "No images found in $IMG_DIR, skipping..."
        fi
    fi
done
