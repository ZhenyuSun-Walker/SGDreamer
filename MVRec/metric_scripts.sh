#!/bin/bash
# Initialize variables to store the sum of metrics and the scene count
total_psnr=0
total_lpips=0
total_ssim=0
scene_count=0
category="ablation/wo_densi/all"
dataset="GPT_dataset_final"

# Loop through each scene directory in the indoor folder
for scene_dir in metrics/$dataset/$category/*; do
    if [ -d "$scene_dir" ]; then
        # Read the metrics.json file
        metric_file="$scene_dir/metrics.json"
        if [ -f "$metric_file" ]; then
            # Use jq to extract psnr, lpips, and ssim values
            psnr=$(jq '.psnr' "$metric_file")
            lpips=$(jq '.lpips' "$metric_file")
            ssim=$(jq '.ssim' "$metric_file")
            
            # Ensure the values are valid and add them to the totals
            if [ -n "$psnr" ] && [ -n "$lpips" ] && [ -n "$ssim" ]; then
                total_psnr=$(echo "$total_psnr + $psnr" | bc)
                total_lpips=$(echo "$total_lpips + $lpips" | bc)
                total_ssim=$(echo "$total_ssim + $ssim" | bc)
                
                # Increment the scene count
                scene_count=$((scene_count + 1))
            fi
        fi
    fi
done

# Calculate the average values
if [ $scene_count -gt 0 ]; then
    avg_psnr=$(echo "scale=6; $total_psnr / $scene_count" | bc)
    avg_lpips=$(echo "scale=6; $total_lpips / $scene_count" | bc)
    avg_ssim=$(echo "scale=6; $total_ssim / $scene_count" | bc)
else
    avg_psnr=0
    avg_lpips=0
    avg_ssim=0
fi

# Create the mean_metric.json file and write the average values
mean_metric_file="metrics/$dataset/$category/mean_metric.json"
printf '{"psnr": %.6f, "lpips": %.6f, "ssim": %.6f}\n' "$avg_psnr" "$avg_lpips" "$avg_ssim" > "$mean_metric_file"

echo "Average metrics calculated and stored in $mean_metric_file"
