#!/bin/bash

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=1

# 修改场景文件夹的路径为相对路径
base_dir="./outputs/results--20241011-205152"

# 遍历场景文件夹
for scene in "$base_dir"/*; do
    if [ -d "$scene" ]; then
        scene_name=$(basename "$scene")
        echo "Processing scene: $scene_name"
        
        # 确保使用相对路径，运行命令并输出日志
        python pano2pers.py --source "$scene/pano.png" --output "$scene_name" 
    fi
done
