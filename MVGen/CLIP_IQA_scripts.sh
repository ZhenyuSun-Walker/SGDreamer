#!/bin/bash

# 设置输入文件夹和输出文件夹路径
INPUT_FOLDER="../generate_mvimages/results--20241101-000523"
OUTPUT_FOLDER="../generate_mvimages/results--20241101-000523"

# 执行CLIP IQA计算
python CLIP_IQA.py -i $INPUT_FOLDER -o $OUTPUT_FOLDER

echo "CLIP IQA 计算完成，结果已保存到 $OUTPUT_FOLDER"
