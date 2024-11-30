#!/bin/bash
CATEGORY="outdoor"

# 设置输入文件夹和输出文件夹路径
INPUT_FOLDER="data/GPT_dataset/$CATEGORY"
OUTPUT_FOLDER="metrics/GPT_dataset/$CATEGORY"

# 执行CLIP IQA计算
python CLIP_IQA.py -i $INPUT_FOLDER -o $OUTPUT_FOLDER

echo "CLIP IQA 计算完成，结果已保存到 $OUTPUT_FOLDER"
