from tqdm import tqdm
from PIL import Image

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import argparse



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
# 加载数据处理器
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 添加命令行参数解析
parser = argparse.ArgumentParser(description="CLIP Score Calculation")
parser.add_argument('--input_folder', type=str, required=True, help="Path to the input images folder")
parser.add_argument('--output_folder', type=str, required=True, help="Path to the output folder where results will be saved")
parser.add_argument('--prompt_file', type=str, required=True, help="Path to the text prompt file")
args = parser.parse_args()

def get_all_folders(folder_path):
    all_files = os.listdir(folder_path)
    folder_files = [file for file in all_files if os.path.isdir(os.path.join(folder_path, file))]
    folder_paths = [os.path.join(folder_path, folder_file) for folder_file in folder_files]
    return folder_paths

def get_all_images(folder_path):
    all_files = os.listdir(folder_path)
    image_files = [file for file in all_files if file.endswith((".jpg", ".png", ".jpeg"))]
    image_paths = [os.path.join(folder_path, image_file) for image_file in image_files]
    return image_paths

def calculate_clip_scores_for_all_categories(images_folder_path, text_prompts, output_folder):
    category_folders = get_all_folders(images_folder_path)
    print(f"Found {len(category_folders)} categories.")

    category_clip_scores = {}
    category_mean_scores = {}

    for category_folder, text_prompt in tqdm(zip(category_folders, text_prompts), total=len(category_folders), desc="Processing categories"):
        print(f"Processing category: {category_folder}")
        category_name = os.path.basename(category_folder)
        images_path = get_all_images(category_folder)
        print(f"Found {len(images_path)} images in category {category_name}.")
        
        if len(images_path) == 0:
            print(f"No images found in category {category_name}. Skipping...")
            continue

        clip_score = get_clip_score(images_path, [text_prompt])
        category_clip_scores[category_name] = clip_score
        mean_score = torch.mean(clip_score, dim=0)
        category_mean_scores[category_name] = mean_score.item()

        # 保存每个场景的结果到对应的文件夹
        category_output_folder = os.path.join(output_folder, category_name)
        if not os.path.exists(category_output_folder):
            os.makedirs(category_output_folder)
        np.savetxt(os.path.join(category_output_folder, f"{category_name}_clip_scores.txt"), clip_score.detach().cpu().numpy(), fmt="%.4f")
        with open(os.path.join(category_output_folder, f"{category_name}_mean_score.txt"), 'w') as f:
            f.write(f"Mean Score: {mean_score.item():.4f}\n")

    return category_clip_scores, category_mean_scores

def get_clip_score(images_path, text):
    images = [Image.open(image_path) for image_path in images_path]
    inputs = processor(text=text, images=images, return_tensors="pt", padding=True)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    return logits_per_image

def read_prompts_from_file(file_path):
    with open(file_path, 'r') as file:
        prompts = file.readlines()
    prompts = [prompt.strip() for prompt in prompts]
    return prompts

# 主程序逻辑
text_prompts = read_prompts_from_file(args.prompt_file)
category_clip_scores, category_mean_scores = calculate_clip_scores_for_all_categories(args.input_folder, text_prompts, args.output_folder)

# 保存所有场景的平均结果到主输出文件夹
mean_scores_output_path = os.path.join(args.output_folder, "clip_scores.txt")
with open(mean_scores_output_path, 'w') as f:
    for category, mean_score in category_mean_scores.items():
        f.write(f"Category: {category}, Mean Score: {mean_score:.4f}\n")

print(f"所有场景的平均分数已保存到 {mean_scores_output_path}")
