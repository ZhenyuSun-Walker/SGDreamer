from tqdm import tqdm
from PIL import Image
import torch
import os
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import argparse

# 设置GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

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

def calculate_clip_scores_for_iterations(images_folder_path, text_prompt, output_folder):
    iteration_folders = get_all_folders(images_folder_path)
    print(f"Found {len(iteration_folders)} iterations.")

    iteration_clip_scores = {}
    iteration_mean_scores = {}

    for iteration_folder in tqdm(iteration_folders, desc="Processing iterations"):
        iteration_name = os.path.basename(iteration_folder)
        images_path = get_all_images(iteration_folder)
        
        if len(images_path) == 0:
            print(f"No images found in iteration {iteration_name}. Skipping...")
            continue

        clip_score = get_clip_score(images_path, text_prompt)
        iteration_clip_scores[iteration_name] = clip_score
        mean_score = torch.mean(clip_score, dim=0)
        iteration_mean_scores[iteration_name] = mean_score.item()

        # save every iteration's result to corresponding folder
        iteration_output_folder = os.path.join(output_folder, iteration_name)
        if not os.path.exists(iteration_output_folder):
            os.makedirs(iteration_output_folder)
        np.savetxt(os.path.join(iteration_output_folder, f"{iteration_name}_clip_scores.txt"), clip_score.detach().cpu().numpy(), fmt="%.4f")
        with open(os.path.join(iteration_output_folder, f"{iteration_name}_mean_score.txt"), 'w') as f:
            f.write(f"Mean Score: {mean_score.item():.4f}\n")

    return iteration_clip_scores, iteration_mean_scores

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

# main function 
text_prompts = read_prompts_from_file(args.prompt_file)
category_clip_scores, category_mean_scores = calculate_clip_scores_for_iterations(args.input_folder, text_prompts, args.output_folder)

# Save the average results for all scenes to the main output folder
mean_scores_output_path = os.path.join(args.output_folder, "clip_scores.txt")
with open(mean_scores_output_path, 'w') as f:
    for category, mean_score in category_mean_scores.items():
        f.write(f"Category: {category}, Mean Score: {mean_score:.4f}\n")

print(f"All category mean scores have been saved to {mean_scores_output_path}")

