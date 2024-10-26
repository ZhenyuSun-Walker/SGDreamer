import os
import torch
import clip
from PIL import Image
import numpy as np
import argparse

# 定义函数来处理图像的CLIP IQA计算
def calculate_clip_iqa(image_folder):
    # 加载CLIP模型和预训练的权重
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 定义提示词
    prompts = ["sharp image", "colorful image", "high resolution image"]
    text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

    # 初始化分数列表
    sharpness_scores = []
    colorfulness_scores = []
    resolution_scores = []

    # 遍历文件夹中的所有图像
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            # 计算图像和提示词的特征
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text_inputs)

            # 计算余弦相似度
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).squeeze(0).cpu().numpy()

            # 计算各指标分数
            sharpness_scores.append(similarity[0])
            colorfulness_scores.append(similarity[1])
            resolution_scores.append(similarity[2])

    # 计算平均分数
    avg_sharpness = np.mean(sharpness_scores) if sharpness_scores else 0
    avg_colorfulness = np.mean(colorfulness_scores) if colorfulness_scores else 0
    avg_resolution = np.mean(resolution_scores) if resolution_scores else 0

    return avg_sharpness, avg_colorfulness, avg_resolution


# 定义主函数
def main(input_folder, output_folder):
    # 初始化所有场景的指标列表
    all_sharpness = []
    all_colorfulness = []
    all_resolution = []

    # 遍历input_folder中的所有场景
    for scene in os.listdir(input_folder):
        scene_folder = os.path.join(input_folder, scene, 'images')
        if os.path.isdir(scene_folder):
            print(f"Processing scene: {scene}")
            # 计算当前场景的CLIP IQA
            avg_sharpness, avg_colorfulness, avg_resolution = calculate_clip_iqa(scene_folder)

            # 为当前场景创建输出文件夹
            scene_output_folder = os.path.join(output_folder, scene)
            if not os.path.exists(scene_output_folder):
                os.makedirs(scene_output_folder)

            # 存储当前场景的平均分数
            with open(os.path.join(scene_output_folder, 'metrics.txt'), 'w') as f:
                f.write(f"Scene: {scene}\n")
                f.write(f"Average Sharpness: {avg_sharpness:.4f}\n")
                f.write(f"Average Colorfulness: {avg_colorfulness:.4f}\n")
                f.write(f"Average Resolution: {avg_resolution:.4f}\n")

            # 更新总的指标
            all_sharpness.append(avg_sharpness)
            all_colorfulness.append(avg_colorfulness)
            all_resolution.append(avg_resolution)

    # 计算所有场景的平均分数
    overall_sharpness = np.mean(all_sharpness) if all_sharpness else 0
    overall_colorfulness = np.mean(all_colorfulness) if all_colorfulness else 0
    overall_resolution = np.mean(all_resolution) if all_resolution else 0

    # 存储所有场景的总体平均分数
    with open(os.path.join(output_folder, 'overall_metrics.txt'), 'w') as f:
        f.write(f"Overall Average Sharpness: {overall_sharpness:.4f}\n")
        f.write(f"Overall Average Colorfulness: {overall_colorfulness:.4f}\n")
        f.write(f"Overall Average Resolution: {overall_resolution:.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLIP-based Image Quality Assessment")
    parser.add_argument('-i', '--input', required=True, help="Input folder containing scenes")
    parser.add_argument('-o', '--output', required=True, help="Output folder to save metrics")
    
    args = parser.parse_args()

    main(args.input, args.output)
