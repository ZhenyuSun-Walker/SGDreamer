import os
import torch
import clip
from PIL import Image
import numpy as np

# 加载CLIP模型和预训练的权重
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 定义提示词
prompts = ["sharp image", #"blurry image", \
           "colorful image", #"dull image", \
            "high resolution image"] #"low resolution image"]
text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

# 初始化分数列表
sharpness_scores = []
colorfulness_scores = []
resolution_scores = []

# 遍历文件夹中的所有图像
folder_path = "metric_folder/images/kitchen_flowmap_llff_10_10"
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
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
        sharpness_score = similarity[0]
        # similarity[0] - similarity[1]
        colorfulness_score = similarity[1]
        # similarity[2] - similarity[3]
        resolution_score = similarity[2]
        # similarity[4] - similarity[5]

        # 将分数添加到列表中
        sharpness_scores.append(sharpness_score)
        colorfulness_scores.append(colorfulness_score)
        resolution_scores.append(resolution_score)

# 计算平均分数
average_sharpness = np.mean(sharpness_scores)
average_colorfulness = np.mean(colorfulness_scores)
average_resolution = np.mean(resolution_scores)

print(f"Average Sharpness Score: {average_sharpness}")
print(f"Average Colorfulness Score: {average_colorfulness}")
print(f"Average Resolution Score: {average_resolution}")
