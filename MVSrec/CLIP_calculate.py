 
from tqdm import tqdm
from PIL import Image
import torch
import os
import numpy as np
 
from transformers import CLIPProcessor, CLIPModel
 
# 设置GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
torch.cuda.set_device(0)
# 加载模型
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
# 加载数据处理器
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
 
 
def get_all_folders(folder_path):
    # 获取文件夹中的所有文件和文件夹
    all_files = os.listdir(folder_path)
    # 过滤所有的文件夹
    folder_files = [file for file in all_files if os.path.isdir(os.path.join(folder_path, file))]
    # 将文件夹的路径添加到一个列表中
    folder_paths = [os.path.join(folder_path, folder_file) for folder_file in folder_files]
    # 返回列表
    return folder_paths
 
def get_all_images(folder_path):
    # 获取文件夹中的所有文件和文件夹
    all_files = os.listdir(folder_path)
    # 过滤所有的图片文件
    image_files = [file for file in all_files if file.endswith((".jpg", ".png", ".jpeg"))]
    # 将图片文件的路径添加到一个列表中
    image_paths = [os.path.join(folder_path, image_file) for image_file in image_files]
    # 返回列表
    return image_paths
 
def calculate_clip_scores_for_all_categories(images_folder_path, text_prompts):
    # 获取所有的类别文件夹
    category_folders = get_all_folders(images_folder_path)
 
    # 初始化一个字典来存储每个类别的 Clip Score
    category_clip_scores = {}
    # 初始化一个字典来存储每个类别的平均值
    category_mean_scores = {}
 
    # 遍历每个类别文件夹
    for category_folder, text_prompt in tqdm(zip(category_folders, text_prompts),total=len(category_folders),desc="Processing categories"):
        # 获取类别名称
        category_name = os.path.basename(category_folder)
        # 获取该类别下的所有图片
        images_path = get_all_images(category_folder)
        # 计算该类别的 Clip Score
        clip_score = get_clip_score(images_path, [text_prompt])
        # 将 Clip Score 存储在字典中
        category_clip_scores[category_name] = clip_score
        # 计算平均值
        mean_score = torch.mean(clip_score, dim=0)
        # 将平均值存储在字典中
        category_mean_scores[category_name] = mean_score.item()  # 使用.item()将torch.Tensor转换为Python的标准数据类型
 
    return category_clip_scores,category_mean_scores
 
 
def get_clip_score(images_path, text):
    images = [Image.open(image_path) for image_path in images_path]
    inputs = processor(text=text, images=images, return_tensors="pt", padding=True)
 
    # 将输入数据移动到GPU
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
 
    outputs = model(**inputs)
    # print(outputs)
 
    logits_per_image = outputs.logits_per_image
    # print(logits_per_image, logits_per_image.shape)  # 1,4
    # probs = logits_per_image.softmax(dim=1)
    # 计算平均值
    # mean_score = torch.mean(logits_per_image,dim=0)
    # print(f"CLIP-T的平均值是:{mean_score}")
    return logits_per_image
 
 
# 从文件中获取文本提示
def read_prompts_from_file(file_path):
    with open(file_path, 'r') as file:
        prompts = file.readlines()
    # 去除每行末尾的换行符
    prompts = [prompt.strip() for prompt in prompts]
    return prompts
 
# 使用函数读取文本提示
text_prompts = read_prompts_from_file('metric_folder/text/kitchen.txt')
# 提示文本列表，每个类别对应一个提示文本
# text_prompts = ['a bird flying in the sky', 'a cat playing with a ball','a dog running in the park' ]
 
# 计算所有类别的 Clip Score 和平均值
category_clip_scores, category_mean_scores = calculate_clip_scores_for_all_categories("metric_folder/images", text_prompts)
 
# 打印结果
for category, clip_score in category_clip_scores.items():
    clip_score_list = clip_score.tolist()
    for score in clip_score_list:
        print(f"Category: {category}, Clip Score: {score[0]:.4f}")
 
for category, mean_score in category_mean_scores.items():
    print(f"Category: {category}, Mean Score: {mean_score:.4f}")
 
 
  
# import torch
# from tqdm import tqdm
# from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
# # from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
# import os
# import cv2
 
# # 设置GPU
 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# torch.cuda.set_device(6)
 
# # Load the CLIP model
# model_ID = "openai/clip-vit-large-patch14"
# model = CLIPModel.from_pretrained(model_ID).to(device)
# preprocess = CLIPImageProcessor.from_pretrained(model_ID)
 
# def get_all_folders(folder_path):
#     # 获取文件夹中的所有文件和文件夹
#     all_files = os.listdir(folder_path)
#     # 过滤所有的文件夹
#     folder_files = [file for file in all_files if os.path.isdir(os.path.join(folder_path, file))]
#     # 将文件夹的路径添加到一个列表中
#     folder_paths = [os.path.join(folder_path, folder_file) for folder_file in folder_files]
#     # 返回列表
#     return folder_paths
 
# def get_all_images(folder_path):
#     # 获取文件夹中的所有文件和文件夹
#     all_files = os.listdir(folder_path)
#     # 过滤所有的图片文件
#     image_files = [file for file in all_files if file.endswith((".jpg", ".png", ".jpeg"))]
#     # 将图片文件的路径添加到一个列表中
#     image_paths = [os.path.join(folder_path, image_file) for image_file in image_files]
#     # 返回列表
#     return image_paths
 
# # Define a function to load an image and preprocess it for CLIP
# def load_and_preprocess_image(image_path):
#     # Load the image from the specified path
#     image = Image.open(image_path)
#     # 使用预先定义的函数preprocess函数对图像进行预处理
#     image = preprocess(image, return_tensors="pt")
#     # 返回预处理后的图像
#     return image
 
# def calculate_img_scores_for_all_images(images_folder_path1, images_folder_path2):
#     # 获取所有的类别文件夹
#     folders1 = get_all_folders(images_folder_path1)
#     folders2 = get_all_folders(images_folder_path2)
 
#     # 初始化一个字典来存储每个类别的 img Score
#     category_img_scores = {}
 
#     # 遍历每个类别文件夹
#     for folder1, folder2 in tqdm(zip(folders1, folders2),total=len(folders1),desc="Processing"):
#         # 获取类别名称
#         name = os.path.basename(folder1)
#         # 获取该类别下的所有图片
#         images_path1 = get_all_images(folder1)
#         images_path2 = get_all_images(folder2)
#         # 初始化一个变量来存储该类别的所有图像分数
#         img_scores = 0
#         # 遍历每个图像路径，并计算其与其他的图像的分数
#         for image1_path in tqdm(images_path1,total=len(images_path1),desc="Processing images"):
#             # img_score_mean = 0
#             for image2_path in images_path2:
#                 img_score = clip_img_score(image1_path,image2_path)
#                 img_scores += img_score
#                 # img_score_mean +=img_score
#         # 计算该类别的平均分数，并将 Clip Score 存储在字典中
#         category_img_scores[name] = img_scores / (len(images_path1) * len(images_path2))
 
#     return category_img_scores
 
 
# def clip_img_score(img1_path, img2_path):
#     # Load the two samples_images and preprocess them for CLIP
#     image_a = load_and_preprocess_image(img1_path)["pixel_values"].to(device)
#     image_b = load_and_preprocess_image(img2_path)["pixel_values"].to(device)
 
#     # Calculate the embeddings for the samples_images using the CLIP model
#     with torch.no_grad():
#         # 使用CLIP 模型计算两个图像的嵌入
#         embedding_a = model.get_image_features(image_a)
#         embedding_b = model.get_image_features(image_b)
 
#     # Calculate the cosine similarity between the embeddings
#     # 通过计算余弦相似度来比较两个嵌入的相似性
#     similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
#     # 返回相似度分数，这个分数越高，表示两个图像越相似
#     return similarity_score.item()
 
 
# images_path1 = "./samples_images"
# images_path2 = "./regulation_images"
 
# # 计算图像的相似度分数
# img_scores = calculate_img_scores_for_all_images(images_path1, images_path2)
# # 打印图像的相似度分数
# for category, score in img_scores.items():
#     print(f"Image similarity score for {category}: {score:.4f}")