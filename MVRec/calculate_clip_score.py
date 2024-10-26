import sys
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor  # Ensure you have the transformers library installed

def calculate_clip_score(image_path):
    # Set GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)

    # Load the model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Load and preprocess the image
    image = Image.open(image_path)
    inputs = processor(text=["A relevant caption for the image"], images=image, return_tensors="pt", padding=True).to(device)

    # Calculate image and text features
    with torch.no_grad():
        outputs = model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

    # Calculate cosine similarity as the CLIP score
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    clip_score = (image_features @ text_features.T).item()
    return clip_score

if __name__ == "__main__":
    image_path = sys.argv[1]
    clip_score = calculate_clip_score(image_path)
    print(clip_score)
