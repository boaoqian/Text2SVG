from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModel
import torch,cairosvg,io

model = None
processor = None

def get_score(texts,svgs):
    global model
    global processor
    if model == None:
        model = AutoModel.from_pretrained("/media/qba/Data/Project/DeepLearning/Model/siglip-so400m",device_map="auto")
        processor = AutoProcessor.from_pretrained("/media/qba/Data/Project/DeepLearning/Model/siglip-so400m")
    imgs = []
    for i in svgs:
        img = cairosvg.svg2png(i)
        img = Image.open(io.BytesIO(img))
        img = img.convert('RGB')
        imgs.append(img)
    
    inputs = processor(text=texts, images=img, padding="max_length", return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = torch.sigmoid(logits_per_image) # these are the probabilities
    return probs

def svg2img(svg):
    img = cairosvg.svg2png(svg)
    img = Image.open(io.BytesIO(img))
    img = img.convert('RGB')
    return img