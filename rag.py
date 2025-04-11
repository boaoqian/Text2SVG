import time

from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModel
import torch,cairosvg,io,os

model = AutoModel.from_pretrained("/media/qba/Data/Project/DeepLearning/Model/siglip-so400m", device_map="auto")
processor = AutoProcessor.from_pretrained("/media/qba/Data/Project/DeepLearning/Model/siglip-so400m")
img_path = "/media/qba/Data/Project/DeepLearning/Text2SVG/img/"
imgs = os.listdir(img_path)
svgs = []
svg_img = []
TEXT = "SVG illustration of "+"a lighthouse overlooking the ocean"
for img in imgs:
    with open(os.path.join(img_path,img),"r") as f:
        svg = f.read()
        svgs.append(svg)
        imgss = cairosvg.svg2png(svg)
        imgss = Image.open(io.BytesIO(imgss))
        imgss = imgss.convert('RGB')
        svg_img.append(imgss)
sc = []
t = time.time()
with torch.inference_mode():
    for i in range(0,len(svgs),20):
        inputs = processor(text=TEXT, images=svg_img[i:i+20], padding="max_length", return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = torch.sigmoid(logits_per_image)  # these are the probabilities
        sc.append(probs.cpu())
        print(probs.cpu().tolist())
print(time.time()-t)
sc = torch.cat(sc, dim=0)
sc = sc.reshape(-1)
print(sc.argmax(dim=0))
print(imgs[sc.argmax()])