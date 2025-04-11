from openai import OpenAI
from tqdm import tqdm
import time, json
import concurrent.futures
import Utils
from transformers import BertTokenizer, BertModel
import torch,os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_name = "deepseek"
# model_name = "openai"

if model_name == "openai":
    import os
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"

deepseek_client = OpenAI(api_key="sk-key", base_url="https://api.deepseek.com")
openai_client = OpenAI(api_key="sk-key")

def get_formated_text(description):
    prompt = f"""Generate SVG code to visually represent the following text description, while respecting the given constraints.
    <constraints>
    * **Allowed Elements:** `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
    * **Allowed Attributes:** `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
    </constraints>

    <example>
    <description>"A vast desert stretches endlessly, with golden sand dunes rolling under a brilliant blue sky."</description>
    <think>
    Let's break down the description:
        Vast Desert: This suggests a large, empty area. We will represent this using a large rectangle or polygon for the desert landscape.
        Golden Sand Dunes: This can be depicted as a series of rolling hills or undulating curves. We can use path elements to create these shapes, filling them with a golden color.
        Brilliant Blue Sky: The sky would be represented by a solid background color, so the whole upper part of the canvas will be filled with blue.
    ### Step 1: Draw the Blue Sky
    In this step, we draw a blue rectangle representing the sky in the upper half of the canvas.
    ```svg
    <svg viewBox="0 0 256 256" width="256" height="256">
    <!-- Sky (blue) -->
    <rect x="0" y="0" width="256" height="128" fill="rgb(0, 121, 184)"/>
    </svg>
    ```
    ### Step 2: Add the Desert (Golden Sand)
    In this step, a golden sand desert is added as a rectangle covering the bottom half of the canvas.
    ```svg
    <svg viewBox="0 0 256 256" width="256" height="256">
    <!-- Sky (blue) -->
    <rect x="0" y="0" width="256" height="128" fill="rgb(0, 121, 184)"/>

    <!-- Desert (golden sand) -->
    <rect x="0" y="128" width="256" height="128" fill="rgb(255, 223, 94)"/>
    </svg>
    ```
    ### Step 3: Add the Sand Dunes (Using Paths)
    Next, sand dunes are added using `path` elements to create rolling hills with a curved shape.
    ```svg
    <svg viewBox="0 0 256 256" width="256" height="256">
    <!-- Sky (blue) -->
    <rect x="0" y="0" width="256" height="128" fill="rgb(0, 121, 184)"/>

    <!-- Desert (golden sand) -->
    <rect x="0" y="128" width="256" height="128" fill="rgb(255, 223, 94)"/>

    <!-- Sand dunes -->
    <path d="M0,170 C50,150 80,180 130,160 C180,140 210,170 256,150" fill="rgb(255, 193, 68)" />
    </svg>
    ```
    ### Step 4: Add More Sand Dunes (Different Colors and Curves)
    Finally, we add more dunes with different shapes and colors to add variation and depth to the landscape.
    ```svg
    <svg viewBox="0 0 256 256" width="256" height="256">
    <!-- Sky (blue) -->
    <rect x="0" y="0" width="256" height="128" fill="rgb(0, 121, 184)"/>

    <!-- Desert (golden sand) -->
    <rect x="0" y="128" width="256" height="128" fill="rgb(255, 223, 94)"/>

    <!-- Sand dunes -->
    <path d="M0,170 C50,150 80,180 130,160 C180,140 210,170 256,150" fill="rgb(255, 193, 68)" />
    <path d="M0,190 C60,170 100,190 140,180 C180,160 210,190 256,170" fill="rgb(255, 194, 65)" />
    </svg>
    ```
    </example>
    <info>
    Common colors and rgb:
    Red: rgb(255, 0, 0)  
    Green: rgb(0, 255, 0)  
    Blue: rgb(0, 0, 255)  
    Yellow: rgb(255, 255, 0)  
    Cyan: rgb(0, 255, 255)  
    Magenta: rgb(255, 0, 255)  
    Black: rgb(0, 0, 0)  
    White: rgb(255, 255, 255)  
    Gray: rgb(128, 128, 128)  
    Orange: rgb(255, 165, 0)  
    Pink: rgb(255, 192, 203)  
    Purple: rgb(128, 0, 128)  
    Brown: rgb(165, 42, 42)  
    Lime: rgb(0, 255, 0)  
    Olive: rgb(128, 128, 0)  
    Maroon: rgb(128, 0, 0)  
    Navy: rgb(0, 0, 128)  
    Teal: rgb(0, 128, 128)  
    Turquoise: rgb(64, 224, 208)  
    Indigo: rgb(75, 0, 130)  
    Violet: rgb(238, 130, 238)  
    Gold: rgb(255, 215, 0)  
    Silver: rgb(192, 192, 192)  
    </info>
    Please ensure that the generated SVG code is well-formed, valid, and strictly follows these constraints, with a certain level of detail. Focus on a clear and concise representation of the input description within the given limitations. 
    Always give the complete SVG code with nothing omitted. Never use an ellipsis.
    Visualize the svg after each reasoning step.,output you svg finally.
    <description>"{description}"</description>

    Now is your turn!"""
    return prompt

# 初始化 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('/media/qba/Data/Project/DeepLearning/Model/bert-base-uncased')
model = BertModel.from_pretrained('/media/qba/Data/Project/DeepLearning/Model/bert-base-uncased')


def get_sentence_embedding(sentence):
    # 对句子进行编码
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    # 获取BERT的输出
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取[CLS]标记的表示作为句子的嵌入
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return cls_embedding.numpy()

img_path = "/media/qba/Data/Project/DeepLearning/Text2SVG/img-svg/"
imgs = os.listdir(img_path)
def find_most_similar_sentence(query, candidate_sentences):
    # 获取查询句子的嵌入
    query_embedding = get_sentence_embedding(query)

    # 获取候选句子的嵌入
    candidate_embeddings = np.array([get_sentence_embedding(sentence) for sentence in candidate_sentences])

    # 计算查询句子和所有候选句子之间的余弦相似度
    similarity_scores = cosine_similarity([query_embedding], candidate_embeddings)

    # 找到相似度最高的句子
    most_similar_idx = np.argmax(similarity_scores)
    with open(os.path.join(img_path, imgs[most_similar_idx]), "r") as f:
        svg_text = f.read()
    return svg_text, similarity_scores[0][most_similar_idx]

def get_prompt_a(des):
    cand = [i.split(".")[0].replace(" ","_") for i in imgs]
    base,score = find_most_similar_sentence(des, cand)
    print(score)
    prompt = f"""Modify the SVG code to make its representation more accurately match the description.
    <constraints>
    * **Allowed Elements:** `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
    * **Allowed Attributes:** `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
    </constraints>
    <example>
    User: 
    <Original SVG><svg viewBox="0 0 256 256" width="256" height="256">
    <!-- Sky (blue) -->
    <rect x="0" y="0" width="256" height="128" fill="rgb(0, 121, 184)"/>

    <!-- Grassy Hill -->
    <path d="M0,128 C50,100 100,150 150,120 C200,90 256,128 256,128 V256 H0 Z" fill="rgb(34, 139, 34)"/>

    <!-- Additional Grass Layer -->
    <path d="M0,150 C50,130 100,170 150,140 C200,110 256,150 256,150 V256 H0 Z" fill="rgb(50, 205, 50)"/>
</svg></Original SVG>
    <description>"a grassy hill under a blue sky"</description>
    Assistant:
    To better represent the description of "a grassy hill under a blue sky," I will enhance the SVG by adding more detail to the grassy hill and refining the sky. Specifically, I will:

1. Add a gradient to the sky to give it a more realistic appearance.
2. Add more layers to the grassy hill to create depth and texture.
3. Ensure the SVG remains well-formed and adheres to the constraints.

Here is the finally SVG code:
```xml
<svg viewBox="0 0 256 256" width="256" height="256">
  <!-- Sky with gradient -->
  <defs>
    <linearGradient id="skyGradient" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="rgb(0, 121, 184)" />
      <stop offset="100%" stop-color="rgb(135, 206, 250)" />
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="256" height="128" fill="url(#skyGradient)" />

  <!-- Grassy Hill Base Layer -->
  <path d="M0,128 C50,100 100,150 150,120 C200,90 256,128 256,128 V256 H0 Z" fill="rgb(34, 139, 34)" />

  <!-- Grassy Hill Middle Layer -->
  <path d="M0,140 C50,120 100,160 150,130 C200,100 256,140 256,140 V256 H0 Z" fill="rgb(50, 205, 50)" />

  <!-- Grassy Hill Top Layer -->
  <path d="M0,160 C50,140 100,180 150,150 C200,120 256,160 256,160 V256 H0 Z" fill="rgb(124, 252, 0)" />
</svg>
```
This SVG adheres to the constraints and provides a more detailed and visually appealing representation of the description.
    </example>
        If the original SVG uses unallowed tags or elements, please remove the unallowed tags or elements and use allowed ones instead.
    eg:
    Original: <path style="fill:#C5EAD4;" d="M467.478,512H33.391c-6.147,0-11.13-4.983-11.13-11.13V66.783c0-24.588,19.933-44.522,44.522-44.522
	h367.304c24.588,0,44.522,19.933,44.522,44.522V500.87C478.609,507.017,473.626,512,467.478,512z"/>
    Removed the style attribute (style="fill:#C5EAD4;").
    Added a direct fill attribute (fill="#C5EAD4") within the <path> element.
    Result:<path fill="#C5EAD4" d="M467.478,512H33.391c-6.147,0-11.13-4.983-11.13-11.13V66.783c0-24.588,19.933-44.522,44.522-44.522
    h367.304c24.588,0,44.522,19.933,44.522,44.522V500.87C478.609,507.017,473.626,512,467.478,512z"/>


    Please ensure that the finally SVG code is well-formed, valid, and strictly follows these constraints, with a certain level of detail. Focus on a clear and concise representation of the input description within the given limitations. 
    Always give the complete SVG code with nothing omitted. Never use an ellipsis.
    Do not use any Elements or Attributes outside the permitted range
    If the original SVG has obvious errors in shape and color, please modify it.
    Visualize the svg after each reasoning step.,output you svg finally.
    <Original SVG>{base}</Original SVG>
    <description>"{des}"</description>

    Now is your turn!"""
    return prompt


def get_deepseek_response(prompt):
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        max_tokens=4000
    )
    return response

def get_chatgpt_response(prompt):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        max_tokens=4000
    )
    return response

def process_data(index, description):
    try:
        prompt = get_formated_text(description)
        response = get_chatgpt_response(prompt) if model_name == "openai" else get_deepseek_response(prompt)
        res = response.choices[0].message.content
        usage = response.usage.total_tokens
        print(f"id: {index}, des: {description}, usage: {usage}")
        return index, (description, res)
    except Exception as e:
        return process_data(index, description)

def process_data_a(index, description):
    try:
        prompt = get_prompt_a(description)
        response = get_chatgpt_response(prompt) if model_name == "openai" else get_deepseek_response(prompt)
        res = response.choices[0].message.content
        usage = response.usage.total_tokens
        print(f"id: {index}, des: {description}, usage: {usage}")
        return index, (prompt, res)
    except Exception as e:
        return process_data_a(index, description)

data = []

with open("/media/qba/Data/Project/DeepLearning/Text2SVG/data/tgt_data_deepseek_300", "r") as f:
    for i in f:
        data.append(i.split(". ")[-1][:-1])
# with open("tgt_data_chatgpt.txt", "r") as f:
#     for i in f:
#         odata.append(i.split(". ")[-1].strip())
# rdata = Utils.load_data("data/deepseek_data_150.json")
# data = []

# for i in range(len(rdata)):
#     data.append((odata[i], Utils.extract_last_svg(rdata[str(i)][1])))

data_with_response = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(process_data_a, i, desc): i for i, desc in enumerate(data)}
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        index, result = future.result()
        data_with_response[index] = result

with open(f"data_with_response_a_{model_name}_{int(time.time())}.json", "w") as f:
    json.dump(data_with_response, f, indent=4)
