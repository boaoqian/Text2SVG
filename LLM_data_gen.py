from openai import OpenAI
from tqdm import tqdm
import time,json

model_name = "deepseek"
# model_name = "openai"

deepseek_client = OpenAI(api_key="sk-", base_url="https://api.deepseek.com")
if model_name == "openai":
    import os
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    
openai_client = OpenAI(api_key="")

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

    Please ensure that the generated SVG code is well-formed, valid, and strictly follows these constraints, with a certain level of detail. Focus on a clear and concise representation of the input description within the given limitations. 
    Always give the complete SVG code with nothing omitted. Never use an ellipsis.
    Visualize the svg after each reasoning step.,output you svg finally.
    <description>"{description}"</description>

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
    max_tokens=2000
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
    max_tokens=2000
    )
    return response

data = []
with open("data/tgt_data_chatgpt.txt", "r") as f:
    for i in f:
        data.append(i.split(". ")[-1][:-1])

data = [data[0],data[50],data[-1]]
data_with_response = {}
for i in tqdm(range(len(data))):
    try:
        prompt = get_formated_text(data[i])
        if model_name == "openai":
            response = get_chatgpt_response(prompt)
        else:
            response = get_deepseek_response(prompt)
        res = response.choices[0].message.content
        data_with_response[i] = [data[i],res]
        usage = response.usage.total_tokens
        print("id:",i," des:", data[i], "usage:", usage)
        time.sleep(0.5)
    except Exception as e:
        data_with_response[i] = [data[i],str(e)]
    

with open("data_with_response_"+model_name+str(time.time()),"w") as f:
    json.dump(data_with_response, f)
