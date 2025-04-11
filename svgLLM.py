from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_path="/home/baqian/qba/model/DeepSeek-R1-Distill-Qwen-7B"

# Quantization Configuration
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto",quantization_config=quantization_config)

description = "	a starlit night over snow-covered peaks"
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
Please ensure that the generated SVG code is well-formed, valid, and strictly adheres to these constraints. Focus on a clear and concise representation of the input description within the given limitations. 
Always give the complete SVG code with nothing omitted. Never use an ellipsis.
Visualize the svg after each reasoning step.,output you svg finally.
<description>"{description}"</description>

Now is your turn!"""

msg = [{"role":"user","content":prompt}]
msg = tokenizer.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
input_msg = tokenizer(msg,return_tensors="pt").to(model.device)
import time 
t = time.time()
out = model.generate(**input_msg,max_length=8000,     
    do_sample=True,  # 启用采样
    top_p=0.9,  # nucleus sampling
    top_k=3,  # top-k采样
    temperature=1.0  # 温度控制
)
print(out)
print(tokenizer.decode(out[0]))
print(time.time()-t)