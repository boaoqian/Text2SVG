# !pip install trl
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,6,7"
# import wandb
# wandb.login(key="f67ffb62b9935cc07064a59076d21e7aa0f575b7")
import Utils
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, BitsAndBytesConfig
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
import torch
def get_formated_text(description,example):
    prompt = f"""Generate SVG code to visually represent the following text description, while respecting the given constraints.
    <constraints>
    * **Allowed Elements:** `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
    * **Allowed Attributes:** `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
    </constraints>
    </info>
    Please ensure that the generated SVG code is well-formed, valid, and strictly follows these constraints, with a certain level of detail. Focus on a clear and concise representation of the input description within the given limitations. 
    Always give the complete SVG code with nothing omitted. Never use an ellipsis.
    Visualize the svg after each reasoning step.,output you svg finally.
    Now is your turn!
    <description>"{description}"</description>
"""
    return prompt

def formated_example(data):
    prompt = f"""User:<description>"{data[0]}"</description>
    Assistant:"{data[1]}"
    """
    return prompt

class SVGDataset:
    def __init__(self, data):
        self.data = data
        self.data_key = list(data.keys())
        self.data_size = len(self.data)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_entry = self.data[self.data_key[idx]]
        prompt = get_formated_text(data_entry[0])
        return {"prompt":prompt,"completion":data_entry[1]}

def to_HFDataset(dataset):
    for i in range(len(dataset)):
        yield dataset[i]

def split_dict_by_ratio(data, ratios):
    """
    按比例随机分配字典中的元素到不同组。
    
    :param data: 需要划分的字典 {key: value}
    :param ratios: 比例列表，如 [0.6, 0.3, 0.1]
    :return: 划分后的字典列表
    """
    keys = list(data.keys())
    random.shuffle(keys)  # 打乱键的顺序，确保随机性
    
    total = sum(ratios)
    ratios = [r / total for r in ratios]  # 归一化，确保总和为1
    counts = [int(len(keys) * r) for r in ratios]  # 计算每部分的元素数量
    
    # 调整最后一个部分，确保所有元素都分配
    counts[-1] = len(keys) - sum(counts[:-1])
    
    result = []
    start = 0
    for count in counts:
        selected_keys = keys[start:start+count]
        result.append({k: data[k] for k in selected_keys})
        start += count
    
    return result

if __name__ == "__main__":
    path = "deepseek_data_150.json"
    model_path = "/home/baqian/Data/Project/Model/DeepSeek-R1-Distill-Qwen-7B"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    lora_config = LoraConfig(
        r=32,
        lora_alpha=2,
        lora_dropout=0.05,
        bias="none",  # Bias type for LoRA. the corresponding biases will be updated during training.
        target_modules="all-linear",  # Which modules to apply LoRA to
        task_type="CAUSAL_LM"  # Task type for model architecture
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", quantization_config=quantization_config, attn_implementation="flash_attention_2")
    data = Utils.load_data(path)
    dataset = Dataset.from_generator(to_HFDataset,gen_kwargs={"dataset":SVGDataset(data)}).train_test_split(0.1)
    sftconfig = SFTConfig(
        output_dir="./svg_fineturn",
        max_steps=500,  # Adjust based on dataset size and desired training duration
        per_device_train_batch_size=2,  # Set according to your GPU memory capacity
        per_device_eval_batch_size=1,  # Set according to your GPU memory capacity
        eval_accumulation_steps=20,
        gradient_accumulation_steps=2,
        optim="adamw_torch_fused",  # Use fused AdamW for efficiency
        learning_rate=2e-4,  # Learning rate (QLoRA paper)
        max_grad_norm=0.3,  # Gradient clipping threshold
        # Learning rate schedule
        warmup_ratio=0.03,  # Portion of steps for warmup
        lr_scheduler_type="constant",  # Keep learning rate constant after warmup
        logging_steps=10,  # Frequency of logging training metrics
        save_steps=50,  # Frequency of saving model checkpoints
        eval_strategy="steps",  # Evaluate the model at regular intervals
        eval_steps=50,
        save_total_limit=3,
        report_to=["tensorboard"],
        metric_for_best_model = "loss",
        torch_empty_cache_steps = 500,
        max_seq_length = 4000,
        save_only_model = True)
    trainer = SFTTrainer(
        model,
        args=sftconfig,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,  # LoRA configuration,
        #formatting_func=template,
    )
    trainer.train()
