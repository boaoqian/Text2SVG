import json,re

def load_data(data_path):
    data_path = "data/deepseek_data_150.json"
    data = {}
    with open(data_path, "r") as f:
        data = json.load(f)
    return data

def extract_last_svg(content):
    # 正则表达式匹配 <svg> 标签及其内容
    pattern = r'<svg[\s\S]*?</svg>'
    matches = re.findall(pattern, content)

    # 返回最后一个匹配项
    return matches[-1] if matches else None