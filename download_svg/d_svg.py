import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
# 目标网页
url = "https://www.svgrepo.com/collection/ecommerce-flat-icons-2/"  # 替换为你要爬取的网页

# 发送请求
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
response.raise_for_status()
# 解析 HTML
soup = BeautifulSoup(response.text, "html.parser")

# 查找所有 itemprop="contentUrl" 的 img 标签
svg_urls = []
for img_tag in soup.find_all("img", itemprop="contentUrl", src=True):
    src = img_tag["src"]
    if src.endswith(".svg"):  # 确保是 SVG 文件
        svg_urls.append(urljoin(url, src))


# 下载 SVG
for svg_url in svg_urls:
    svg_name = svg_url.split("/")[-1]
    svg_path = os.path.join("svgs", svg_name)

    print(f"Downloading {svg_name}...")
    svg_response = requests.get(svg_url, headers=headers)
    with open(svg_path.replace("-","_"), "wb") as f:
        f.write(svg_response.content)

print("Download completed!")
