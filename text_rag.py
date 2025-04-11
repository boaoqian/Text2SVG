from transformers import BertTokenizer, BertModel
import torch,os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

img_path = "/media/qba/Data/Project/DeepLearning/Text2SVG/img/"
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




# 示例句子列表
sentences = [i.split(".")[0].replace("_", " ") for i in imgs]
# 找到最相似的句子
q = "a lighthouse overlooking the ocean"
sentence1, sentence2, idx = find_most_similar_sentence(q ,sentences)
print(f"The most similar sentences are:\n'{sentence1}'\n'{sentence2}'")
with open(os.path.join(img_path,imgs[idx]), "r") as f:
    print(f.read())

