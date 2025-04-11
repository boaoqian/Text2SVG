import json,re
import Score
data_path = "/home/qba/Data/Project/DeepLearning/Text2SVG/data_with_response_a_deepseek_1741425999.json"
data = {}
with open(data_path, "r") as f:
    data = json.load(f)

def extract_last_svg(content):
    # 正则表达式匹配 <svg> 标签及其内容
    pattern = r'<svg[\s\S]*?</svg>'
    matches = re.findall(pattern, content)

    # 返回最后一个匹配项
    return matches[-1] if matches else None
print(len(data))
result = []
err = []
for i in range(len(data)):
    i=str(i)
    print("-"*100)
    print("id:",i)
    try:
        # score = Score.get_score([data[i][0]],[extract_last_svg(data[i][1])])
        # result.append(score.item())
        with open("img/"+str(i).replace(" ","_")+".svg","w") as f:
            f.write(extract_last_svg(data[i][1]))
            print("-"*100)
            # print("des:",data[i][0]," score:",score)
    except:

        err.append(int(i))
print(err)
# print("avg_score:",sum(result)/len(result))