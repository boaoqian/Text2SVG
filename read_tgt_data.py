data = []

with open("data/tgt_data_chatgpt.txt", "r") as f:
    for i in f:
        data.append(i.split(". ")[-1][:-1])
