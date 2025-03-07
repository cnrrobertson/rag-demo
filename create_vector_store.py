import matplotlib.pyplot as plt
import numpy as np
import json
import ollama
import faiss
import time

# Settings
metadata_path = "data/meta_Appliances.jsonl"
data_path = "data/Appliances_20000.jsonl"
model = "llama3.2:3b"

# Load metadata
metadata = {}
with open(metadata_path, "r") as file:
    for line in file:
        info = json.loads(line)
        metadata[info["parent_asin"]] = info["title"]

# Load data
data = []
with open(data_path, "r") as file:
    for line in file:
        info = json.loads(line)
        item_title = metadata[info["parent_asin"]]
        data_str = f"""Item:{item_title}
Rating:{info['rating']}
Title:{info['title']}
Text:{info['text']}"""
        data.append(data_str)

# %%
# Create vector store
time1 = time.time()
embeddings = np.array(
    ollama.embed(model, data).embeddings
)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
time2 = time.time()
print(time2-time1)

# %%
# Save vector store / data
faiss.write_index(index, "index/Appliances_20000.index")
save_data = [d.replace("\n", "|:|") for d in data]
with open("index/Appliances_20000.txt", "w") as file:
    for sd in save_data:
        file.write(sd + "\n")
