import numpy as np
import ollama
import faiss

# Settings
index_name = "index/Appliances_20000.index"
data_name = "index/Appliances_20000.txt"
model = "llama3.2:3b"

# Load vector store / data
index = faiss.read_index(index_name)
data = []
with open(data_name, "r") as file:
    for line in file:
        data.append(line.strip())
data = [d.replace('|:|', '\n') for d in data]
data = np.array(data)

# %%
# Test retrieval
msg1 = "What are peoples favorite coffee related items?"
msg1_embed = np.array(
    ollama.embed(model, msg1).embeddings
)
msg1_scores, msg1_reviews = index.search(msg1_embed, 10)
msg1_context = data[msg1_reviews]

msg1_prompt = f"""
You are an assistant who is helping users sift through a large corpus of user reviews. Use the following context to answer the question. Keep the answer concise but always try to include the name of a product with a summary of its rating and what reviews like or dislike about it.
Context: {msg1_context}
Question: {msg1}
Answer:
"""

msg1_answer = ollama.generate(model, msg1_prompt)
print(msg1_answer.response)
