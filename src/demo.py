import marimo

__generated_with = "0.11.16"
app = marimo.App(width="medium")

@app.cell
def _():
    import marimo as mo
    import numpy as np
    import ollama
    import faiss
    return faiss, mo, np, ollama

@app.cell
def _(faiss, np):
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
    return data, data_name, file, index, index_name, line, model

@app.cell
def _(data, index, model, np, ollama):
    def find_reviews(question, n_documents=10):
        embedding = np.array(
            ollama.embed(model, question).embeddings
        )
        scores, reviews = index.search(embedding, n_documents)
        return data[reviews].flatten()

    def chat(prompt, history):
        response = ollama.chat(
            model,
            messages=history + [
                {'role': 'user', 'content': prompt}
            ]
        )
        return response.message.content
    return chat, find_reviews

@app.cell
def _(chat, find_reviews, mo):
    def my_model(messages, config):
        # Retrieve related reviews
        question = messages[-1].content
        reviews = find_reviews(question)

        # Build prompt
        context = "\n".join(reviews.tolist())
        prompt = "You are an expert on home appliances. Answer questions as an expert would, without referencing 'documents,' or 'provided context.' Never use phrases like 'based on the information provided' or 'based on the text snippet.' Simply provide direct, helpful information about the appliances as if you inherently know these details. Always include the name of a product with a summary of its rating and reviews from consumers if possible." + "\n"
        prompt += f"Context: {context}" + "\n\n"
        prompt += f"Question: {question}" + "\n\n"
        prompt += "Answer:" + "\n\n"

        # Submit message
        history = [{"role":m.role, "content":m.content} for m in messages[:-1]]
        response = chat(prompt, history)
        return response

    mo.ui.chat(my_model)
    return (my_model,)

if __name__ == "__main__":
    app.run()
