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
    def find_reviews(question, messages, n_documents=10):
        # Get item context
        history = [{"role":m.role, "content":m.content} for m in messages[:-1]]
        item_prompt = f"Given this chat history and the question: {question}, are we discussing a specific item? Answer with the name of the product or with a blank space.\nITEM NAME: "
        relevant_item = chat(item_prompt, history)

        # Look for relevant documents
        new_question = f"Question: {question}\nItem: {relevant_item}"
        embedding = np.array(
            ollama.embed(model, new_question).embeddings
        )
        scores, reviews = index.search(embedding, n_documents)

        # Filter scores by relevant product
        item_embedding = np.array(
            ollama.embed(model, relevant_item).embeddings
        )
        item_scores, item_reviews = index.search(item_embedding, n_documents)

        good_reviews = np.intersect1d(reviews, item_reviews)
        if len(good_reviews) > 0:
            return data[good_reviews].flatten()
        else:
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
        reviews = find_reviews(question, messages)

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
