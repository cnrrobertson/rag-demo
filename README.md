# rag-demo

This repo gives a very basic demo of using a RAG structure to ask questions about product reviews from Amazon.

## Details

**Data**: This code is meant to use the `Appliances` category from the [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io). To allow for the demo to run locally, 20,000 reviews are randomly sampled from this category to build the vector store.
**Vector Store**: The database is built using [FAISS](https://github.com/facebookresearch/faiss) with a simple $L2$ comparison.
**LLM**: For simplicity and to run the demo locally, it uses [Ollama](https://ollama.com) to call the [Llama3.2:3b](https://ollama.com/library/llama3.2) model.

## Usage

The demo can be quickly run by running `make all` in this repo which will:

1. Download and subsample the review data and metadata (if needed)
2. Create the vector store (if needed)
3. Run `src/demo.py` as a [marimo](https://marimo.io) app

Alternatively, steps can be run individually with `make data`, `make vector_store`, and `make demo`.

Note that this assumes [`curl`](https://curl.se) and [`pixi`](https://pixi.sh/latest/) are installed on the system.

## Notes

As this is a demo, there are several shortcomings:
1. In a real review RAG, queries would likely be associated to a specific product, which would narrow the vectors to be considered. This might require building vector stores for each product.
    - As an approximation, I append each review text with the name of the product. Thus, if a query asks about a specific product, its information will likely emerge.
2. The chat is not particularly context aware. A new query in a chat sequence will retrieve new information from the vector store which may not be associated with the previous messages.
    - To work around this, I use an intermediate query to see if previous messages and the current query relate to a specific product. If so, the title of the product is appended to the current query for vector store search.
3. The Llama3.2:3b model is fairly rudimentary (though very impressive to me given its size). Thus, the chat often makes meta references to the provided instructions, context, library of reviews, etc.
    - To mitigate this issue, the core prompt includes instructions to avoid this kind of discussion and to prioritize sharing expert knowledge and summarizing review information about specific products.
