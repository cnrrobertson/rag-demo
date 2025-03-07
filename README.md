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
