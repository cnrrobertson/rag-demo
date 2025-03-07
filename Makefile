.PHONY: appliances, vector_store, demo

# Download and set up dataset
data: data/Appliances.jsonl data/meta_Appliances.jsonl data/Appliances_20000.jsonl

data/Appliances.jsonl:
	mkdir -p data
	curl -L https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Appliances.jsonl.gz | gunzip > $@

data/meta_Appliances.jsonl:
	mkdir -p data
	curl -L https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Appliances.jsonl.gz | gunzip > $@

data/Appliances_20000.jsonl:
	mkdir -p data
	gshuf -n 20000 data/Appliances.jsonl -o data/Appliances_20000.jsonl

# Process data and make vector store
vector_store: index/Appliances.index

index/Appliances.index:
	pixi run python src/create_vector_store.py

# Run demo RAG chat
demo:
	pixi run marimo run src/demo.py

# Run all
all: data vector_store demo
