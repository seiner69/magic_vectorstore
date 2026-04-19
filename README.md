# magic-vectorstore

Modular vector storage and similarity search library for RAG applications.

## Features

| Strategy | Class | Description |
|----------|-------|-------------|
| ChromaDB | `ChromaVectorStore` | Persistent/in-memory, cosine/euclidean metrics |
| FAISS | `FAISSVectorStore` | flat/ivf/hnsw indices, persist/load support |

## Installation

```bash
pip install chromadb faiss-cpu  # or faiss-gpu
```

## Quick Start

```python
from magic_vectorstore import ChromaVectorStore, VectorEntry

store = ChromaVectorStore(collection_name="my_collection")
store.add([
    VectorEntry(id="1", embedding=[0.1, 0.2, 0.3], text="Hello world")
])
result = store.search([0.1, 0.2, 0.3], top_k=5)
print(f"Found {len(result.entries)} results")
```

## Module Structure

```
magic_vectorstore/
    __init__.py
    run.py
    core/           # VectorEntry, QueryResult, BaseVectorStore
    strategies/
        chroma/     # ChromaDB implementation
        faiss/      # FAISS implementation
    utils/
```

## CLI

```bash
python -m magic_vectorstore.run --input entries.json --strategy chroma --action add
```
