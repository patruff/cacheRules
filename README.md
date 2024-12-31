# Cache-Augmented Generation (CAG) Implementation

This is an implementation of the Cache-Augmented Generation approach based on the paper "Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks".

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the dickens.txt file in the parent directory.

3. Run the implementation:
```bash
python cag.py
```

## How it works

The system:
1. Preloads the document content
2. Computes and caches the KV (Key-Value) representations
3. Uses the cache for generating responses to queries without retrieval

## Features

- Document preloading and caching
- Cache persistence (save/load functionality)
- Query response generation using cached context
- No retrieval required during inference

## Note

This implementation uses the Llama 2 model by default, but you can modify the model_name in the CAGSystem initialization to use other models.
