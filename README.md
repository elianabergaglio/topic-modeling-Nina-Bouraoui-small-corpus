# Topic Modeling of Nina Bouraoui's Novels

Computational literary analysis of three French novels by Nina Bouraoui using unsupervised topic modeling. The pipeline extracts semantic themes from sentence-level text chunks using multilingual transformer embeddings, UMAP dimensionality reduction, HDBSCAN clustering, and BERTopic.

**Novels analyzed:**
- *La Voyeuse interdite* (1991)
- *Garçon manqué* (2000)
- *Mes mauvaises pensées* (2005)

## Methodology

```
Raw text → Sentence chunking → Multilingual embeddings
       → UMAP (5D) → HDBSCAN clustering → BERTopic topic extraction
```

1. **Chunking** (`text_chunker.py`): texts are split on sentence-ending punctuation (`.?;!»`) into chunks of 80–500 characters.
2. **Embedding**: each chunk is encoded with [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), a sentence transformer trained for multilingual semantic similarity.
3. **Dimensionality reduction**: UMAP projects embeddings to 5 dimensions (cosine metric, 5 neighbors).
4. **Clustering**: HDBSCAN identifies dense semantic clusters (`min_cluster_size=7`).
5. **Topic representation**: BERTopic + KeyBERT extract the most discriminative n-grams per cluster using c-TF-IDF.

All hyperparameters are in `config.yaml`.

## Installation

**Requirements:** Python 3.12, pip

```bash
pip install -r requirements.txt
```

> GPU is optional but strongly recommended for the embedding step.

## Reproduce Results

**Option 1 — CLI (recommended):**
```bash
# Process all novels
python run_pipeline.py --config config.yaml

# Process a single novel
python run_pipeline.py --config config.yaml --text "La_Voyeuse_interdite"
```

Outputs are written to `topic_modeling_output/<experiment_name>/`:
- `chunks.csv` — filtered text chunks
- `chunks_with_topic.csv` — chunks with assigned topic IDs and keywords
- `embeddings_*.npy` — cached sentence embeddings (reused on re-runs)
- `saved_model/` — serialized BERTopic model (safetensors format)
- `params.json` — hyperparameters used for the run

## Explore Results

Open `analysis.ipynb` after fitting. It loads the saved model and displays topic distributions and representative sentences.

## License

Code: [MIT License](LICENSE). See `LICENSE` for details.

The text data in `data/` is the property of Nina Bouraoui and her publishers. It is included here with permission solely for academic research purposes.
