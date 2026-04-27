"""
Run the BERTopic pipeline on French literary texts.

Usage:
    python run_pipeline.py --config config.yaml
    python run_pipeline.py --config config.yaml --text "Bouraoui_La_Voyeuse_interdite"
"""
import argparse
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from utils import create_or_load_embeddings, save_parameters
from text_chunker import TextChunker


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_")


def _find_text_file(data_dir: Path, text_filter: str) -> Path:
    txt_files = list(data_dir.glob("*.txt"))
    matches = [f for f in txt_files if text_filter in f.name]
    if not matches:
        raise ValueError(
            f"No .txt file matching '{text_filter}' in {data_dir}. "
            f"Available: {[f.name for f in txt_files]}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple files match '{text_filter}': {[f.name for f in matches]}. Be more specific."
        )
    return matches[0]


def run(cfg: dict, text_filter: str | None = None):
    data_dir = Path(cfg["data_dir"])
    output_base = Path(cfg["output_dir"])
    stopwords_file = Path(cfg["stopwords_file"])

    c = cfg["chunking"]
    chunker = TextChunker(
        min_chunk_size=c["min_chunk_size"],
        max_chunk_size=c.get("max_chunk_size", 500),
        method=c.get("method", "default"),
        min_tokens=c.get("min_tokens", 3),
    )

    if text_filter:
        file_path = _find_text_file(data_dir, text_filter)
        print(f"Processing single file: {file_path.name}")
        chunks = chunker.process_file(str(file_path))
        data = pd.DataFrame(chunks)[["chunk_id", "chunk_text", "num_tokens", "num_characters", "year", "work_name"]]
    else:
        data = chunker.process_directory(str(data_dir))

    work_names = data["work_name"].unique().tolist()

    with open(stopwords_file, "r") as f:
        fr_stop_words = [line.strip() for line in f.readlines()]

    emb_cfg = cfg["embedding"]
    embedding_model = SentenceTransformer(emb_cfg["model_name"])

    for work_name in work_names:
        print(f"\n{'='*60}")
        print(f"Processing: {work_name}")
        print(f"{'='*60}")

        work_data = data[data["work_name"] == work_name].copy()
        min_chars = c["min_chunk_size"]
        max_chars = c.get("max_chunk_size", 500)
        work_data = work_data[
            (work_data["num_characters"] >= min_chars)
            & (work_data["num_characters"] <= max_chars)
        ]
        print(f"Chunks after size filtering: {len(work_data)}")

        experiment_name = f"nina_{sanitize_name(work_name)}"
        output_dir = output_base / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for title, df, fname in [
            ("Chunk length distribution (before filtering)", data[data["work_name"] == work_name], "chunks_length_before_filtering.png"),
            ("Chunk length distribution (after filtering)", work_data, "chunks_length_after_filtering.png"),
        ]:
            fig, ax = plt.subplots()
            df["num_characters"].hist(bins=200, ax=ax)
            ax.set_xlabel("Number of characters")
            ax.set_ylabel("Count")
            ax.set_title(title)
            fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

        work_data.to_csv(output_dir / "chunks.csv", index=False)

        docs = [doc.lower() for doc in work_data["chunk_text"]]
        with open(output_dir / "docs.pkl", "wb") as f:
            pickle.dump(docs, f)

        embeddings_file = output_dir / f"embeddings_{emb_cfg['model_name'].replace('/', '_')}.npy"
        embeddings = create_or_load_embeddings(docs, embedding_model, embeddings_file)

        u = cfg["umap"]
        umap_model = UMAP(
            n_neighbors=u["n_neighbors"],
            n_components=u["n_components"],
            min_dist=u["min_dist"],
            metric=u.get("metric", "cosine"),
            random_state=u.get("random_state", 42),
        )

        h = cfg["hdbscan"]
        hdbscan_model = HDBSCAN(
            min_cluster_size=h["min_cluster_size"],
            metric=h.get("metric", "euclidean"),
            cluster_selection_method=h.get("cluster_selection_method", "eom"),
            prediction_data=True,
        )

        v = cfg["vectorizer"]
        ngram = tuple(v["ngram_range"])
        vectorizer_model = CountVectorizer(
            stop_words=fr_stop_words,
            min_df=v["min_df"],
            ngram_range=ngram,
        )

        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=False)
        representation_model = {"KeyBERT": KeyBERTInspired()}

        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            verbose=True,
        )

        topic_model.fit_transform(docs, embeddings=embeddings)
        print(f"Topics found: {len(topic_model.get_topics())}")

        topic_model.save(
            output_dir / "saved_model",
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=embedding_model,
        )

        doc_info_df = topic_model.get_document_info(docs, df=work_data)
        drop_cols = ["Representative_document", "Representative_Docs", "Representation", "Name", "Probability"]
        doc_info_df.drop(columns=[c for c in drop_cols if c in doc_info_df.columns]).to_csv(
            output_dir / "chunks_with_topic.csv", index=False
        )

        save_parameters(
            output_dir,
            experiment_name=experiment_name,
            work_name=work_name,
            **{k: v for k, v in {
                "min_chunk_size": c["min_chunk_size"],
                "embedding_model_name": emb_cfg["model_name"],
                "n_neighbors": u["n_neighbors"],
                "n_components": u["n_components"],
                "min_dist": u["min_dist"],
                "min_cluster_size": h["min_cluster_size"],
                "min_df": v["min_df"],
                "ngram_range": list(ngram),
                "stopwords_file": str(stopwords_file),
            }.items()},
        )

        print(f"Output written to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run BERTopic on French literary texts.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--text",
        default=None,
        help="Filter to a specific work by partial work_name match (e.g. 'La_Voyeuse')",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg, text_filter=args.text)


if __name__ == "__main__":
    main()
