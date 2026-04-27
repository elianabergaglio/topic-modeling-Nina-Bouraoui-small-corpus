import json
import os
import re
from pathlib import Path
from typing import Tuple

import numpy as np


def extract_metadata_from_filename(filename: str) -> Tuple[str, int]:
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')

    if len(parts) >= 4:
        year = None
        for i, part in enumerate(parts):
            if re.match(r'^\d{4}$', part):
                year = int(part)
                work_name = '_'.join(parts[1:i]) if i >= 2 else (parts[1] if len(parts) > 1 else "Unknown")
                break

        if year is None:
            for part in reversed(parts):
                if part.isdigit():
                    year = int(part)
                    break
            work_name = '_'.join(parts[1:-1]) if len(parts) > 2 else "Unknown"
    else:
        work_name = name_without_ext
        year = None

    if year is None:
        raise Exception(f"Year is None for work: {filename}")

    return work_name or "Unknown", year or 0


def create_or_load_embeddings(docs, embedding_model, embeddings_file):
    if os.path.exists(embeddings_file):
        print(f"Loading embeddings from disk: {embeddings_file}")
        embeddings = np.load(embeddings_file)
    else:
        print("Computing embeddings...")
        embeddings = embedding_model.encode(docs, show_progress_bar=True)
        np.save(embeddings_file, embeddings)
        print(f"Embeddings saved to {embeddings_file}")
    return embeddings


def save_parameters(output_dir: Path, **params):
    serializable_params = {
        key: str(value) if hasattr(value, '__fspath__') else value
        for key, value in params.items()
    }
    with open(output_dir / "params.json", "w") as f:
        json.dump(serializable_params, f, indent=4)
