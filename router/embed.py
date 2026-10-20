"""Local sentence embeddings (fastembed / ONNX, no torch), L2-normalized so dot product = cosine."""

from functools import cache

import numpy as np
from fastembed import TextEmbedding

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


@cache
def _model(name: str) -> TextEmbedding:
    return TextEmbedding(name)


def embed(texts: list[str], model: str = DEFAULT_MODEL) -> np.ndarray:
    vectors = np.array(list(_model(model).embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)
