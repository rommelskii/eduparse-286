import openvino_genai
from utils.defs import * 
from utils.audio_utils import read_wav
from datetime import datetime

def create_embeddings(chunk: str):
    device = "CPU"  # GPU can be used as well

    config = openvino_genai.TextEmbeddingPipeline.Config()
    config.pooling_type = openvino_genai.TextEmbeddingPipeline.PoolingType.MEAN

    pipeline = openvino_genai.TextEmbeddingPipeline(EMBEDDING_MODEL_DIR, device, config)
    query_embeddings = pipeline.embed_query(chunk)

    return query_embeddings

def cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    return dot_product / (norm_vec1 * norm_vec2)
