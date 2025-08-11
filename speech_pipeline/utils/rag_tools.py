import openvino_genai
from speech_pipeline.utils.defs import * 
from speech_pipeline.utils.audio_utils import read_wav
from datetime import datetime

def create_embeddings(chunk: str):
    device = "CPU"  # GPU can be used as well

    config = openvino_genai.TextEmbeddingPipeline.Config()
    config.pooling_type = openvino_genai.TextEmbeddingPipeline.PoolingType.MEAN
    embedding_pipeline = openvino_genai.TextEmbeddingPipeline(EMBEDDING_MODEL_DIR, device, config)

    query_embeddings = embedding_pipeline.embed_query(chunk)

    return query_embeddings
