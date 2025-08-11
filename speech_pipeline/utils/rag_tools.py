import openvino_genai
from speech_pipeline.utils.defs import * 
from speech_pipeline.utils.audio_utils import read_wav
from datetime import datetime

def create_embeddings(embedding_pipeline, chunk: str):
    device = "CPU"  # GPU can be used as well

    query_embeddings = embedding_pipeline.embed_query(chunk)

    return query_embeddings
