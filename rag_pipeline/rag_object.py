from rag_pipeline.defs import *
import openvino_genai

"""
    RAGPipeline:
        1. Perform indexing
        2. Receive queries
        3. Embedding function
"""

class RAGPipeline:
    config = openvino_genai.TextEmbeddingPipeline.Config()
    config.pooling_type = openvino_genai.TextEmbeddingPipeline.PoolingType.MEAN
    embedding_pipeline = openvino_genai.TextEmbeddingPipeline(EMBEDDING_MODEL_DIR, "CPU", config)

    def __init__(self):
        pass

    def create_embeddings(self, text):
        return self.embedding_pipeline.embed_query(text.strip())

