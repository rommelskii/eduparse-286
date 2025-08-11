from rag_pipeline.defs import *
import os
import json
import openvino_genai
import numpy as np

"""
    RAGPipeline:
        1. Perform indexing
        2. Receive queries
        3. Embedding function
        4. Perform cosine similarity 
"""

TRANSCRIPTS_DIR = './outputs/transcripts/'

class RAGPipeline:
    config = openvino_genai.TextEmbeddingPipeline.Config()
    config.pooling_type = openvino_genai.TextEmbeddingPipeline.PoolingType.MEAN
    embedding_pipeline = openvino_genai.TextEmbeddingPipeline(EMBEDDING_MODEL_DIR, "CPU", config)

    def __init__(self):
        self.vector_buf = []
        self.json_buf = None

    def fetch_vector_database(self, timestamp):
        json_files = [f for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith('.json')]
        target_data = None
        
        if timestamp in json_files:
            target_path = os.path.join(TRANSCRIPTS_DIR, timestamp)
            with open(target_path, 'r', encoding='utf-8') as file:
                target_data = json.load(file)
                self.json_buf = target_data

        for segment in target_data['segments']:
            self.vector_buf.append(segment['embedding'])

    def create_embeddings(self, text):
        return self.embedding_pipeline.embed_query(text.strip())

    def print_buffer(self):
        print(self.vector_buf)

    def cosine_similarity(self, query_embedding):
        product_buf = [] 
        if not self.vector_buf:
            print("Error: cannot perform cosine similarity with an empty vector buffer")
            return 
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        for vector in self.vector_buf:
            norm = vector / np.linalg.norm(vector)
            product_buf.append( np.dot(norm, query_norm) )

        return product_buf

    def fetch_most_relevant_sentence(self, document_dir, query_string):
        query_embeddings = self.create_embeddings(query_string)
        self.fetch_vector_database(document_dir)
        cos = [float(x) for x in self.cosine_similarity(query_embeddings)]
        return self.json_buf["segments"][cos.index(max(cos))]["text"]
        
            




