from rag_pipeline.defs import *
from qwen_agent.llm import get_chat_model
import os, json, openvino_genai, numpy as np
import openvino.properties as props
import re
import openvino.properties.hint as hints
import openvino.properties.streams as streams

TRANSCRIPTS_DIR = './outputs/transcripts/'
LLM_DIR = './rag_pipeline/models/qwen'

class RAGPipeline:
    config = openvino_genai.TextEmbeddingPipeline.Config()
    config.pooling_type = openvino_genai.TextEmbeddingPipeline.PoolingType.MEAN
    embedding_pipeline = openvino_genai.TextEmbeddingPipeline(EMBEDDING_MODEL_DIR, "CPU", config)

    def __init__(self, json_buffer=None):
        self.vector_buf = []
        self.json_buffer = json_buffer 

        self.ov_config = {
            hints.performance_mode(): hints.PerformanceMode.LATENCY,
            streams.num(): "1",
            props.cache_dir(): ""
        }

        self.topic = None
        self.llm_cfg = {
            "ov_model_dir": LLM_DIR,
            "model_type": "openvino",
            "device": "CPU",
            "ov_config": self.ov_config,
            "generate_cfg": {"top_p": 0.95, "fncall_prompt_type": "qwen"},
        }
        self.llm = get_chat_model(self.llm_cfg)

    def set_transcript_buffer(self, json_input):
        self.json_buffer = json_input

    def get_outline(self, text):
        prompt = f"""
    For the context below, generate a simple bulleted topic outline.

    --CONTEXT--
    {text}

        """
        messages = [{"role": "user", "content": prompt}]
        responses = self.llm.chat(messages=messages, stream=False)
        return responses[0]['content']

    def refactor(self, text):
        prompt = f"""
            With the given text below, try to correct the sentence if there are any errors. Otherwise
            keep the sentence as is:

            ---TEXT---
            {text}
        """
        messages = [{"role": "user", "content": prompt}]
        responses = self.llm.chat(messages=messages, stream=False)
        return responses[0]['content']

    def perform_prompt(self, question, topic, supplemental=None):
        """
        Function that takes in a question from the user and the topic denoted in the inbound JSON payload
        """
        self.fetch_vector_database()
        context_sentences, context_sentences_with_timestamps = self.fetch_most_relevant_sentences(question)

        context_sentences_with_timestamps = [re.sub(r"```", "", s) for s in context_sentences_with_timestamps]
        context_block = "\n".join(context_sentences)
        timestamped_block = "\n".join(context_sentences_with_timestamps)

        prompt = f"""
    You are an assistant with regards to a specific lecture video discussion transcript context and learning materials associated with the topic. You are given the main topic of the document alongside the transcript context for your supporting details for your answer.

    --- TOPIC ---
    {topic}

    --- DISCUSSION TRANSCRIPT CONTEXT ---
    {context_block}

    --- SUPPLEMENTAL INFORMATION --- 
    {supplemental or ""}

    --- QUESTION ---
    {question}
    """

        messages = [{"role": "user", "content": prompt}]
        responses = self.llm.chat(messages=messages, stream=False)

        result = responses[0]['content']

        final_result = f"""{result}\n\nReferences:\n{timestamped_block}"""

        return final_result, timestamped_block


    def get_topic(self, document_text):
        prompt = f"""
What is the document about? Text: {document_text}
        """
        messages = [{"role": "user", "content": prompt}]
        responses = self.llm.chat(messages=messages, stream=False)
        return responses[0]['content']

    def fetch_vector_database(self, timestamp=None):
        self.vector_buf = []
        for segment in self.json_buffer['segments']:
            self.vector_buf.append(segment['embedding'])

    def create_embeddings(self, text):
        return self.embedding_pipeline.embed_query(text.strip())

    def cosine_similarity(self, query_embedding, cutoff=0):
        if not self.vector_buf:
            print("Error: cannot perform cosine similarity with an empty vector buffer")
            return []

        query_norm = query_embedding / np.linalg.norm(query_embedding)
        return [
            (i, np.dot(vector / np.linalg.norm(vector), query_norm))
            for i, vector in enumerate(self.vector_buf)
            if np.dot(vector / np.linalg.norm(vector), query_norm) >= cutoff
        ]

    def fetch_most_relevant_sentences(self, query_string, top_n=5):
        def format_timestamp(seconds):
            minutes = int(seconds) // 60
            secs = int(seconds) % 60
            return f"{minutes}m {secs}s"

        query_embeddings = self.create_embeddings(query_string)
        self.fetch_vector_database()
        scored_segments = self.cosine_similarity(query_embeddings)

        sorted_segments = sorted(scored_segments, key=lambda x: x[1], reverse=True)
        top_segments = sorted_segments[:top_n]
        top_indices = [index for index, score in top_segments]

        top_sentences = [
            self.json_buffer["segments"][index]["text"]
            for index in top_indices
        ]

        top_sentences_with_timestamps = [
            f"[{format_timestamp(self.json_buffer['segments'][index]['start_s'])} - {format_timestamp(self.json_buffer['segments'][index]['end_s'])}] {self.json_buffer['segments'][index]['text']}"
            for index in top_indices
        ]

        return (top_sentences, top_sentences_with_timestamps)


