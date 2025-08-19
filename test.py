from rag_pipeline.rag_object import RAGPipeline
from speech_pipeline.sp_object import SpeechPipeline
from backend.session import Database

def main():
    rag = RAGPipeline()
    sp = SpeechPipeline(rag)
    current_document = sp.inference("./inputs/audio/mlk.old.mp3", "Martin Luther")   
    rag.set_transcript_buffer(current_document)
    rag.fetch_vector_database()
    rag.get_topic_from_current_document()
    result, _ = rag.perform_prompt("What is this all about?")


if __name__ == "__main__":
    main()
