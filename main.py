from speech_pipeline.sp_object import SpeechPipeline
from rag_pipeline.rag_object import RAGPipeline 

def main():
    rag = RAGPipeline()
    sp = SpeechPipeline(rag)

    sp.inference()
    

if __name__ == "__main__":
    main()
