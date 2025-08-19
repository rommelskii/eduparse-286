from speech_pipeline.sp_object import SpeechPipeline
from rag_pipeline.rag_object import RAGPipeline 

rag = RAGPipeline()
sp = SpeechPipeline(rag)

rag.set_transcript_buffer( sp.inference('./inputs/audio/Mitochondria_Cell_Biology.mp3', "mimel") )

