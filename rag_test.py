from rag_pipeline.rag_object import RAGPipeline

r = RAGPipeline("2025-08-12T02-10-08.json")

print(r.perform_prompt("What is the mitochondria of a cell"))

