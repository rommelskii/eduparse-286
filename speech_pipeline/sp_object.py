import openvino_genai
from speech_pipeline.utils.defs import * 
from speech_pipeline.utils.audio_utils import read_wav, fetch_audio_files
from speech_pipeline.utils.json_processor import result_to_json

class SpeechPipeline:
    pipeline_object = openvino_genai.WhisperPipeline('./speech_pipeline/models/whisper-base-int8-ov', "CPU")
    def __init__(self, embedding_pipeline):
        self.embedding_pipeline = embedding_pipeline
    def inference(self, path_to_audio_file, name, ocr_text):
        raw_speech = read_wav(path_to_audio_file)
        print("Performing inference...")
        result = self.pipeline_object.generate(raw_speech, return_timestamps=True)
        print("Proceeding to embeddings transformation...")
        return result_to_json(result, './outputs/transcripts/', self.embedding_pipeline, name, ocr_text)



