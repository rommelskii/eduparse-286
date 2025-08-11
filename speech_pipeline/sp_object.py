import openvino_genai
from speech_pipeline.utils.defs import * 
from speech_pipeline.utils.audio_utils import read_wav, fetch_audio_files
from speech_pipeline.utils.json_processor import result_to_json

class SpeechPipeline:
    pipeline_object = openvino_genai.WhisperPipeline('./speech_pipeline/'+MODEL_DIR, "CPU")
    def __init__(self, embedding_pipeline):
        self.embedding_pipeline = embedding_pipeline
    def inference(self, path_to_audio_file=None):
        if path_to_audio_file is None:
            for audio_file in fetch_audio_files():
                raw_speech = read_wav(audio_file)
                print("Audio transformation finished")
                result = self.pipeline_object.generate(raw_speech, return_timestamps=True)
                print("Speech inferencing finished. Performing embedding")
                result_to_json(result, './outputs/transcripts/', self.embedding_pipeline)
        else:
            raw_speech = read_wav(path_to_audio_file)
            print("Audio transformation finished")
            result = self.pipeline_object.generate(raw_speech, return_timestamps=True)
            print("Speech inferencing finished. Performing embedding")
            result_to_json(result, './outputs/transcripts/', self.embedding_pipeline)



