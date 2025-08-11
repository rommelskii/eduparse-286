import openvino_genai
from tqdm import tqdm
from utils.defs import * 
from utils.audio_utils import read_wav, fetch_audio_files
from utils.json_processor import result_to_json
from datetime import datetime

def main():
    pipe = openvino_genai.WhisperPipeline(MODEL_DIR, "CPU")
    for audio in fetch_audio_files():
        raw_speech = read_wav(audio)
        print("Audio transformation finished")
        result = pipe.generate(raw_speech, return_timestamps=True)
        print("Speech inferencing finished. Performing embedding")
        result_to_json(result, './results')
    print("Done")
    

if __name__ == "__main__":
    main()
