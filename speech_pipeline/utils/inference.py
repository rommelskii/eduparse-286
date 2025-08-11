import openvino_genai
import librosa
from utils.defs import *
from utils.audio_utils import *
from utils.file import *

def main():
    pipe = openvino_genai.WhisperPipeline(MODEL_DIR, "CPU")
    audio_path = input()
    raw_speech = read_wav(audio_path)
    result = pipe.generate(raw_speech, return_timestamps=True)
    save_transcription_to_file(result.chunks, "./test.txt")

if __name__ == "__main__":
    main()

