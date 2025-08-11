from pathlib import Path
from speech_pipeline.utils.defs import *
import openvino_genai
import os
import librosa

def fetch_audio_files():
    target_extensions = {".mp3", ".m4a"}
    search_path = Path('./audio')
    audio_file_list = []
    print(f"Searching for audio files in: {search_path.resolve()}...")
    if not search_path.is_dir():
        print(f"Error: Directory not found at '{search_path.resolve()}'")
        return []
    for item in search_path.iterdir():
        if item.is_file() and item.suffix.lower() in target_extensions:
            audio_file_list.append('./audio/'+item.name)
    return audio_file_list

def read_wav(filepath):
    raw_speech, samplerate = librosa.load(filepath, sr=SAMPLE_RATE)
    return raw_speech.tolist()

def debug_info():
    print("--- Starting Transcription ---")
    print(f"Loading model from: {defs.MODEL_DIR}")
    print(f"Using device: {defs.MODEL_DEVICE}")
