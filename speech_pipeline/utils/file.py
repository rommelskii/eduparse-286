from speech_pipeline.utils.defs import *

def save_transcription_to_file(chunks, output_filename):
    try:
        with open(output_filename, 'a', encoding='utf-8') as f:
            print(f"Appending transcription to '{output_filename}'...")
            for chunk in chunks:
                timestamp_text = f"[{chunk.start_ts:.2f}s -> {chunk.end_ts:.2f}s] {chunk.text}\n"
                
                f.write(timestamp_text)
        
        print(f"Transcription successfully appended.")
    except Exception as e:
        print(f"❌ An error occurred while writing to the file: {e}")
