from rag_pipeline.rag_object import RAGPipeline
from speech_pipeline.sp_object import SpeechPipeline
import os

def read_file_as_string(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: File '{filename}' not found."
    except Exception as e:
        return f"Error reading file: {str(e)}"

def main():
    rag = RAGPipeline()
    sp = SpeechPipeline(rag)

    print("""
          EduPARSE-286 beta front-end

          Abadiano, Malatag, Ronduen, Sangilan
          """)

    audio_dir = "./inputs/audio/"
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith((".wav", ".mp3", ".m4a"))]

    if not audio_files:
        print("No audio files found in ./inputs/audio/")
        return

    print("\nAvailable audio files:")
    for i, file in enumerate(audio_files, start=1):
        print(f"{i}. {file}")

    while True:
        try:
            choice = int(input("\nEnter the number of the audio file to process: "))
            if 1 <= choice <= len(audio_files):
                selected_file = os.path.join(audio_dir, audio_files[choice - 1])
                break
            else:
                print("Invalid selection. Please choose a number from the list.")
        except ValueError:
            print("Please enter a valid number.")

    name = input('Enter name of the session: ')

    filename = sp.inference(selected_file, name)
    rag.set_filename(filename)

    transcript_path = f'./outputs/transcripts/{filename}.txt'
    document_text = read_file_as_string(transcript_path)

    print(document_text)

    rag.document_receive(document_text)

    print(rag.topic)

    print("\nYou can now ask questions regarding the document.")
    while True:
        question = input("\nWhat would you like to ask? ")
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response, ts = rag.perform_prompt(question)
        print("\nResponse:")
        print(response)
        print("\nReferences:")
        print(ts)

if __name__ == "__main__":
    main()

