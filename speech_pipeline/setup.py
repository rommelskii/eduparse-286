import subprocess
import sys
from pathlib import Path

# --- Configuration for the models to be downloaded and optimized ---
MODELS_TO_OPTIMIZE = [
    {
        "name": "Whisper (Speech-to-Text)",
        "model_id": "openai/whisper-base",
        "task": None,  # Task can be auto-detected for Whisper
        "output_dir": "whisper-base-int8-ov"
    },
    {
        "name": "Embedding Model (Sentence Transformer)",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "task": "feature-extraction",
        "output_dir": "all-MiniLM-L6-v2-int8-ov"
    },
    {
        "name": "Reranking Model (Cross-Encoder)",
        "model_id": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "task": "text-classification",
        "output_dir": "ms-marco-MiniLM-L6-v2-int8-ov"
    }
]

def check_prerequisites():
    """Checks if optimum-cli is available in the environment."""
    try:
        # Use 'subprocess.run' with 'check=True' to see if the command exists.
        # 'capture_output=True' hides the command's help message from the console.
        subprocess.run(["optimum-cli", "--help"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: 'optimum-cli' not found.")
        print("Please install the necessary libraries by running:")
        print("pip install \"optimum[openvino]\"")
        return False

def optimize_model(model_info):
    """
    Constructs and runs the optimum-cli command to download and optimize a model.
    """
    model_id = model_info["model_id"]
    output_dir = Path(model_info["output_dir"])
    task = model_info["task"]
    
    print("-" * 60)
    print(f"Processing Model: {model_info['name']} ({model_id})")

    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"✅ Directory '{output_dir}' already exists and is not empty. Skipping.")
        return

    # Construct the command
    command = [
        "optimum-cli", "export", "openvino",
        "--model", model_id,
        "--weight-format", "int8",
        str(output_dir)
    ]
    
    # Add the --task argument only if it's specified
    if task:
        command.extend(["--task", task])

    print(f"Running command: {' '.join(command)}")
    
    try:
        # Execute the command
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ Successfully optimized and saved to '{output_dir}'")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error optimizing model '{model_id}'.")
        print(f"Return Code: {e.returncode}")
        print(f"Stderr: {e.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    print("--- Starting RAG Model Download and Optimization ---")
    
    if not check_prerequisites():
        sys.exit(1)
        
    for model in MODELS_TO_OPTIMIZE:
        optimize_model(model)
        
    print("-" * 60)
    print("\n🎉 All models have been processed.")

