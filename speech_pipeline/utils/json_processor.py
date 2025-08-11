from tqdm import tqdm
import json
import openvino_genai
import os
from datetime import datetime
from typing import Any
from speech_pipeline.utils.rag_tools import create_embeddings

def result_to_json(result: Any, out_dir: str = ".", embedding_pipeline: Any = None, time_format: str = "%Y-%m-%dT%H-%M-%S") -> str:
    now = datetime.now()
    ts_str = now.strftime(time_format)
    text_lines = []
    
    payload = {
        "generated_at": ts_str,
        "segments": [
            {
                "start_s": float(chunk.start_ts),
                "end_s":   float(chunk.end_ts),
                "text":    chunk.text.strip(),
                "embedding": embedding_pipeline.create_embeddings(chunk.text.strip())
            }
            for chunk in tqdm(result.chunks)
        ],
    }

    for chunk in result.chunks:
        text = chunk.text.strip()
        line = f"[{chunk.start_ts:.2f} - {chunk.end_ts:.2f}] {text}"
        text_lines.append(line)
    
    filename = f"{ts_str}.json"
    filepath = os.path.join(out_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(filepath+'.txt', "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))
    
    return filepath

