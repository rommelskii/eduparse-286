from tqdm import tqdm
from typing import Any

def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"

def result_to_json(result: Any, out_dir: str = ".", embedding_pipeline: Any = None, name: str = None, ocr_text: str = None):
    segments = []
    all_text_lines = []
    timestamped_text_lines = []

    print("\n📝 Transcript with Timestamps:\n")

    for chunk in tqdm(result.chunks):
        text = chunk.text.strip()
        start = float(chunk.start_ts)
        end = float(chunk.end_ts)

        # Format timestamps as "Xm Ys"
        start_str = format_timestamp(start)
        end_str = format_timestamp(end)

        # Print transcript line
        print(f"[{start_str} - {end_str}] {text}")

        # Build segment for JSON
        segments.append({
            "start_s": start,
            "end_s": end,
            "text": text,
            "embedding": embedding_pipeline.create_embeddings(text)
        })

        # Collect plain and timestamped lines
        all_text_lines.append(text)
        timestamped_text_lines.append(f"[{start_str} - {end_str}] {text}")

    full_text = "\n".join(all_text_lines)
    timestamped_full_text = "\n".join(timestamped_text_lines)

    payload = {
        "session_name": name,
        "segments": segments,
        "topic": embedding_pipeline.get_topic(full_text),
        "outline": embedding_pipeline.get_outline(full_text),
        "ocr": ocr_text  # Placeholder for future OCR integration
    }

    return payload, timestamped_full_text

