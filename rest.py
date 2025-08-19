from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from backend.session import Database
from rag_pipeline.rag_object import RAGPipeline
from speech_pipeline.sp_object import SpeechPipeline 
from ocr_pipeline.ocr_object import OCRPipeline
from flask_cors import CORS
import os
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './inputs/'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'mp4', 'mp3', 'docx'}
DB_URL = 'mongodb://localhost:27017/'
COLLECTION_NAME = 'eduparse'

CORS(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config.setdefault('SECONDARY_IMAGE_FORMAT', 'PNG')  # or 'JPEG'

from werkzeug.utils import secure_filename
import os
import subprocess
from flask import request, jsonify

DEFAULT_UPLOAD_DIR = "./inputs"
DEFAULT_AUDIO_SAMPLE_RATE = 16000  # keep consistent with your pipeline
DEFAULT_AUDIO_CHANNELS = 1         # mono is typical for speech

def _abs_upload_dir(app):
    upload_dir = app.config.get('UPLOAD_FOLDER', DEFAULT_UPLOAD_DIR)
    return os.path.abspath(upload_dir)


def _convert_to_mp3(input_path, output_path, sample_rate, channels):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-acodec", "libmp3lame",
        "-q:a", "2",
        output_path,
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed: {res.stderr.decode(errors='ignore') or 'Unknown error'}"
        )


@app.route('/receive_file', methods=['POST'])
def process_lecture():
    # Validate primary file
    if 'primary' not in request.files:
        return jsonify({'error': 'No primary file provided'}), 400

    primary = request.files['primary']
    secondary = request.files.get('secondary')

    if primary.filename == '' or not allowed_file(primary.filename):
        return jsonify({'error': 'Invalid primary file'}), 400

    # Normalize lecture name once
    lecture_name_raw = request.form.get('lectureName') or 'untitled'
    safe_base = secure_filename(lecture_name_raw).strip('_') or 'untitled'
    print(f"Lecture Name Received: {lecture_name_raw} -> normalized: {safe_base}")

    # Prepare upload dir
    upload_dir = _abs_upload_dir(app)
    os.makedirs(upload_dir, exist_ok=True)

    # Audio config
    sample_rate = int(app.config.get('AUDIO_SAMPLE_RATE', DEFAULT_AUDIO_SAMPLE_RATE))
    channels = int(app.config.get('AUDIO_CHANNELS', DEFAULT_AUDIO_CHANNELS))

    warnings = []
    primary_path = None
    tmp_input_path = None

    # Extension check
    _, primary_ext = os.path.splitext(secure_filename(primary.filename))
    primary_ext = primary_ext.lower()
    print("Detected primary ext:", primary_ext)

    # Handle upload & convert
    if primary_ext == '.mp3':
        primary_path = os.path.join(upload_dir, f"{safe_base}.mp3")
        primary.save(primary_path)
        print("Saved MP3 to:", primary_path)

    else:
        tmp_input_path = os.path.join(upload_dir, f"{safe_base}{primary_ext}")
        primary.save(tmp_input_path)
        print("Saved temp input to:", tmp_input_path)

        primary_path = os.path.join(upload_dir, f"{safe_base}.mp3")
        _convert_to_mp3(tmp_input_path, primary_path, sample_rate, channels)

        try:
            os.remove(tmp_input_path)
        except Exception:
            warnings.append(f"Could not remove temp file: {tmp_input_path}")

    # Verify MP3
    if not os.path.exists(primary_path):
        raise RuntimeError(f"Expected MP3 not found at {primary_path}")
    if os.path.getsize(primary_path) == 0:
        raise RuntimeError("Output MP3 is empty; conversion failed.")

    print("Final MP3:", primary_path, "size:", os.path.getsize(primary_path))

    # Handle secondary
    secondary_path = None
    if secondary and secondary.filename:
        sec_filename = secure_filename(secondary.filename)
        _, sec_ext = os.path.splitext(sec_filename)
        sec_ext = sec_ext.lower()

        if sec_ext == '.pdf':
            pdf_path = os.path.join(upload_dir, f"{safe_base}_secondary.pdf")
            secondary.save(pdf_path)

            out_fmt = app.config.get('SECONDARY_IMAGE_FORMAT', 'PNG').upper()
            out_ext = '.png' if out_fmt == 'PNG' else '.jpg'
            secondary_path = os.path.join(upload_dir, f"{safe_base}_secondary{out_ext}")

            try:
                images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=200)
                if not images:
                    raise RuntimeError('No pages rendered from PDF')

                img = images[0]
                if out_fmt == 'JPEG' and img.mode in ('RGBA', 'P', 'LA'):
                    img = img.convert('RGB')

                save_kwargs = {'quality': 92} if out_fmt == 'JPEG' else {}
                img.save(secondary_path, out_fmt, **save_kwargs)

                try:
                    os.remove(pdf_path)
                except Exception:
                    warnings.append(f"Could not remove PDF: {pdf_path}")
            except Exception as e:
                secondary_path = pdf_path
                warnings.append(f'PDF-to-image conversion failed: {e}. Kept original PDF.')
        else:
            secondary_path = os.path.join(upload_dir, f"{safe_base}_secondary{sec_ext}")
            secondary.save(secondary_path)

    # Respond — note: primaryFilePath is exact file to use in inference
    return jsonify({
        'message': 'Files saved successfully',
        'normalizedLectureName': safe_base,
        'primaryFilePath': os.path.abspath(primary_path),
        'secondaryFilePath': os.path.abspath(secondary_path) if secondary_path else None,
        'warnings': warnings or None
    }), 200


db = Database(DB_URL, COLLECTION_NAME)
rag = RAGPipeline()
sp = SpeechPipeline(rag)
ocr = OCRPipeline()

@app.route('/addUser', methods=['POST'])
def add_user():
    name = request.json.get("name")
    user = db.fetchUser(name)
    if user:
        return jsonify({"error": "User exists"}), 500 
    db.addUser(name)
    return jsonify(name), 200

@app.route('/deleteUser', methods=['POST'])
def delete_user():
    name = request.json.get("name")
    user = db.fetchUser(name)
    if user is None:
        return jsonify({"error": "User does not exist"}), 500 
    db.deleteUser(name)
    return jsonify(name), 200

@app.route('/fetchSessions', methods=['GET'])
def fetch_sessions():
    name = request.json.get("name")
    user = db.fetchUser(name)
    if user is None:
        return jsonify({"error": "User does not exist"}), 500 
    sessions = db.fetchSessionByName(name)
    return jsonify(sessions), 200

@app.route('/findSession', methods=['POST'])
def find_session():
    name = request.json.get("name")
    session_name = request.json.get("session_name")
    user = db.fetchUser(name)
    if user is None:
        return jsonify({"error": "User does not exist / invalid directory"}), 500 

    return jsonify(db.fetchSession(name, session_name)), 200


@app.route('/addSession', methods=['POST'])
def add_session():
    name = request.form.get("user_name")
    session_name = request.form.get("session_name")
    print("Name: " + name)

    base_path = './inputs/'
    primary_filename = f"{session_name}.mp3"
    safe_base = secure_filename(primary_filename).strip('_') or 'untitled'
    primary_path = os.path.join(base_path, safe_base)

    # Check for a single secondary file
    secondary_path = None
    for ext in ['.jpg', '.jpeg', '.png']:
        candidate = os.path.join(base_path, f"{session_name}_secondary{ext}")
        if os.path.exists(candidate):
            secondary_path = candidate
            break

    print(f"Secondary file found: {secondary_path}" if secondary_path else "No secondary file found.")

    # Run inference (pass secondary_path if needed)
    if secondary_path:
        text = ocr.inference(secondary_path)
        refactored = rag.refactor(text)
        session, timestamped_text = sp.inference(primary_path, primary_filename, refactored)

    session, timestamped_text = sp.inference(primary_path, primary_filename, None)
    # Save session to DB
    db.addSession(name, session)

    return jsonify({
        "transcript": timestamped_text,
        "outline": session['outline'],
        "ocr_text": session['ocr'],
    }), 200


@app.route('/removeSession', methods=['POST'])
def remove_session():
    name = request.form.get("user_name")
    session_name = request.form.get("session_name")
    user = db.fetchUser(name)

    filename_proper = secure_filename(session_name).strip('_') or 'untitled'

    print(session_name)
    if user is None:
        return jsonify({"error": "User does not exist"}), 500 
    db.removeSession(name, f'{session_name}.mp3')
    return jsonify(name), 200


@app.route('/performRAG', methods=['POST'])
def perform_rag():
    name = request.json.get("user_name")
    session_name = f"{request.json.get("session_name")}.mp3"
    prompt = request.json.get("prompt")
    print(f"""
Name: {name}
Session name: {session_name}
prompt: {prompt}
          """)
    session = db.fetchSession(name, session_name)
    topic = session['topic']
    ocr = session['ocr']
    rag.set_transcript_buffer(session)
    result = rag.perform_prompt(prompt, topic, ocr)

    return jsonify(result), 200



if __name__ == '__main__':
    app.run(debug=True)
