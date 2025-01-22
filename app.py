import os
import torch
from flask import Flask, render_template, request, send_file, jsonify
from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play

# Initialize Flask app
app = Flask(__name__)

# Set paths for uploads and static files
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load TTS model
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# List of languages for selection (ensure the keys are language names and the values are language codes)
languages = {
    "中文": "zh",
    "英文": "en",
    "德语": "de",
    "法语": "fr",
    "韩语": "ko"
}

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html', languages=languages)

# Route for handling speech generation
@app.route('/generate', methods=['POST'])
def generate_speech():
    text = request.form['text']
    language = request.form['language']
    audio_file = request.files['audio_file']

    if not text:
        return jsonify({"error": "请输入要转换的文本!"})

    if not audio_file:
        return jsonify({"error": "请上传声音文件!"})

    # Save the uploaded audio file
    audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
    audio_file.save(audio_file_path)

    # Check if the selected language is valid
    lang_code = languages.get(language)
    if not lang_code:
        return jsonify({"error": "选择了无效的语言!"})

    try:
        # Run TTS: Generate speech from text using the selected user's voice
        wav_output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.wav')
        tts.tts_to_file(text=text, speaker_wav=audio_file_path, language=lang_code, file_path=wav_output_path)

        # Convert the WAV to MP3 if necessary
        mp3_output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.mp3')
        sound = AudioSegment.from_wav(wav_output_path)
        sound.export(mp3_output_path, format="mp3")

        # Return the MP3 file URL for the front-end to play
        return jsonify({"success": True, "file_url": '/static/output.mp3'})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Create necessary folders
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Run the Flask app, specifying IP and port
    app.run(host='0.0.0.0', port=5000)
