import os
import torch
from flask import Flask, render_template, request, jsonify
from TTS.api import TTS
from pydub import AudioSegment

# Initialize Flask app
app = Flask(__name__)

# Set paths for uploads and static files
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Set maximum text length (no longer limiting the length, or change it as needed)
MAX_TEXT_LENGTH = 1000  # 可以调整为一个更大的值

# Load TTS model
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# List of languages for selection
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

    # Check if the text is empty
    if not text:
        return jsonify({"error": "请输入要转换的文本!"})

    # Check if the text exceeds the maximum length
    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({"error": f"文本长度超过最大限制({MAX_TEXT_LENGTH}个字符)!"})

    # Check if the audio file is uploaded
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
        # Split the text into smaller chunks if needed
        text_chunks = [text[i:i + MAX_TEXT_LENGTH] for i in range(0, len(text), MAX_TEXT_LENGTH)]

        # Generate speech for each chunk and save to individual files
        audio_urls = []
        for i, chunk in enumerate(text_chunks):
            wav_output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'output_{i}.wav')
            tts.tts_to_file(text=chunk, speaker_wav=audio_file_path, language=lang_code, file_path=wav_output_path)
            audio_urls.append(f'/static/output_{i}.wav')

        # Return the list of audio file URLs for the front-end to play
        return jsonify({"success": True, "file_urls": audio_urls})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Create necessary folders
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
