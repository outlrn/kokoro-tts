import torch
import numpy as np
import base64
import json
import re
from flask import Flask, request, Response, stream_with_context, jsonify
from kokoro import KPipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Loading Kokoro on {device}...")
pipeline = KPipeline(lang_code='a', device=device)

@app.route('/generate_stream', methods=['POST'])
def generate_stream():
    data = request.get_json()
    text = data.get('text', '')
    voice_name = data.get('voice', 'af_bella')
    speed = float(data.get('speed', 1.0))

    if not text: return jsonify({"error": "No text"}), 400

    def generate():
        try:
            pack = pipeline.load_voice(voice_name)
        except:
            pack = pipeline.load_voice('af')

        # --- CRITICAL CHANGE HERE ---
        # Old: split_pattern=r'\n+' (Only splits on new lines)
        # New: Splits on [. , ! ? ;] followed by a space.
        # Regex Explanation:
        # (?<=[.,;!?]) : Lookbehind. Checks if the previous char was punctuation.
        # \s+          : Matches one or more spaces.
        aggressive_split = r'(?<=[.;!?])\s+'

        for graphemes, phonemes, audio in pipeline(text, voice=pack, speed=speed, split_pattern=aggressive_split):
            
            # 1. Convert Audio
            audio_np = audio.cpu().numpy()
            # audio_int16 = (audio_np * 32767).astype(np.int16)
            # audio_b64 = base64.b64encode(audio_int16.tobytes()).decode('utf-8')

            max_val = np.abs(audio_np).max()
            if max_val > 0:
                audio_np = audio_np / max_val
            # ------------------------------------------

            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_b64 = base64.b64encode(audio_int16.tobytes()).decode('utf-8')

            # 2. Send Packet
            packet = {
                "audio": audio_b64,
                "text": graphemes 
            }

            yield json.dumps(packet) + "\n"

    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)