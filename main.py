import base64
import json
import numpy as np
from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS
from kokoro_onnx import Kokoro

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION & PRE-LOADING ---
# Load the ONNX model once globally
# Download from: https://github.com/thewh1teagle/kokoro-onnx/releases
kokoro = Kokoro("kokoro-v1.0.onnx", "voices.bin")

# Pre-load common voices into a cache to save milliseconds per request
VOICE_CACHE = {}
def get_voice(name):
    if name not in VOICE_CACHE:
        try:
            VOICE_CACHE[name] = kokoro.get_voice_style(name)
        except:
            VOICE_CACHE[name] = kokoro.get_voice_style("af_bella")
    return VOICE_CACHE[name]

@app.route('/generate_stream', methods=['POST'])
def generate_stream():
    data = request.get_json()
    text = data.get('text', '')
    voice_name = data.get('voice', 'af_bella')
    speed = float(data.get('speed', 1.0))

    if not text:
        return jsonify({"error": "No text"}), 400

    def generate():
        voice_style = get_voice(voice_name)
        
        # kokoro-onnx supports a create_stream generator for low-latency streaming
        stream = kokoro.create_stream(
            text, 
            voice_style, 
            speed=speed, 
            phonemes=False # Set to True if you want to stream phonemes too
        )

        for audio, graphemes in stream:
            # 1. Normalize and Convert Audio
            # ONNX usually returns float32 arrays
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_b64 = base64.b64encode(audio_int16.tobytes()).decode('utf-8')

            # 2. Send Packet
            packet = {
                "audio": audio_b64,
                "text": graphemes 
            }
            yield json.dumps(packet) + "\n"

    return Response(
        stream_with_context(generate()), 
        mimetype='application/x-ndjson'
    )

if __name__ == '__main__':
    # Use threaded=True or a production WSGI like Gunicorn
    app.run(host='0.0.0.0', port=5000, threaded=True)