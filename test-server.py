import requests
import pyaudio
import time

# --- CONFIG ---
URL = "http://localhost:5000/generate_stream"
TEXT = "This is a latency test. We are measuring how fast the first audio chunk arrives, and how long the total generation takes."
VOICE = "af_bella"

# --- SETUP AUDIO ---
p = pyaudio.PyAudio()
# Ensure rate matches server (Kokoro is usually 24000)
stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

print(f"⚡ Sending request to {URL}...")
print(f"📝 Text length: {len(TEXT)} characters")
print("-" * 40)

# --- START TIMING ---
start_time = time.time()
first_byte_time = None
chunk_count = 0

try:
    with requests.post(URL, json={"text": TEXT, "voice": VOICE, "speed": 1.0}, stream=True) as r:
        if r.status_code != 200:
            print(f"❌ Error: {r.status_code} - {r.text}")
            exit()

        # Iterate over incoming chunks
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                # 1. CAPTURE FIRST BYTE TIME
                if first_byte_time is None:
                    first_byte_time = time.time()
                    latency_ms = (first_byte_time - start_time) * 1000
                    print(f"🚀 FIRST AUDIO RECEIVED: {latency_ms:.2f} ms")
                
                # Play audio
                stream.write(chunk)
                chunk_count += 1

except KeyboardInterrupt:
    print("\n⚠️ Stopped by user.")

# --- FINAL ANALYSIS ---
end_time = time.time()
total_duration = end_time - start_time
latency_ms = (first_byte_time - start_time) * 1000 if first_byte_time else 0

print("-" * 40)
print(f"📊 ANALYSIS REPORT")
print(f"-----------------")
print(f"⏱️  Time to First Audio:  {latency_ms:.2f} ms")
print(f"⌛  Total Playback Time:  {total_duration:.2f} s")
print(f"📦  Total Chunks:         {chunk_count}")

stream.stop_stream()
stream.close()
p.terminate()