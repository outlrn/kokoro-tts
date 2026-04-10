import torch
import numpy as np
import zipfile
import io

input_path = "voices.bin"
output_path = "voices1.npz" # We use .npz for native NumPy compatibility

print(f"📦 Converting {input_path} to {output_path}...")

voices_dict = {}
try:
    # Open the v1.0 bin as a zip
    with zipfile.ZipFile(input_path, 'r') as z:
        for file_name in z.namelist():
            if file_name.endswith('.npy'):
                voice_name = file_name.replace('.npy', '')
                with z.open(file_name) as f:
                    # Load the raw weights
                    voices_dict[voice_name] = np.load(io.BytesIO(f.read()))
                    print(f"✅ Ready: {voice_name}")

    # Save as a standard NumPy archive (NOT a pickle)
    np.savez(output_path, **voices_dict)
    print(f"\n🚀 Success! Created {output_path}")
    print("This file is optimized for CPU loading.")

except Exception as e:
    print(f"❌ Error: {e}")