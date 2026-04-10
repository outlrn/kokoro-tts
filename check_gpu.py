import onnxruntime as ort
import os

print("--------------------------------------------------")
print("🔥 GPU DEBUGGER: FORCING CRASH IF NO GPU 🔥")
print("--------------------------------------------------")

# 1. Check if ONNX even sees the CUDA provider
available = ort.get_available_providers()
print(f"👀 Available Providers: {available}")

if 'CUDAExecutionProvider' not in available:
    print("❌ CRITICAL: 'CUDAExecutionProvider' is NOT in the list.")
    print("   Fix: pip install onnxruntime-gpu")
    exit()

# 2. Force-Load the model on GPU ONLY.
# We REMOVE 'CPUExecutionProvider'. If GPU fails, this script MUST crash.
print("⏳ Attempting to load model on NVIDIA GPU (No CPU Fallback)...")

try:
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 0  # 0 = Verbose (Print EVERYTHING)
    
    # POINT THIS TO YOUR EXACT MODEL FILE
    model_path = "kokoro-v1.0.onnx" 
    
    # 🚨 THE TRAP: We only allow CUDA. If it fails, it crashes.
    sess = ort.InferenceSession(
        model_path, 
        sess_options=sess_options, 
        providers=['CUDAExecutionProvider']
    )
    print("\n✅ SUCCESS! The model accepted the GPU.")
    print("   If you see this, the issue is your Python code, not the drivers.")

except Exception as e:
    print("\n❌ CRASHED! The GPU rejected the model.")
    print("--------------------------------------------------")
    print(f"ERROR DETAILS:\n{e}")
    print("--------------------------------------------------")
    print("💡 COMMON FIXES:")
    print("1. 'LoadLibrary failed with error 126': You are missing 'zlibwapi.dll'.")
    print("2. 'cudnn64_8.dll not found': You need to install cuDNN.")
    print("3. 'Protobuf error': Your model file is corrupt/empty.")