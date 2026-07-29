import sys
import torch
import sounddevice as sd
from transformers import AutoProcessor, AutoModel

def main():
    print("[AI Worker] Booting up and loading Qwen3 Model into RAM...")
    
    # The exact model you requested
    model_id = "AtomGradient/Qwen3-TTS-0.6B-CustomVoice-4bit-pruned-vocab-lite"
    
    try:
        # Load the Processor and Model
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        
        # Move to Apple Silicon GPU (MPS) if available, otherwise CPU
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        print(f"[AI Worker] Qwen3 Model loaded on {device.upper()}! Waiting for sentences from Rust...")
        
    except Exception as e:
        print(f"[AI Worker] Fatal Error loading model: {e}")
        return

    # Read from Rust's pipe
    for line in sys.stdin:
        sentence = line.strip()
        if not sentence:
            continue
            
        print(f"\n[AI Worker] Generating audio for: '{sentence}'")
        
        try:
            # Process text and generate audio
            inputs = processor(text=sentence, return_tensors="pt").to(device)
            
            with torch.no_grad():
                audio_array = model.generate(**inputs)
            
            # Extract raw audio data and sample rate
            audio_data = audio_array.cpu().numpy().squeeze()
            sample_rate = getattr(model.config, 'sample_rate', 24000) # Default Qwen3 sample rate
            
            # Play seamlessly
            sd.play(audio_data, samplerate=sample_rate)
            sd.wait()
            
        except Exception as e:
            print(f"[AI Worker] Error generating audio for sentence: {e}")
        
    print("\n[AI Worker] Rust closed the pipe. Shutting down and clearing RAM!")

if __name__ == "__main__":
    main()