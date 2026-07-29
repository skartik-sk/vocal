import sys
import torch
import sounddevice as sd
from transformers import AutoProcessor, BarkModel

def main():
    print("[AI Worker] Booting up and loading Bark Model into RAM...")
    
    # Use a standard, fully supported model
    model_id = "suno/bark-small"
    
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = BarkModel.from_pretrained(model_id)
        
        # Move to Apple Silicon GPU (MPS) if available
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        print(f"[AI Worker] Model loaded on {device.upper()}! Waiting for sentences from Rust...")
        
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
            # Process and generate
            inputs = processor(text=[sentence], return_tensors="pt").to(device)
            
            with torch.no_grad():
                audio_array = model.generate(**inputs)
            
            # Extract audio and play
            audio_data = audio_array.cpu().numpy().squeeze()
            sample_rate = model.generation_config.sample_rate
            
            sd.play(audio_data, samplerate=sample_rate)
            sd.wait()
            
        except Exception as e:
            print(f"[AI Worker] Error generating audio: {e}")
        
    print("\n[AI Worker] Rust closed the pipe. Shutting down and clearing RAM!")

if __name__ == "__main__":
    main()