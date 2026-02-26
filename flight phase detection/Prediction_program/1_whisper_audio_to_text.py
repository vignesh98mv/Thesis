import os
import traceback
import sys
sys.stdout.reconfigure(encoding="utf-8")

# Disable TensorFlow completely
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_TENSORFLOW'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*TensorFlow.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*tensorflow.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Force transformers to use PyTorch
os.environ['USE_TORCH'] = '1'

# Import numpy first to avoid conflicts
import numpy as np

print(f"Using NumPy version: {np.__version__}")

import whisper
from pydub import AudioSegment
import os
from pathlib import Path

# =========================
# Paths (location-independent)
# =========================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent / "Outputs" / "0-whipser-output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log_path = BASE_DIR / "Whisper_error_log.txt"

# =========================
# Whisper model (lazy-loaded)
# =========================
_MODEL = None

def get_model(model_name="medium", device="cuda"):
    global _MODEL
    if _MODEL is None:
        print("🔄 Loading Whisper model...")
        _MODEL = whisper.load_model(model_name, device=device)
    return _MODEL


# =========================
# Utility functions
# =========================
def format_time(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}" if hours else f"{minutes:02d}:{sec:02d}"


def transcribe_chunk(model, filename, offset, output_file):
    filename = str(filename)
    result = model.transcribe(filename, task="translate", language="English")

    with open(output_file, "a", encoding="utf-8") as f:
        for seg in result["segments"]:
            start = seg["start"] + offset
            end = seg["end"] + offset
            text = seg["text"]
            line = f"[{format_time(start)} → {format_time(end)}] {text}\n"
            print(line, end="")
            f.write(line)


# =========================
# Public API (used by main.py)
# =========================
def split_and_transcribe(
    file_path: str,
    chunk_length_ms: int = 600_000,
    model_name: str = "medium",
    device: str = "cuda",
):
    """
    Transcribe an audio file and return output file path.
    Can be called from another script or used standalone.
    """
    model = get_model(model_name, device)

    base_name = Path(file_path).stem
    output_file = OUTPUT_DIR / f"whisper-op-{base_name}.txt"

    audio = AudioSegment.from_file(file_path)
    total_length = len(audio)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Transcription of: {os.path.basename(file_path)}\n\n")

    for i, start_ms in enumerate(range(0, total_length, chunk_length_ms)):
        end_ms = min(start_ms + chunk_length_ms, total_length)
        chunk = audio[start_ms:end_ms]

        chunk_filename = OUTPUT_DIR / f"chunk_{i}.wav"
        chunk.export(str(chunk_filename), format="wav")
        
        header = (
            f"\n--- Processing chunk {i+1} "
            f"({format_time(start_ms/1000)} → {format_time(end_ms/1000)}) ---\n"
        )
        print(header)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(header + "\n")

        transcribe_chunk(
            model,
            str(chunk_filename),
            offset=start_ms / 1000.0,
            output_file=output_file,
        )

        chunk_filename.unlink()

    return output_file


# =========================
# Standalone execution
# =========================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Called from main.py (or CLI)
        audio_file = sys.argv[1]
    else:
        # Standalone fallback
        audio_file = r"C:\CRV\Audios\Youtube data\Full flight\Airbus_LDN_CGN_ss.wav"

    
try:
    output = split_and_transcribe(
        audio_file,
        chunk_length_ms=900_000
    )
    print(output)

except Exception as e:
    print("\n===== FULL ERROR =====")
    traceback.print_exc()

    # also save to file (important when UI hides console)
    with open(log_path, "w", encoding="utf-8") as f:
        traceback.print_exc(file=f)

    print("Error log saved at: ", log_path)
    raise
