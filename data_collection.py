import logging
from pathlib import Path
import sys

import torch
import whisper
from tqdm import tqdm

# === SETUP ===
SAVE_DIR = Path("transcripts")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Load Whisper (small/base/medium/large)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("medium", device=DEVICE)  # adjust to "small" for faster runs
USE_FP16 = DEVICE == "cuda"


# === LOGGING ===
class TqdmLoggingHandler(logging.Handler):
    """Route logging records through tqdm to keep progress bars intact."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


logger = logging.getLogger("data_collection")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_handler = TqdmLoggingHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.propagate = False


# === TRANSCRIBE ===
def transcribe_audio(audio_path: Path, transcript_path: Path, video_id: str) -> None:
    """Transcribe a single MP3 file using Whisper."""
    logger.info(f"🎧  Starting transcription for {video_id}")
    result = model.transcribe(str(audio_path), fp16=USE_FP16)
    text = result.get("text", "").strip()

    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(text, encoding="utf-8")
    logger.info(f"💾  Transcription saved to {transcript_path}")


# === MAIN ===
def main(reverse: bool = False) -> None:
    """Main transcription loop. If reverse=True, process files in reverse order."""
    logger.info("🧠  Starting transcription phase...")
    mp3_files = sorted(SAVE_DIR.glob("*.mp3"), reverse=reverse)

    for mp3_path in tqdm(mp3_files, desc="Transcribing", unit="file"):
        video_id = mp3_path.stem
        transcript_path = SAVE_DIR / f"{video_id}.txt"

        if transcript_path.exists():
            logger.info(f"⏭️  Skipping {video_id} (already transcribed)")
            continue

        try:
            transcribe_audio(mp3_path, transcript_path, video_id)
        except Exception as exc:
            logger.error(f"❌  Error transcribing {video_id}: {exc}")

    logger.info("🏁  All videos processed successfully.")


# === ENTRY POINT ===
if __name__ == "__main__":
    try:
        reverse = "--reverse" in sys.argv
        main(reverse=reverse)
    except KeyboardInterrupt:
        logger.error("❌  Processing interrupted by user")
