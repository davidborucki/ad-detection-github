import logging
import re
from pathlib import Path
from typing import Iterable, List

import torch
import whisper
from tqdm import tqdm
from yt_dlp import YoutubeDL

# === SETUP ===
SAVE_DIR = Path("transcripts")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Load Whisper (small/base/medium/large)
model = whisper.load_model("small")  # adjust to "medium" if you prefer
USE_FP16 = torch.cuda.is_available()


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


# === UTILITIES ===
def ensure_single_mp3_extension(path: Path) -> Path:
    """Rename the given file so it ends with exactly one .mp3 extension."""
    if not path.exists():
        return path

    cleaned_name = re.sub(r"(\.mp3)+$", ".mp3", path.name, flags=re.IGNORECASE)
    cleaned_path = path.with_name(cleaned_name)
    if cleaned_path != path:
        if cleaned_path.exists():
            cleaned_path.unlink()
        path.rename(cleaned_path)
    return cleaned_path


# === DOWNLOAD ===
def download_audio(video_id: str) -> Path:
    url = f"https://www.youtube.com/watch?v={video_id}"
    logger.info(f"▶️  Starting download for {video_id}")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(SAVE_DIR / f"{video_id}.%(ext)s"),
        "quiet": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    expected_path = SAVE_DIR / f"{video_id}.mp3"
    fallback_matches = list(SAVE_DIR.glob(f"{video_id}.mp3*"))
    if expected_path.exists():
        audio_path = ensure_single_mp3_extension(expected_path)
    elif fallback_matches:
        audio_path = ensure_single_mp3_extension(fallback_matches[0])
    else:
        raise FileNotFoundError(f"Unable to locate downloaded audio for {video_id}")

    logger.info(f"✅  Download complete for {video_id}")
    return audio_path


# === TRANSCRIBE ===
def transcribe_audio(audio_path: Path, transcript_path: Path, video_id: str) -> None:
    logger.info(f"🎧  Starting transcription for {video_id}")
    result = model.transcribe(str(audio_path), fp16=USE_FP16)
    text = result.get("text", "").strip()

    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(text, encoding="utf-8")
    logger.info(f"💾  Transcription saved to {transcript_path}")

    try:
        audio_path.unlink(missing_ok=True)
    except OSError as err:
        logger.warning(f"⚠️  Unable to delete audio file {audio_path}: {err}")


# === HELPERS ===
def load_video_ids(source: Path) -> List[str]:
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")
    with source.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def iter_video_ids(source: Path) -> Iterable[str]:
    for video_id in load_video_ids(source):
        yield video_id


# === MAIN ===
def main() -> None:
    video_ids = list(iter_video_ids(Path("scrape.txt")))

    # STEP 1: Download all MP3s first
    logger.info("📥  Starting download phase...")
    for video_id in tqdm(video_ids, desc="Downloading", unit="video"):
        mp3_path = SAVE_DIR / f"{video_id}.mp3"
        if mp3_path.exists():
            logger.info(f"⏭️  Skipping {video_id} (already downloaded)")
            continue
        try:
            download_audio(video_id)
        except Exception as exc:
            logger.error(f"❌  Error downloading {video_id}: {exc}")

    # STEP 2: Transcribe all MP3s afterward
    logger.info("🧠  Starting transcription phase...")
    mp3_files = list(SAVE_DIR.glob("*.mp3"))
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.error("❌  Processing interrupted by user")

