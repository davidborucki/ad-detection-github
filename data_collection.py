import os, subprocess
from tqdm import tqdm
import whisper
from yt_dlp import YoutubeDL

# Load Whisper (small/base/medium/large)
model = whisper.load_model("small")  # speed vs accuracy tradeoff

SAVE_DIR = "transcripts"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_audio(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    out_path = f"{SAVE_DIR}/{video_id}.mp3"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_path,
        "quiet": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return out_path

def transcribe_audio(path_or_url, video_id):
    # Run Whisper
    result = model.transcribe(path_or_url, fp16=True)
    text = result["text"].strip()
    with open(f"{SAVE_DIR}/{video_id}.txt", "w", encoding="utf-8") as f:
        f.write(text)

def process_video(video_id):
    mp3_path = download_audio(video_id)
    transcribe_audio(mp3_path, video_id)
    os.remove(mp3_path)  # optional cleanup

if __name__ == "__main__":
    with open("scrape.txt") as f:
        ids = [line.strip() for line in f if line.strip()]
    for vid in tqdm(ids, desc="Processing videos"):
        try:
            process_video(vid)
        except Exception as e:
            print(f"[ERROR] {vid}: {e}")
