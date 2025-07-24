import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional
import re

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
import yt_dlp
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(
    title="YouTube Video Clipper API",
    description="Download and trim YouTube video clips",
    version="1.0.0"
)

# Configuration - Render-specific paths
MAX_DURATION = int(os.getenv("MAX_CLIP_DURATION", "300"))  # 5 minutes default
TEMP_DIR = Path(os.getenv("TEMP_DIR", "/tmp/video_clips"))
TEMP_DIR.mkdir(exist_ok=True)

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=int(os.getenv("MAX_WORKERS", "2")))


class ClipRequest(BaseModel):
    youtube_url: str
    start_time: str  # Format: "hh:mm:ss" or seconds as string
    end_time: str  # Format: "hh:mm:ss" or seconds as string

    @field_validator('youtube_url')
    @classmethod
    def validate_youtube_url(cls, v):
        youtube_pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
        if not re.match(youtube_pattern, v):
            raise ValueError('Invalid YouTube URL')
        return v

    @field_validator('start_time', 'end_time')
    @classmethod
    def validate_time_format(cls, v):
        # Accept both "hh:mm:ss" and seconds format
        if ':' in v:
            try:
                parts = v.split(':')
                if len(parts) == 3:
                    h, m, s = map(int, parts)
                    return str(h * 3600 + m * 60 + s)
                elif len(parts) == 2:
                    m, s = map(int, parts)
                    return str(m * 60 + s)
            except ValueError:
                raise ValueError('Invalid time format. Use hh:mm:ss or seconds')
        else:
            try:
                int(v)
                return v
            except ValueError:
                raise ValueError('Invalid time format. Use hh:mm:ss or seconds')


def time_to_seconds(time_str: str) -> int:
    """Convert time string to seconds"""
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(int, parts)
            return m * 60 + s
    return int(time_str)


def download_youtube_video(url: str, output_path: str) -> str:
    """Download YouTube video using yt-dlp with advanced anti-detection"""
    ydl_opts = {
        'format': 'best[height<=720]',
        'outtmpl': output_path,
        'no_warnings': True,
        'ignoreerrors': False,
        # Advanced anti-detection headers
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        },
        # Enhanced extraction options
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web', 'ios'],
                'player_skip': ['configs'],
                'skip': ['hls', 'dash'],
            }
        },
        # Additional anti-detection measures
        'sleep_interval': 1,
        'max_sleep_interval': 5,
        'sleep_interval_subtitles': 1,
        'force_ipv4': True,
        'no_cache_dir': True,
        'proxy': None,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            return output_path
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")


def trim_video(input_path: str, output_path: str, start_time: int, end_time: int) -> str:
    """Trim video using ffmpeg"""
    duration = end_time - start_time

    if duration <= 0:
        raise HTTPException(status_code=400, detail="End time must be after start time")

    if duration > MAX_DURATION:
        raise HTTPException(status_code=400, detail=f"Clip duration cannot exceed {MAX_DURATION} seconds")

    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-ss', str(start_time),
        '-t', str(duration),
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-preset', 'fast',
        '-crf', '23',
        '-y',  # Overwrite output file
        output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFmpeg error: {result.stderr}")
        return output_path
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Video processing timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")


async def process_video_clip(request: ClipRequest) -> str:
    """Process video clip in background"""
    clip_id = str(uuid.uuid4())

    # Create temporary file paths
    temp_video = TEMP_DIR / f"{clip_id}_full.%(ext)s"
    temp_clip = TEMP_DIR / f"{clip_id}_clip.mp4"

    try:
        # Convert time formats
        start_seconds = time_to_seconds(request.start_time)
        end_seconds = time_to_seconds(request.end_time)

        # Download video
        loop = asyncio.get_event_loop()
        downloaded_file = await loop.run_in_executor(
            executor, download_youtube_video, request.youtube_url, str(temp_video)
        )

        # Find the actual downloaded file (yt-dlp changes extension)
        actual_file = None
        for ext in ['mp4', 'webm', 'mkv']:
            potential_file = TEMP_DIR / f"{clip_id}_full.{ext}"
            if potential_file.exists():
                actual_file = str(potential_file)
                break

        if not actual_file:
            raise HTTPException(status_code=500, detail="Downloaded file not found")

        # Trim video
        output_file = await loop.run_in_executor(
            executor, trim_video, actual_file, str(temp_clip), start_seconds, end_seconds
        )

        # Clean up full video file
        Path(actual_file).unlink(missing_ok=True)

        return str(temp_clip)

    except Exception as e:
        # Clean up files on error
        for pattern in [f"{clip_id}_*"]:
            for file in TEMP_DIR.glob(pattern):
                file.unlink(missing_ok=True)
        raise e


def cleanup_file(file_path: str):
    """Clean up temporary file"""
    try:
        Path(file_path).unlink(missing_ok=True)
    except:
        pass


@app.get("/")
async def root():
    return {
        "message": "YouTube Video Clipper API - Deployed on Render",
        "docs": "/docs",
        "health": "/health",
        "environment": os.getenv("RENDER_SERVICE_NAME", "development")
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": os.getenv("RENDER_SERVICE_NAME", "local"),
        "region": os.getenv("RENDER_REGION", "unknown")
    }


@app.post("/clip")
async def create_clip(request: ClipRequest, background_tasks: BackgroundTasks):
    """Create a video clip from YouTube URL"""
    try:
        output_file = await process_video_clip(request)

        # Schedule file cleanup after response
        background_tasks.add_task(cleanup_file, output_file)

        return FileResponse(
            path=output_file,
            media_type='video/mp4',
            filename=f"clip_{Path(output_file).stem}.mp4"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/limits")
async def get_limits():
    """Get API limits and constraints"""
    return {
        "max_clip_duration_seconds": MAX_DURATION,
        "supported_time_formats": ["hh:mm:ss", "mm:ss", "seconds"],
        "max_video_quality": "720p",
        "supported_output_format": "mp4",
        "max_workers": executor._max_workers
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
