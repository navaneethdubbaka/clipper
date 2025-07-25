import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional
import re
import time
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
import yt_dlp

app = FastAPI(
    title="YouTube Video Clipper API",
    description="Download and trim YouTube video clips",
    version="1.0.0"
)

# Configuration - Environment variables with defaults
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


def is_video_accessible(url: str) -> bool:
    """Quick check if video is accessible"""
    try:
        with yt_dlp.YoutubeDL({
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 10,
            'extractor_args': {'youtube': {'player_client': ['android_testsuite']}}
        }) as ydl:
            info = ydl.extract_info(url, download=False)
            return 'title' in info and len(info.get('formats', [])) > 0
    except:
        return False


def download_youtube_video(url: str, output_path: str) -> str:
    """Download YouTube video with latest 2025 anti-detection methods"""

    # Get environment variables for configuration
    sleep_interval = int(os.getenv("YT_DLP_SLEEP_INTERVAL", "3"))
    max_retries = int(os.getenv("YT_DLP_MAX_RETRIES", "5"))
    timeout = int(os.getenv("YT_DLP_TIMEOUT", "90"))
    user_agent = os.getenv("YT_DLP_USER_AGENT",
                           "com.google.android.youtube.tv/2.12.08 (Linux; U; Android 9; AFTT Build/PS7329.3153N) gzip")

    extraction_configs = [
        # Config 1: Android TV client (most reliable in 2025)
        {
            'extractor_args': {
                'youtube': {
                    'player_client': ['android_testsuite', 'android_vr'],
                    'player_skip': ['webpage', 'configs'],
                }
            },
            'http_headers': {
                'User-Agent': user_agent,
                'X-YouTube-Client-Name': '56',
                'X-YouTube-Client-Version': '2.12.08',
            }
        },
        # Config 2: YouTube Music client with latest headers
        {
            'extractor_args': {
                'youtube': {
                    'player_client': ['android_music'],
                }
            },
            'http_headers': {
                'User-Agent': 'com.google.android.apps.youtube.music/7.02.52 (Linux; U; Android 13; Pixel 7) gzip',
                'X-YouTube-Client-Name': '21',
                'X-YouTube-Client-Version': '7.02.52',
            }
        },
        # Config 3: Embedded client bypass
        {
            'extractor_args': {
                'youtube': {
                    'player_client': ['tv_embedded'],
                    'player_skip': ['configs'],
                }
            },
            'age_limit': 99,
        },
        # Config 4: Ultra-low quality fallback
        {
            'format': 'worst[height<=360]/worst',
            'extractor_args': {
                'youtube': {
                    'player_client': ['web'],
                }
            }
        }
    ]

    base_opts = {
        'format': 'best[height<=720]/best',
        'outtmpl': output_path,
        'no_warnings': True,
        'no_cache_dir': True,
        'force_ipv4': True,
        'retries': max_retries,
        'fragment_retries': 1,
        'socket_timeout': timeout,
        'sleep_interval': sleep_interval,
        'max_sleep_interval': 8,
    }

    for i, config in enumerate(extraction_configs):
        try:
            opts = {**base_opts, **config}

            with yt_dlp.YoutubeDL(opts) as ydl:
                # Pre-validate extraction
                info = ydl.extract_info(url, download=False)
                if info and ('formats' in info or 'url' in info):
                    ydl.download([url])
                    return output_path

        except Exception as e:
            print(f"Strategy {i + 1} failed: {str(e)}")
            if i == len(extraction_configs) - 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"YouTube is actively blocking automated requests. All strategies failed. Last error: {str(e)}"
                )
            # Add random delay between strategies
            time.sleep(random.uniform(2, 5))
            continue


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


async def download_with_human_behavior(url: str, output_path: str) -> str:
    """Add human-like delays and behavior"""
    # Random delay to simulate human browsing
    await asyncio.sleep(random.uniform(3, 7))

    # Run in executor to avoid blocking
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, download_youtube_video, url, output_path)


async def process_video_clip(request: ClipRequest) -> str:
    """Process video clip in background with enhanced error handling"""
    clip_id = str(uuid.uuid4())

    # Create temporary file paths
    temp_video = TEMP_DIR / f"{clip_id}_full.%(ext)s"
    temp_clip = TEMP_DIR / f"{clip_id}_clip.mp4"

    try:
        # Pre-validate video accessibility
        if not is_video_accessible(request.youtube_url):
            raise HTTPException(
                status_code=400,
                detail="Video is currently unavailable or restricted. Please try a different video."
            )

        # Convert time formats
        start_seconds = time_to_seconds(request.start_time)
        end_seconds = time_to_seconds(request.end_time)

        # Download video with human-like behavior
        downloaded_file = await download_with_human_behavior(request.youtube_url, str(temp_video))

        # Find the actual downloaded file (yt-dlp changes extension)
        actual_file = None
        for ext in ['mp4', 'webm', 'mkv', 'm4a']:
            potential_file = TEMP_DIR / f"{clip_id}_full.{ext}"
            if potential_file.exists():
                actual_file = str(potential_file)
                break

        if not actual_file:
            raise HTTPException(status_code=500, detail="Downloaded file not found")

        # Trim video
        loop = asyncio.get_event_loop()
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
        "environment": os.getenv("RENDER_SERVICE_NAME", "development"),
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": os.getenv("RENDER_SERVICE_NAME", "local"),
        "region": os.getenv("RENDER_REGION", "unknown"),
        "max_duration": MAX_DURATION,
        "temp_dir": str(TEMP_DIR)
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
        "max_workers": executor._max_workers,
        "temp_directory": str(TEMP_DIR)
    }


@app.get("/test-video")
async def test_video_accessibility(url: str):
    """Test if a YouTube video is accessible before processing"""
    try:
        accessible = is_video_accessible(url)
        return {
            "url": url,
            "accessible": accessible,
            "message": "Video is accessible" if accessible else "Video may be restricted or unavailable"
        }
    except Exception as e:
        return {
            "url": url,
            "accessible": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
