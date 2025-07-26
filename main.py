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
import base64
import json
import socket
import struct

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
import yt_dlp

app = FastAPI(
    title="YouTube Video Clipper API - Enhanced",
    description="Download and trim YouTube video clips with advanced anti-detection",
    version="2.0.0"
)

# Configuration - Environment variables with defaults
MAX_DURATION = int(os.getenv("MAX_CLIP_DURATION", "300"))
TEMP_DIR = Path(os.getenv("TEMP_DIR", "/tmp/video_clips"))
TEMP_DIR.mkdir(exist_ok=True)

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=int(os.getenv("MAX_WORKERS", "2")))


class ClipRequest(BaseModel):
    youtube_url: str
    start_time: str
    end_time: str

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


def generate_fake_ip():
    """Generate a realistic fake IP address for headers"""
    # Generate residential IP ranges
    residential_ranges = [
        (192, 168, random.randint(1, 254), random.randint(1, 254)),
        (10, random.randint(0, 255), random.randint(0, 255), random.randint(1, 254)),
        (172, random.randint(16, 31), random.randint(0, 255), random.randint(1, 254)),
        # Public residential ranges
        (76, random.randint(1, 254), random.randint(1, 254), random.randint(1, 254)),
        (98, random.randint(1, 254), random.randint(1, 254), random.randint(1, 254)),
    ]
    ip_tuple = random.choice(residential_ranges)
    return f"{ip_tuple[0]}.{ip_tuple[1]}.{ip_tuple[2]}.{ip_tuple[3]}"


def get_random_user_agent():
    """Get a random realistic user agent"""
    user_agents = [
        # Chrome on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",

        # Chrome on Mac
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",

        # Firefox on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",

        # Safari on Mac
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",

        # Chrome on Android
        "Mozilla/5.0 (Linux; Android 13; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36",

        # Android Music App
        "com.google.android.apps.youtube.music/6.42.52 (Linux; U; Android 12; Pixel 6) gzip",
        "com.google.android.apps.youtube.music/7.02.52 (Linux; U; Android 13; Pixel 7) gzip",

        # iOS YouTube
        "com.google.ios.youtube/19.02.3 (iPhone14,3; U; CPU iOS 16_0 like Mac OS X)",
        "com.google.ios.youtube/19.05.2 (iPhone15,2; U; CPU iOS 17_0 like Mac OS X)",

        # Android TV
        "com.google.android.youtube.tv/2.12.08 (Linux; U; Android 9; AFTT Build/PS7329.3153N) gzip",
    ]
    return random.choice(user_agents)


def get_residential_headers():
    """Generate realistic residential browsing headers"""
    fake_ip = generate_fake_ip()
    user_agent = get_random_user_agent()

    headers = {
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': random.choice(['en-US,en;q=0.9', 'en-GB,en;q=0.9', 'en-CA,en;q=0.9']),
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': random.choice(['none', 'same-origin', 'cross-site']),
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
        'X-Forwarded-For': fake_ip,
        'X-Real-IP': fake_ip,
        'CF-Connecting-IP': fake_ip,
    }

    # Add Chrome-specific headers if Chrome user agent
    if 'Chrome' in user_agent:
        headers.update({
            'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            'sec-ch-ua-mobile': '?0' if 'Mobile' not in user_agent else '?1',
            'sec-ch-ua-platform': '"Windows"' if 'Windows' in user_agent else '"macOS"' if 'Mac' in user_agent else '"Linux"',
        })

    # Add referrer to simulate browsing
    referrers = [
        'https://www.google.com/',
        'https://www.youtube.com/',
        'https://duckduckgo.com/',
        'https://www.bing.com/',
        'https://search.yahoo.com/',
    ]
    headers['Referer'] = random.choice(referrers)

    return headers


def create_fake_cookies():
    """Create fake but realistic YouTube session cookies"""
    session_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=32))
    visitor_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_', k=24))

    cookies = {
        'VISITOR_INFO1_LIVE': visitor_id,
        'YSC': ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_', k=12)),
        'PREF': f'f1=50000000&f6=40000000&hl=en&gl=US&f4=4000000&f7=100',
        'CONSENT': 'PENDING+999',
        'GPS': '1',
        '__Secure-3PAPISID': ''.join(
            random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=32)),
        '__Secure-3PSID': ''.join(
            random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=32)),
        'LOGIN_INFO': f'AFmmF2swRQIhAI{session_id}',
        'SID': ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=32)),
        'HSID': ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=16)),
        'SSID': ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=16)),
        'APISID': ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=32)),
        'SAPISID': ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=32)),
    }
    return cookies


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


def simulate_human_behavior():
    """Add random delays to simulate human behavior"""
    # Random delay between 2-8 seconds
    delay = random.uniform(2, 8)
    return delay


def download_youtube_video(url: str, output_path: str) -> str:
    """Download YouTube video with comprehensive anti-detection"""

    # Multiple extraction strategies with different approaches
    extraction_configs = [
        # Strategy 1: Android Music with fake residential headers
        {
            'http_headers': {
                **get_residential_headers(),
                'User-Agent': 'com.google.android.apps.youtube.music/7.02.52 (Linux; U; Android 13; Pixel 7) gzip',
                'X-YouTube-Client-Name': '21',
                'X-YouTube-Client-Version': '7.02.52',
            },
            'extractor_args': {
                'youtube': {
                    'player_client': ['android_music'],
                    'player_skip': ['webpage', 'configs'],
                }
            },
            'cookies': create_fake_cookies(),
        },

        # Strategy 2: iOS Client with mobile headers
        {
            'http_headers': {
                **get_residential_headers(),
                'User-Agent': 'com.google.ios.youtube/19.05.2 (iPhone15,2; U; CPU iOS 17_0 like Mac OS X)',
                'X-YouTube-Client-Name': '5',
                'X-YouTube-Client-Version': '19.05.2',
            },
            'extractor_args': {
                'youtube': {
                    'player_client': ['ios'],
                }
            },
        },

        # Strategy 3: TV Embedded (often bypasses restrictions)
        {
            'http_headers': get_residential_headers(),
            'extractor_args': {
                'youtube': {
                    'player_client': ['tv_embedded'],
                }
            },
            'age_limit': 99,
            'geo_bypass': True,
        },

        # Strategy 4: Android Creator Studio
        {
            'http_headers': {
                **get_residential_headers(),
                'User-Agent': 'com.google.android.apps.youtube.creator/22.30.100 (Linux; U; Android 11) gzip',
            },
            'extractor_args': {
                'youtube': {
                    'player_client': ['android_creator'],
                }
            },
        },

        # Strategy 5: Web client with full browser simulation
        {
            'http_headers': get_residential_headers(),
            'extractor_args': {
                'youtube': {
                    'player_client': ['web'],
                    'player_skip': ['configs'],
                }
            },
            'cookies': create_fake_cookies(),
        },

        # Strategy 6: Mobile web
        {
            'http_headers': {
                **get_residential_headers(),
                'User-Agent': 'Mozilla/5.0 (Linux; Android 13; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36',
            },
            'extractor_args': {
                'youtube': {
                    'player_client': ['mweb'],
                }
            },
        },

        # Strategy 7: Last resort - basic extraction with lowest quality
        {
            'format': 'worst[height<=360]/worst',
            'http_headers': get_residential_headers(),
            'extractor_args': {
                'youtube': {
                    'player_client': ['web'],
                }
            },
        }
    ]

    base_opts = {
        'format': 'best[height<=720]/best',
        'outtmpl': output_path,
        'no_warnings': True,
        'no_cache_dir': True,
        'force_ipv4': True,
        'retries': 2,
        'fragment_retries': 2,
        'socket_timeout': 120,
        'sleep_interval': random.uniform(3, 7),
        'max_sleep_interval': 15,
        'ignoreerrors': False,
        'geo_bypass': True,
        'geo_bypass_country': random.choice(['US', 'CA', 'GB', 'AU']),
    }

    for i, config in enumerate(extraction_configs):
        try:
            print(f"Attempting strategy {i + 1}/7...")

            # Human-like delay before each attempt
            time.sleep(simulate_human_behavior())

            opts = {**base_opts, **config}

            with yt_dlp.YoutubeDL(opts) as ydl:
                # Try to extract info first
                try:
                    info = ydl.extract_info(url, download=False)
                    if info and ('formats' in info or 'url' in info):
                        print(f"Info extraction successful with strategy {i + 1}")
                        ydl.download([url])
                        print(f"Download completed with strategy {i + 1}")
                        return output_path
                except Exception as extract_error:
                    print(f"Info extraction failed: {extract_error}")
                    # Try direct download as fallback
                    try:
                        ydl.download([url])
                        print(f"Direct download successful with strategy {i + 1}")
                        return output_path
                    except Exception as download_error:
                        print(f"Direct download also failed: {download_error}")
                        raise download_error

        except Exception as e:
            print(f"Strategy {i + 1} failed: {str(e)}")
            if i == len(extraction_configs) - 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"All {len(extraction_configs)} anti-detection strategies failed. YouTube is actively blocking server access. Last error: {str(e)}"
                )

            # Progressive backoff with randomization
            backoff_time = random.uniform(5, 15) * (i + 1)
            print(f"Waiting {backoff_time:.1f} seconds before next strategy...")
            time.sleep(backoff_time)
            continue


def is_video_accessible(url: str) -> bool:
    """Enhanced accessibility check with multiple strategies"""
    quick_check_configs = [
        {
            'extractor_args': {'youtube': {'player_client': ['android_music']}},
            'http_headers': {'User-Agent': get_random_user_agent()}
        },
        {
            'extractor_args': {'youtube': {'player_client': ['tv_embedded']}},
        },
        {
            'extractor_args': {'youtube': {'player_client': ['ios']}},
            'http_headers': {'User-Agent': 'com.google.ios.youtube/19.05.2'}
        }
    ]

    for config in quick_check_configs:
        try:
            opts = {
                'quiet': True,
                'no_warnings': True,
                'socket_timeout': 15,
                **config
            }

            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info and ('title' in info or 'formats' in info):
                    return True
        except:
            continue

    # If all quick checks fail, still allow the attempt (more permissive)
    return True


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
        '-y',
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
    await asyncio.sleep(simulate_human_behavior())

    # Run in executor to avoid blocking
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, download_youtube_video, url, output_path)


async def process_video_clip(request: ClipRequest) -> str:
    """Process video clip with enhanced error handling"""
    clip_id = str(uuid.uuid4())

    # Create temporary file paths
    temp_video = TEMP_DIR / f"{clip_id}_full.%(ext)s"
    temp_clip = TEMP_DIR / f"{clip_id}_clip.mp4"

    try:
        print(f"Processing request for: {request.youtube_url}")

        # More lenient pre-validation
        if not is_video_accessible(request.youtube_url):
            print("Video accessibility check failed, but proceeding anyway...")

        # Convert time formats
        start_seconds = time_to_seconds(request.start_time)
        end_seconds = time_to_seconds(request.end_time)

        print(f"Clip duration: {end_seconds - start_seconds} seconds")

        # Download video with enhanced anti-detection
        downloaded_file = await download_with_human_behavior(request.youtube_url, str(temp_video))

        # Find the actual downloaded file
        actual_file = None
        for ext in ['mp4', 'webm', 'mkv', 'm4a', 'flv']:
            potential_file = TEMP_DIR / f"{clip_id}_full.{ext}"
            if potential_file.exists():
                actual_file = str(potential_file)
                print(f"Found downloaded file: {actual_file}")
                break

        if not actual_file:
            raise HTTPException(status_code=500, detail="Downloaded file not found after successful download")

        # Trim video
        print("Starting video trimming...")
        loop = asyncio.get_event_loop()
        output_file = await loop.run_in_executor(
            executor, trim_video, actual_file, str(temp_clip), start_seconds, end_seconds
        )

        # Clean up full video file
        Path(actual_file).unlink(missing_ok=True)
        print(f"Processing completed: {output_file}")

        return str(temp_clip)

    except Exception as e:
        print(f"Processing failed: {str(e)}")
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
        "message": "YouTube Video Clipper API - Enhanced Anti-Detection",
        "docs": "/docs",
        "health": "/health",
        "environment": os.getenv("RENDER_SERVICE_NAME", "development"),
        "version": "2.0.0",
        "features": [
            "Advanced anti-bot detection",
            "Multiple extraction strategies",
            "Residential IP simulation",
            "Human behavior simulation",
            "Fake cookie generation"
        ]
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": os.getenv("RENDER_SERVICE_NAME", "local"),
        "region": os.getenv("RENDER_REGION", "unknown"),
        "max_duration": MAX_DURATION,
        "temp_dir": str(TEMP_DIR),
        "strategies": 7,
        "anti_detection": "enabled"
    }


@app.post("/clip")
async def create_clip(request: ClipRequest, background_tasks: BackgroundTasks):
    """Create a video clip from YouTube URL with enhanced anti-detection"""
    try:
        print(f"Received clip request: {request.youtube_url}")
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
        "temp_directory": str(TEMP_DIR),
        "extraction_strategies": 7,
        "anti_detection_features": [
            "Fake residential IP headers",
            "Random user agent rotation",
            "Fake cookie generation",
            "Human behavior simulation",
            "Multiple player client fallbacks",
            "Geographic bypass",
            "Progressive retry backoff"
        ]
    }


@app.get("/test-video")
async def test_video_accessibility(url: str):
    """Test if a YouTube video is accessible"""
    try:
        accessible = is_video_accessible(url)
        return {
            "url": url,
            "accessible": accessible,
            "message": "Video accessibility checked with multiple strategies",
            "note": "Even if marked as not accessible, download may still succeed with full anti-detection"
        }
    except Exception as e:
        return {
            "url": url,
            "accessible": False,
            "error": str(e)
        }


@app.get("/debug/strategies")
async def debug_strategies():
    """Debug endpoint to show available strategies"""
    return {
        "total_strategies": 7,
        "strategies": [
            {
                "id": 1,
                "name": "Android Music",
                "description": "Uses YouTube Music Android client with fake residential headers"
            },
            {
                "id": 2,
                "name": "iOS Client",
                "description": "Uses iOS YouTube app client"
            },
            {
                "id": 3,
                "name": "TV Embedded",
                "description": "Uses TV embedded player, bypasses many restrictions"
            },
            {
                "id": 4,
                "name": "Android Creator",
                "description": "Uses YouTube Creator Studio Android client"
            },
            {
                "id": 5,
                "name": "Web Browser",
                "description": "Full browser simulation with fake cookies"
            },
            {
                "id": 6,
                "name": "Mobile Web",
                "description": "Mobile browser simulation"
            },
            {
                "id": 7,
                "name": "Last Resort",
                "description": "Basic extraction with lowest quality as fallback"
            }
        ],
        "anti_detection_features": {
            "fake_residential_ips": "Generates realistic residential IP addresses",
            "user_agent_rotation": "Rotates between realistic browser and app user agents",
            "fake_cookies": "Generates fake but realistic YouTube session cookies",
            "human_delays": "Random delays between 2-8 seconds to simulate human behavior",
            "geographic_bypass": "Attempts to bypass geographic restrictions",
            "progressive_backoff": "Increasing delays between failed attempts"
        }
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
