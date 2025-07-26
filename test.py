import requests
import time
import os

API_BASE = "https://clipper-clwz.onrender.com/"


def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"Health Check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health Check Failed: {e}")
        return False


def test_limits():
    """Test limits endpoint"""
    try:
        response = requests.get(f"{API_BASE}/limits")
        print(f"Limits: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Limits Check Failed: {e}")
        return False


def test_clip_creation():
    """Test video clip creation with multiple fallback videos"""
    # Known working videos (older, well-established, less likely to be restricted)
    test_videos = [
        {
            "url": "https://www.youtube.com/watch?v=jNQXAC9IVRw",
            "name": "Me at the zoo (2005 - First YouTube video)"
        },
        {
            "url": "https://www.youtube.com/watch?v=hFcLyDb6niA",
            "name": "Keyboard Cat (2007)"
        },
        {
            "url": "https://www.youtube.com/watch?v=oHg5SJYRHA0",
            "name": "RickRoll original (2009)"
        },
        {
            "url": "https://www.youtube.com/watch?v=kJQP7kiw5Fk",
            "name": "Despacito (Popular music video)"
        }
    ]

    for i, video in enumerate(test_videos):
        try:
            print(f"Testing video {i + 1}/{len(test_videos)}: {video['name']}")

            # First test if video is accessible
            test_response = requests.get(f"{API_BASE}/test-video?url={video['url']}")
            if test_response.status_code == 200:
                test_data = test_response.json()
                print(f"  Accessibility check: {'✅ Accessible' if test_data.get('accessible') else '❌ Not accessible'}")

                if not test_data.get('accessible'):
                    continue

            payload = {
                "youtube_url": video['url'],
                "start_time": "5",
                "end_time": "15"
            }

            print("  Creating clip... (this may take a moment)")
            response = requests.post(f"{API_BASE}/clip", json=payload, timeout=180)

            if response.status_code == 200:
                filename = f"test_clip_{i + 1}.mp4"
                with open(filename, "wb") as f:
                    f.write(response.content)

                file_size = os.path.getsize(filename)
                print(f"  ✅ SUCCESS - File size: {file_size} bytes")
                print(f"Clip Creation: SUCCESS with {video['name']}")
                return True
            else:
                print(f"  ❌ Failed: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("❌ All test videos failed")
    return False


def run_all_tests():
    """Run all tests"""
    print("=== YouTube Clipper API Test Suite ===\n")

    tests = [
        ("Health Check", test_health),
        ("Limits Check", test_limits),
        ("Clip Creation", test_clip_creation)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
        print(f"{'✅ PASSED' if result else '❌ FAILED'}\n")
        time.sleep(1)

    print("=== Test Results ===")
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")

    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")


if __name__ == "__main__":
    run_all_tests()
