import requests
import time
import os

API_BASE = "https://jsuullmoohfg.us-east-1.clawcloudrun.com"


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
    """Test video clip creation"""
    try:
        payload = {
            "youtube_url": "https://youtu.be/H0XYANRosVo?si=bf9X85v0zdbIzFpA",
            "start_time": "5",
            "end_time": "15"
        }

        print("Creating clip... (this may take a moment)")
        response = requests.post(f"{API_BASE}/clip", json=payload, timeout=120)

        if response.status_code == 200:
            # Save the clip
            with open("test_clip.mp4", "wb") as f:
                f.write(response.content)

            file_size = os.path.getsize("test_clip.mp4")
            print(f"Clip Creation: SUCCESS - File size: {file_size} bytes")
            return True
        else:
            print(f"Clip Creation Failed: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"Clip Creation Failed: {e}")
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
