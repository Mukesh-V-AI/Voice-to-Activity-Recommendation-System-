import requests
import json

def test_api():
    """Test the API endpoints."""
    base_url = "http://localhost:8000"

    print("🧪 Testing Voice Activity Recommendation API...")

    try:
        # Test health endpoint
        print("\n1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")

        # Test text recommendation
        print("\n2. Testing text recommendation...")
        test_text = {"text": "I feel stressed and need something relaxing for 20 minutes"}
        response = requests.post(f"{base_url}/recommend/text", json=test_text)
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"   Intent: {result['intent_summary']}")
            print(f"   Recommendations: {len(result['recommendations'])}")

            for i, rec in enumerate(result['recommendations'][:2], 1):
                print(f"   #{i}: {rec['activity']} (Score: {rec['score']:.2f})")

        # Test stats endpoint
        print("\n3. Testing stats endpoint...")
        response = requests.get(f"{base_url}/stats")
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            stats = response.json()
            dataset_stats = stats['dataset_stats']
            print(f"   Total activities: {dataset_stats.get('total_activities', 0)}")
            print(f"   Categories: {list(dataset_stats.get('categories', {}).keys())}")

        print("\n✅ All tests completed!")

    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API. Make sure the backend is running:")
        print("   cd app && python main.py")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    test_api()
