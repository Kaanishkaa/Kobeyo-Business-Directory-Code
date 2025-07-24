import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

print("🧪 Testing Email Classifier API")
print("=" * 50)

# Test single classification
print("\n1️⃣ Testing single email classification:")
response = requests.post(f"{BASE_URL}/classify", json={
    "email": "careers@techcompany.com",
    "include_explanation": True
})
print(json.dumps(response.json(), indent=2))

# Test batch classification
print("\n2️⃣ Testing batch classification:")
response = requests.post(f"{BASE_URL}/classify/batch", json={
    "emails": [
        "hiring@startup.com",
        "sales@company.com",
        "info@business.com",
        "orders@shop.com",
        "support@company.com"
    ]
})
print(json.dumps(response.json(), indent=2))

# Test feedback
print("\n3️⃣ Testing feedback submission:")
response = requests.post(f"{BASE_URL}/feedback", json={
    "email": "info@business.com",
    "correct_category": "Sales"
})
print(json.dumps(response.json(), indent=2))

print("\n✅ API tests complete!")
