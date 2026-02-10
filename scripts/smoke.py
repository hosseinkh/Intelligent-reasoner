import requests
import sys

BASE_URL = "http://127.0.0.1:8000"
print("smoke code is starting")

def check(name, response):
    if response.status_code != 200:
        print(f"❌ {name} failed: {response.status_code}")
        print(response.text)
        sys.exit(1)
    print(f"✅ {name} OK")


def main():
    print("🚀 Running smoke test...\n")

    # 1) Health
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    check("health", r)

    # 2) Ready
    r = requests.get(f"{BASE_URL}/ready", timeout=5)
    check("ready", r)

    # 3) Ask – non-RAG
    r = requests.post(
        f"{BASE_URL}/ask",
        json={"query": "How are you?"},
        timeout=10,
    )
    check("ask (non-rag)", r)

    # 4) Ask – RAG path
    r = requests.post(
        f"{BASE_URL}/ask",
        json={"query": "Selon ANSM, pourquoi un médicament est en pénurie ?"},
        timeout=30,
    )
    check("ask (rag)", r)

    print("\n🎉 Smoke test passed. API is healthy.")


if __name__ == "__main__":
    main()
