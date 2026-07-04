import json
from app import app

client = app.test_client()

def test_crud():
    print("--- Testing CREATE ---")
    data = {
        "title": "Test Title",
        "text": "This is test text",
        "subject": "politics",
        "label": "FAKE"
    }
    res = client.post("/api/news", json=data)
    print("POST /api/news Status:", res.status_code, res.data.decode('utf-8'))

    print("--- Testing READ ---")
    res = client.get("/api/news")
    print("GET /api/news Status:", res.status_code)
    resp_data = json.loads(res.data)
    print("Data:", resp_data)
    
    news_id = resp_data["data"][0]["id"]
    
    print("--- Testing UPDATE ---")
    update_data = {
        "title": "Updated Title",
        "text": "Updated text",
        "subject": "health",
        "label": "REAL"
    }
    res = client.put(f"/api/news/{news_id}", json=update_data)
    print(f"PUT /api/news/{news_id} Status:", res.status_code, res.data.decode('utf-8'))
    
    print("--- Testing READ AFTER UPDATE ---")
    res = client.get("/api/news")
    resp_data = json.loads(res.data)
    print("Data:", resp_data)

    print("--- Testing DELETE ---")
    res = client.delete(f"/api/news/{news_id}")
    print(f"DELETE /api/news/{news_id} Status:", res.status_code, res.data.decode('utf-8'))

    print("--- Testing READ AFTER DELETE ---")
    res = client.get("/api/news")
    resp_data = json.loads(res.data)
    print("Current Count:", len(resp_data["data"]))

if __name__ == "__main__":
    test_crud()
