import pytest
from datetime import date
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["ml_model_loaded"] is True


def test_extract_endpoint_success():
    payload = {
        "transcript": "Rahul: I'll complete the API integration by Wednesday.\nPriya: Amit, please review the design.",
        "meeting_date": "2026-10-27",
        "extraction_mode": "hybrid",
    }
    res = client.post("/api/meetings/extract", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["tasks"]) >= 2
    assert data["action_item_count"] >= 2
    assert data["extraction_mode"] in ("hybrid", "rule_based", "ml")
    
    assignees = [t["assignee"] for t in data["tasks"]]
    assert "Rahul" in assignees
    assert "Amit" in assignees


def test_extract_endpoint_empty_transcript():
    payload = {
        "transcript": "   \n\t   ",
        "meeting_date": "2026-10-27",
    }
    res = client.post("/api/meetings/extract", json=payload)
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_crud_and_export_lifecycle():
    # 1. Save Meeting
    save_payload = {
        "title": "Weekly Engineering Sync",
        "meeting_date": "2026-10-27",
        "transcript": "Rahul: I will prepare the presentation by Friday.",
        "extraction_mode": "hybrid",
        "tasks": [
            {
                "description": "Prepare the presentation",
                "assignee": "Rahul",
                "deadline_text": "by Friday",
                "deadline": "2026-10-30",
                "status": "pending",
                "classification": "ACTION_ITEM",
                "confidence": 0.88,
                "evidence": "Rahul: I will prepare the presentation by Friday.",
                "needs_review": False,
            }
        ],
    }
    create_res = client.post("/api/meetings", json=save_payload)
    assert create_res.status_code == 201
    meeting_data = create_res.json()
    meeting_id = meeting_data["id"]
    task_id = meeting_data["tasks"][0]["id"]
    assert meeting_id is not None
    assert task_id is not None

    # 2. List Meetings
    list_res = client.get("/api/meetings")
    assert list_res.status_code == 200
    meetings = list_res.json()["meetings"]
    assert any(m["id"] == meeting_id for m in meetings)

    # 3. Get Meeting
    get_res = client.get(f"/api/meetings/{meeting_id}")
    assert get_res.status_code == 200
    assert get_res.json()["title"] == "Weekly Engineering Sync"

    # 4. Patch Task
    patch_res = client.patch(
        f"/api/tasks/{task_id}",
        json={"status": "confirmed", "description": "Prepare the client presentation"},
    )
    assert patch_res.status_code == 200
    updated_task = patch_res.json()
    assert updated_task["status"] == "confirmed"
    assert updated_task["description"] == "Prepare the client presentation"

    # 5. Export as CSV
    csv_res = client.get(f"/api/meetings/{meeting_id}/export?format=csv")
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.headers["content-type"]
    assert "Prepare the client presentation" in csv_res.text

    # 6. Export as JSON
    json_res = client.get(f"/api/meetings/{meeting_id}/export?format=json")
    assert json_res.status_code == 200
    assert "application/json" in json_res.headers["content-type"]
    exported_tasks = json_res.json()
    assert len(exported_tasks) == 1
    assert exported_tasks[0]["status"] == "confirmed"

    # 7. Delete Task
    del_task_res = client.delete(f"/api/tasks/{task_id}")
    assert del_task_res.status_code == 204

    # 8. Delete Meeting
    del_meeting_res = client.delete(f"/api/meetings/{meeting_id}")
    assert del_meeting_res.status_code == 204


def test_upload_file_valid_txt():
    content = b"Rahul: I'll complete the API.\nPriya: Amit, please review."
    files = {"file": ("test_transcript.txt", content, "text/plain")}
    res = client.post("/api/meetings/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["format"] == ".txt"
    assert "Rahul: I'll complete the API." in data["content"]


def test_upload_file_unsupported_extension():
    content = b"some random content"
    files = {"file": ("malicious.exe", content, "application/octet-stream")}
    res = client.post("/api/meetings/upload", files=files)
    assert res.status_code == 400
    assert "unsupported" in res.json()["detail"].lower()


def test_benchmark_endpoint():
    res = client.get("/api/benchmark")
    assert res.status_code == 200
    data = res.json()
    assert "dataset_summary" in data
    assert data["dataset_summary"]["total_utterances"] == 15179
    assert "classifier_comparison" in data
    assert len(data["classifier_comparison"]) >= 6
    assert "ner_and_slot_filling" in data


def test_extract_v2_transformer():
    payload = {
        "transcript": "Rahul: I will complete the documentation by next Monday.",
        "meeting_date": "2026-10-27",
        "extraction_mode": "v2_transformer",
    }
    res = client.post("/api/meetings/extract", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["extraction_mode"] == "v2_transformer"
    assert len(data["tasks"]) >= 1
    assert data["tasks"][0]["assignee"] == "Rahul"

