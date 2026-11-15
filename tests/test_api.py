"""
Unit and integration tests for FastAPI endpoints.
"""
import pytest
import asyncio


@pytest.mark.asyncio
async def test_health_check(async_client):
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "llm_mode" in data


@pytest.mark.asyncio
async def test_legacy_endpoint_backward_compatibility(async_client):
    payload = {"text": "The biryani was delicious and warm."}
    response = await async_client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "score" in data
    assert "theme" in data
    assert data["label"] == "positive"
    assert data["score"] >= 4


@pytest.mark.asyncio
async def test_v1_single_comprehensive_analyze(async_client):
    payload = {"text": "Cold food again. This is the third time delivery was late."}
    response = await async_client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["overall_label"] == "negative"
    assert data["overall_score"] <= 2
    assert len(data["aspects"]) > 0
    assert any(a["aspect"] in ["delivery_speed", "food_quality"] for a in data["aspects"])
    assert data["urgency"] in ["medium", "high", "critical"]
    assert data["root_cause"] is not None


@pytest.mark.asyncio
async def test_batch_job_submission_and_polling(async_client):
    reviews = [
        "Amazing taste and the order arrived hot and early.",
        "Prices have gone up too much for the same small portion.",
        "Service was rude when I called to ask about my order.",
    ]
    # 1. Submit batch job
    submit_res = await async_client.post("/api/v1/jobs", json={"reviews": reviews})
    assert submit_res.status_code == 202
    job_info = submit_res.json()
    job_id = job_info["job_id"]
    assert job_info["total"] == 3

    # 2. Poll for background execution to complete
    status_data = {}
    for _ in range(30):
        await asyncio.sleep(0.1)
        status_res = await async_client.get(f"/api/v1/jobs/{job_id}")
        assert status_res.status_code == 200
        status_data = status_res.json()
        if status_data["completed"] == 3:
            break

    # 3. Assert final status
    assert status_data["job_id"] == job_id
    assert status_data["completed"] == 3
    assert len(status_data["results"]) == 3


@pytest.mark.asyncio
async def test_metrics_endpoint(async_client):
    res = await async_client.get("/api/v1/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "system_metrics" in data
    assert "feedback_analytics" in data

