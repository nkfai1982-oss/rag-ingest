"""Integration tests for FastAPI REST endpoints."""

import io

from fastapi.testclient import TestClient


def test_api_auth_required(test_client: TestClient) -> None:
    # Missing auth header
    res = test_client.get("/v1/teams/me")
    assert res.status_code == 401

    # Invalid auth token
    res = test_client.get("/v1/teams/me", headers={"Authorization": "Bearer invalid_token"})
    assert res.status_code == 401


def test_api_team_and_dataset_lifecycle(test_client: TestClient) -> None:
    # 1. Register team
    create_res = test_client.post(
        "/v1/teams",
        json={"name": "New Team Alpha", "default_chunk_size": 400},
    )
    assert create_res.status_code == 201
    team_data = create_res.json()
    api_key = team_data["api_key"]
    team_id = team_data["id"]
    assert team_id.startswith("team_")

    headers = {"Authorization": f"Bearer {api_key}"}

    # 2. Get my team
    me_res = test_client.get("/v1/teams/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.json()["name"] == "New Team Alpha"

    # 3. Create dataset
    ds_res = test_client.post(
        "/v1/datasets",
        json={"name": "Engineering Docs", "id": "eng_docs"},
        headers=headers,
    )
    assert ds_res.status_code == 201
    assert ds_res.json()["id"] == "eng_docs"

    # 4. List datasets
    list_ds = test_client.get("/v1/datasets", headers=headers)
    assert list_ds.status_code == 200
    assert len(list_ds.json()) == 1


def test_api_sync_and_async_ingest_flow(test_client: TestClient) -> None:
    headers = {"Authorization": "Bearer test_api_key_123"}

    # 1. Synchronous Ingestion
    content = b"# Architecture Overview\n\nThis guide describes the microservice architecture."
    sync_res = test_client.post(
        "/v1/ingest",
        data={"dataset_id": "test_dataset_1", "async_mode": "false"},
        files={"file": ("architecture.md", io.BytesIO(content), "text/markdown")},
        headers=headers,
    )
    assert sync_res.status_code == 200
    sync_data = sync_res.json()
    assert sync_data["status"] == "SUCCESS"
    doc_id = sync_data["document_id"]
    assert sync_data["total_chunks"] >= 1

    # 2. Asynchronous Ingestion
    async_content = b"# Async Guide\n\nThis guide is processed asynchronously via background task."
    async_res = test_client.post(
        "/v1/ingest",
        data={"dataset_id": "test_dataset_1", "async_mode": "true"},
        files={"file": ("async_guide.md", io.BytesIO(async_content), "text/markdown")},
        headers=headers,
    )
    assert async_res.status_code == 202
    job_id = async_res.json()["job_id"]

    # Check job status
    job_res = test_client.get(f"/v1/jobs/{job_id}", headers=headers)
    assert job_res.status_code == 200
    # In test environment with BackgroundTasks, the background task executes
    assert job_res.json()["status"] in ("RUNNING", "SUCCESS")

    # 3. Retrieval - Hybrid Search
    search_res = test_client.post(
        "/v1/retrieval",
        json={
            "query": "architecture overview microservice",
            "search_mode": "hybrid",
            "top_k": 3,
        },
        headers=headers,
    )
    assert search_res.status_code == 200
    results = search_res.json()["results"]
    assert len(results) >= 1
    assert any("architecture" in r["content"].lower() for r in results)

    # 4. List Documents
    docs_res = test_client.get("/v1/documents", headers=headers)
    assert docs_res.status_code == 200
    assert len(docs_res.json()) >= 1

    # 5. Download Document
    dl_res = test_client.get(f"/v1/documents/{doc_id}/download", headers=headers)
    assert dl_res.status_code == 200
    assert dl_res.content == content

    # 6. Delete Document
    del_res = test_client.delete(f"/v1/documents/{doc_id}", headers=headers)
    assert del_res.status_code == 204

    # Confirm deletion
    post_del_res = test_client.get(f"/v1/documents/{doc_id}/download", headers=headers)
    assert post_del_res.status_code == 404
