"""Integration tests for Team Member management and member-level authentication."""

from fastapi.testclient import TestClient


def test_member_registration_and_team_creation(test_client: TestClient) -> None:
    # 1. Register first member with a new team
    reg_res = test_client.post(
        "/v1/members/register",
        json={
            "email": "alice@team-eng.com",
            "team_id": "team_eng",
            "team_name": "Engineering Team",
        },
    )
    assert reg_res.status_code == 201
    alice_data = reg_res.json()
    assert alice_data["email"] == "alice@team-eng.com"
    assert alice_data["team_id"] == "team_eng"
    assert alice_data["api_key"].startswith("rag_m_")
    alice_key = alice_data["api_key"]

    # 2. Verify alice profile
    alice_headers = {"Authorization": f"Bearer {alice_key}"}
    me_res = test_client.get("/v1/members/me", headers=alice_headers)
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "alice@team-eng.com"
    assert me_res.json()["team_id"] == "team_eng"

    # 3. Verify team profile
    team_res = test_client.get("/v1/teams/me", headers=alice_headers)
    assert team_res.status_code == 200
    assert team_res.json()["name"] == "Engineering Team"


def test_multiple_members_in_same_team(test_client: TestClient) -> None:
    # 1. Register Alice
    res1 = test_client.post(
        "/v1/members/register",
        json={
            "email": "lead@fintech.com",
            "team_id": "fintech_team",
            "team_name": "Fintech Solutions",
        },
    )
    assert res1.status_code == 201
    key1 = res1.json()["api_key"]

    # 2. Register Bob specifying the same team
    res2 = test_client.post(
        "/v1/members/register",
        json={
            "email": "developer@fintech.com",
            "team_id": "fintech_team",
        },
    )
    assert res2.status_code == 201
    key2 = res2.json()["api_key"]
    assert key1 != key2

    # 3. List members from either member's key
    list_res1 = test_client.get(
        "/v1/teams/fintech_team/members",
        headers={"Authorization": f"Bearer {key1}"},
    )
    assert list_res1.status_code == 200
    members = list_res1.json()
    assert len(members) == 2
    emails = {m["email"] for m in members}
    assert "lead@fintech.com" in emails
    assert "developer@fintech.com" in emails

    list_res2 = test_client.get(
        "/v1/teams/fintech_team/members",
        headers={"Authorization": f"Bearer {key2}"},
    )
    assert list_res2.status_code == 200
    assert len(list_res2.json()) == 2


def test_duplicate_email_registration_rejected(test_client: TestClient) -> None:
    # Register first
    res = test_client.post(
        "/v1/members/register",
        json={
            "email": "duplicate@test.com",
            "team_id": "team_dup",
        },
    )
    assert res.status_code == 201

    # Attempt second registration with same email
    dup_res = test_client.post(
        "/v1/members/register",
        json={
            "email": "duplicate@test.com",
            "team_id": "team_dup_other",
        },
    )
    assert dup_res.status_code == 409
    assert "already registered" in dup_res.json()["detail"]


def test_regenerate_member_api_key(test_client: TestClient) -> None:
    # 1. Register member
    reg_res = test_client.post(
        "/v1/members/register",
        json={
            "email": "rotate@test.com",
            "team_id": "team_rotate",
        },
    )
    old_key = reg_res.json()["api_key"]

    # 2. Rotate API key
    rotate_res = test_client.post(
        "/v1/members/me/api-key",
        headers={"Authorization": f"Bearer {old_key}"},
    )
    assert rotate_res.status_code == 200
    new_key = rotate_res.json()["api_key"]
    assert new_key != old_key

    # 3. Old key should now be unauthorized
    unauth_res = test_client.get(
        "/v1/members/me",
        headers={"Authorization": f"Bearer {old_key}"},
    )
    assert unauth_res.status_code == 401

    # 4. New key should succeed
    auth_res = test_client.get(
        "/v1/members/me",
        headers={"Authorization": f"Bearer {new_key}"},
    )
    assert auth_res.status_code == 200
    assert auth_res.json()["email"] == "rotate@test.com"


def test_add_and_remove_team_member(test_client: TestClient) -> None:
    # 1. Register owner
    reg = test_client.post(
        "/v1/members/register",
        json={"email": "owner@ops.com", "team_id": "ops_team"},
    )
    headers = {"Authorization": f"Bearer {reg.json()['api_key']}"}

    # 2. Add member via team endpoint
    add_res = test_client.post(
        "/v1/teams/ops_team/members?email=intern@ops.com",
        headers=headers,
    )
    assert add_res.status_code == 201
    intern_key = add_res.json()["api_key"]
    assert intern_key.startswith("rag_m_")

    # 3. List members
    list_res = test_client.get("/v1/teams/ops_team/members", headers=headers)
    assert len(list_res.json()) == 2

    # 4. Remove intern
    del_res = test_client.delete(
        "/v1/teams/ops_team/members/intern@ops.com",
        headers=headers,
    )
    assert del_res.status_code == 204

    # 5. List members again
    list_res2 = test_client.get("/v1/teams/ops_team/members", headers=headers)
    assert len(list_res2.json()) == 1


def test_cross_team_isolation_between_members(test_client: TestClient) -> None:
    # Team Red
    red_reg = test_client.post(
        "/v1/members/register",
        json={"email": "red@redcorp.com", "team_id": "team_red"},
    )
    red_headers = {"Authorization": f"Bearer {red_reg.json()['api_key']}"}

    # Team Blue
    blue_reg = test_client.post(
        "/v1/members/register",
        json={"email": "blue@bluecorp.com", "team_id": "team_blue"},
    )
    blue_headers = {"Authorization": f"Bearer {blue_reg.json()['api_key']}"}

    # Red attempts to view Blue members
    forbidden_res = test_client.get(
        "/v1/teams/team_blue/members",
        headers=red_headers,
    )
    assert forbidden_res.status_code == 403

    # Red creates a dataset
    ds_res = test_client.post(
        "/v1/datasets",
        json={"id": "red_ds", "name": "Red Data"},
        headers=red_headers,
    )
    assert ds_res.status_code == 201

    # Blue listing datasets should NOT see Red's dataset
    blue_ds = test_client.get("/v1/datasets", headers=blue_headers)
    assert blue_ds.status_code == 200
    assert len(blue_ds.json()) == 0
