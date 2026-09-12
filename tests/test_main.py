from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_and_dashboard_are_available():
    assert client.get("/health").status_code == 200
    dashboard = client.get("/dashboard/")
    assert dashboard.status_code == 200
    assert "AegisOps" in dashboard.text


def test_fault_and_approval_workflow_is_auditable():
    fault = client.post("/admin/fault", json={"mode": "errors"})
    assert fault.status_code == 200
    assert fault.json()["active_fault"] == "errors"

    created = client.post(
        "/admin/remediations",
        json={
            "action": "route_to_last_known_healthy_model",
            "requested_by": "test-operator",
            "rationale": "The measured error rate exceeded the incident threshold.",
        },
    )
    assert created.status_code == 201
    request_id = created.json()["id"]
    assert created.json()["status"] == "pending_human_approval"

    approved = client.post(
        f"/admin/remediations/{request_id}/approve",
        json={"approved_by": "test-approver"},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved_for_execution"
    assert approved.json()["approved_by"] == "test-approver"
