from __future__ import annotations

from fastapi.testclient import TestClient

from alphaevolve_webui.backend.main import create_app


def test_product_api_lists_tasks_with_evaluator_contracts() -> None:
    client = TestClient(create_app())

    response = client.get("/api/tasks")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    task_ids = {task["task_id"] for task in payload["data"]}
    assert len(task_ids) >= 20
    assert {"sort_numbers", "find_first_index"}.issubset(task_ids)
    assert all("initial_program" in task for task in payload["data"])


def test_product_api_runs_search_and_returns_archive_evidence() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/runs",
        json={"task_id": "sort_numbers", "generations": 3, "archive_size": 4},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    run = payload["data"]
    assert run["status"] == "completed"
    assert run["best_candidate"]["score"] == 1.0
    assert run["best_candidate"]["parent_id"] is not None
    assert len(run["archive"]) <= 4
    assert run["history"][1]["prompt"]
    assert run["history"][1]["evaluation"]["total_cases"] == 4

    fetched = client.get(f"/api/runs/{run['run_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["data"]["run_id"] == run["run_id"]


def test_product_api_exports_best_program() -> None:
    client = TestClient(create_app())
    run = client.post(
        "/api/runs",
        json={"task_id": "find_first_index", "generations": 2},
    ).json()["data"]

    response = client.get(f"/api/runs/{run['run_id']}/export")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/x-python")
    assert "def solve" in response.text
    assert "return" in response.text


def test_product_api_runs_custom_task_from_user_supplied_cases() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/runs/custom",
        json={
            "title": "Double a number",
            "objective": "Return twice the input number.",
            "function_name": "solve",
            "initial_program": "def solve(value):\n    return value\n",
            "generations": 1,
            "archive_size": 4,
            "constraints": ["Return a number."],
            "cases": [
                {
                    "case_id": "double_positive",
                    "args": [3],
                    "expected": 6,
                    "description": "Positive integer.",
                },
                {
                    "case_id": "double_zero",
                    "args": [0],
                    "expected": 0,
                    "description": "Zero stays zero.",
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    run = payload["data"]
    assert run["task"]["title"] == "Double a number"
    assert run["task"]["cases"][0]["case_id"] == "double_positive"
    assert run["history"][1]["prompt"]
    assert "double_positive" in run["history"][1]["prompt"]
