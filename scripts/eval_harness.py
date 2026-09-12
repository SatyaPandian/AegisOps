"""Evaluation harness for the AegisOps LangGraph investigator/reporter agent.

Runs a series of simulated incidents with KNOWN ground-truth severity,
restarting the inference-service container between scenarios so each
measurement starts from clean, uncorrupted metrics and event history
(Prometheus counters and the event deque are in-process state that would
otherwise accumulate across scenarios and invalidate the comparison).

For each scenario:
  1. Restart inference-service and wait for /health.
  2. Set the fault mode and generate traffic.
  3. Run the LangGraph investigation pipeline against it.
  4. Compare the policy-derived severity to the expected ground truth.

Logs each scenario as a child MLflow run under a parent evaluation run, and
prints a final accuracy score across all scenarios.
"""

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import mlflow

from agentic_graph import run_investigation, MLFLOW_DB_PATH  # noqa: F401  (reuses same tracking URI)

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
mlflow.set_experiment("aegisops_eval_harness")


@dataclass
class Scenario:
    name: str
    fault_mode: str
    expected_severity: str
    requests: int = 20


SCENARIOS = [
    Scenario("baseline_healthy", "none", "low"),
    Scenario("upstream_errors", "errors", "high"),
    Scenario("low_prediction_confidence", "low_confidence", "medium"),
    Scenario("input_drift", "drift", "medium"),
    Scenario("baseline_healthy_repeat", "none", "low"),
    Scenario("upstream_errors_repeat", "errors", "high"),
    Scenario("low_prediction_confidence_repeat", "low_confidence", "medium"),
    Scenario("input_drift_repeat", "drift", "medium"),
]


def request_json(url: str, method: str = "GET", payload: Optional[Dict] = None) -> Tuple[int, Dict]:
    body = json.dumps(payload).encode("utf-8") if payload else None
    request = Request(url, data=body, method=method)
    request.add_header("Content-Type", "application/json")
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))
    except URLError as error:
        raise RuntimeError(f"Could not reach AegisOps at {url}: {error.reason}") from error


def restart_service_and_wait(base_url: str, timeout_seconds: int = 60) -> None:
    """Restart the container so Prometheus counters and events start clean."""
    subprocess.run(
        ["docker", "compose", "restart", "inference-service"],
        check=True,
        capture_output=True,
    )
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            status, _ = request_json(f"{base_url}/health")
            if status == 200:
                return
        except Exception:
            # Container may be mid-restart: connection refused, reset, or
            # any other transient socket error. Keep polling until timeout.
            pass
        time.sleep(1)
    raise RuntimeError("inference-service did not become healthy after restart.")


def generate_traffic(base_url: str, fault_mode: str, requests: int, interval: float = 0.05) -> None:
    status, result = request_json(f"{base_url}/admin/fault", "POST", {"mode": fault_mode})
    if status != 200:
        raise RuntimeError(f"Could not set fault mode: {result}")
    for number in range(1, requests + 1):
        request_json(
            f"{base_url}/predict",
            "POST",
            {
                "image_id": f"eval-harness/frame-{number:05d}.jpg",
                "brightness": 0.62,
                "sharpness": 0.81,
            },
        )
        time.sleep(interval)


def run_scenario(base_url: str, model: str, scenario: Scenario) -> Dict:
    print(f"\n=== Scenario: {scenario.name} (fault={scenario.fault_mode}, expected={scenario.expected_severity}) ===")
    restart_service_and_wait(base_url)
    generate_traffic(base_url, scenario.fault_mode, scenario.requests)

    final_state = run_investigation(base_url, model)
    actual_severity = final_state.get("severity", "unknown")
    passed = actual_severity == scenario.expected_severity

    print(f"Expected severity: {scenario.expected_severity} | Actual severity: {actual_severity} | "
          f"{'PASS' if passed else 'FAIL'}")

    with mlflow.start_run(run_name=f"eval-{scenario.name}", nested=True):
        mlflow.log_param("scenario", scenario.name)
        mlflow.log_param("fault_mode", scenario.fault_mode)
        mlflow.log_param("model", model)
        mlflow.log_param("expected_severity", scenario.expected_severity)
        mlflow.set_tag("actual_severity", actual_severity)
        mlflow.log_metric("passed", 1 if passed else 0)
        mlflow.log_text(final_state.get("diagnosis", ""), "diagnosis.txt")

    return {
        "scenario": scenario.name,
        "fault_mode": scenario.fault_mode,
        "expected": scenario.expected_severity,
        "actual": actual_severity,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AegisOps agent eval harness across simulated incidents.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--scenarios", type=int, default=len(SCENARIOS), help="How many scenarios to run (from the top of the list).")
    args = parser.parse_args()

    scenarios_to_run = SCENARIOS[: args.scenarios]
    results = []

    with mlflow.start_run(run_name=f"eval-harness-{int(time.time())}"):
        mlflow.log_param("num_scenarios", len(scenarios_to_run))
        mlflow.log_param("model", args.model)

        for scenario in scenarios_to_run:
            results.append(run_scenario(args.base_url, args.model, scenario))

        passed_count = sum(1 for r in results if r["passed"])
        accuracy = passed_count / len(results) if results else 0.0
        mlflow.log_metric("scenarios_run", len(results))
        mlflow.log_metric("scenarios_passed", passed_count)
        mlflow.log_metric("diagnosis_accuracy", accuracy)

    print("\n" + "=" * 60)
    print("EVAL HARNESS SUMMARY")
    print("=" * 60)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['scenario']:<35} expected={r['expected']:<8} actual={r['actual']}")
    print("-" * 60)
    print(f"Diagnosis accuracy: {passed_count}/{len(results)} ({accuracy:.1%})")


if __name__ == "__main__":
    main()
