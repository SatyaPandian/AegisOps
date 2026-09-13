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

from agentic_graph import run_investigation, ollama_json, MLFLOW_DB_PATH  # noqa: F401  (reuses same tracking URI)

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
mlflow.set_experiment("aegisops_eval_harness")

JUDGE_SYSTEM_PROMPT = """You are grading an incident diagnosis written by another AI agent.
You were NOT given the original evidence: judge only whether the diagnosis text
below identifies the correct root cause. Be strict: a vague or generic diagnosis
that never names the actual cause should be marked incorrect.

Respond with valid JSON only, no markdown:
{"correct": true or false, "reasoning": "one short sentence"}
"""


@dataclass
class Scenario:
    name: str
    fault_mode: str
    expected_severity: str
    expected_root_cause: str
    requests: int = 20


SCENARIOS = [
    Scenario("baseline_healthy", "none", "low",
             "the system is healthy with no significant errors or drift"),
    Scenario("upstream_errors", "errors", "high",
             "a high rate of upstream/prediction request failures (simulated_upstream_error)"),
    Scenario("low_prediction_confidence", "low_confidence", "medium",
             "unusually low model prediction confidence"),
    Scenario("input_drift", "drift", "medium",
             "input data drift (the input drift score is elevated)"),
    Scenario("baseline_healthy_repeat", "none", "low",
             "the system is healthy with no significant errors or drift"),
    Scenario("upstream_errors_repeat", "errors", "high",
             "a high rate of upstream/prediction request failures (simulated_upstream_error)"),
    Scenario("low_prediction_confidence_repeat", "low_confidence", "medium",
             "unusually low model prediction confidence"),
    Scenario("input_drift_repeat", "drift", "medium",
             "input data drift (the input drift score is elevated)"),
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


def judge_diagnosis(model: str, diagnosis: str, expected_root_cause: str) -> Dict:
    """Independently grade the Reporter's free-text diagnosis, with no access
    to the original evidence, only the text and the expected root cause."""
    if not diagnosis.strip():
        return {"correct": False, "reasoning": "Diagnosis text was empty."}
    user_content = (
        f'Diagnosis to grade: "{diagnosis}"\n\n'
        f"Expected root cause: {expected_root_cause}\n\n"
        "Does the diagnosis correctly identify this root cause?"
    )
    try:
        return ollama_json(model, JUDGE_SYSTEM_PROMPT, user_content)
    except Exception as error:  # noqa: BLE001 - judge failures shouldn't crash the whole harness run
        return {"correct": False, "reasoning": f"Judge call failed: {error}"}


def run_scenario(base_url: str, model: str, scenario: Scenario) -> Dict:
    print(f"\n=== Scenario: {scenario.name} (fault={scenario.fault_mode}, expected={scenario.expected_severity}) ===")
    restart_service_and_wait(base_url)
    generate_traffic(base_url, scenario.fault_mode, scenario.requests)

    final_state = run_investigation(base_url, model)
    actual_severity = final_state.get("severity", "unknown")
    diagnosis = final_state.get("diagnosis", "")
    severity_correct = actual_severity == scenario.expected_severity

    judge_verdict = judge_diagnosis(model, diagnosis, scenario.expected_root_cause)
    diagnosis_correct = bool(judge_verdict.get("correct", False))

    print(f"Expected severity: {scenario.expected_severity} | Actual severity: {actual_severity} | "
          f"{'PASS' if severity_correct else 'FAIL'}")
    print(f"Diagnosis judged: {'CORRECT' if diagnosis_correct else 'INCORRECT'} "
          f"({judge_verdict.get('reasoning', 'no reasoning given')})")

    with mlflow.start_run(run_name=f"eval-{scenario.name}", nested=True):
        mlflow.log_param("scenario", scenario.name)
        mlflow.log_param("fault_mode", scenario.fault_mode)
        mlflow.log_param("model", model)
        mlflow.log_param("expected_severity", scenario.expected_severity)
        mlflow.log_param("expected_root_cause", scenario.expected_root_cause)
        mlflow.set_tag("actual_severity", actual_severity)
        mlflow.log_metric("severity_correct", 1 if severity_correct else 0)
        mlflow.log_metric("diagnosis_correct", 1 if diagnosis_correct else 0)
        mlflow.log_text(diagnosis, "diagnosis.txt")
        mlflow.log_text(judge_verdict.get("reasoning", ""), "judge_reasoning.txt")

    return {
        "scenario": scenario.name,
        "fault_mode": scenario.fault_mode,
        "expected_severity": scenario.expected_severity,
        "actual_severity": actual_severity,
        "severity_correct": severity_correct,
        "diagnosis_correct": diagnosis_correct,
        "judge_reasoning": judge_verdict.get("reasoning", ""),
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

        severity_correct_count = sum(1 for r in results if r["severity_correct"])
        diagnosis_correct_count = sum(1 for r in results if r["diagnosis_correct"])
        both_correct_count = sum(1 for r in results if r["severity_correct"] and r["diagnosis_correct"])
        total = len(results) if results else 1

        severity_accuracy = severity_correct_count / total
        diagnosis_accuracy = diagnosis_correct_count / total
        overall_accuracy = both_correct_count / total

        mlflow.log_metric("scenarios_run", len(results))
        mlflow.log_metric("severity_accuracy", severity_accuracy)
        mlflow.log_metric("diagnosis_accuracy", diagnosis_accuracy)
        mlflow.log_metric("overall_accuracy", overall_accuracy)

    print("\n" + "=" * 90)
    print("EVAL HARNESS SUMMARY")
    print("=" * 90)
    for r in results:
        sev_status = "PASS" if r["severity_correct"] else "FAIL"
        diag_status = "PASS" if r["diagnosis_correct"] else "FAIL"
        print(f"[severity:{sev_status}] [diagnosis:{diag_status}] {r['scenario']:<35} "
              f"expected={r['expected_severity']:<8} actual={r['actual_severity']}")
    print("-" * 90)
    print(f"Severity accuracy (policy-derived, checks the policy function against itself): "
          f"{severity_correct_count}/{len(results)} ({severity_accuracy:.1%})")
    print(f"Diagnosis accuracy (LLM-judge grading the Reporter's actual reasoning): "
          f"{diagnosis_correct_count}/{len(results)} ({diagnosis_accuracy:.1%})")
    print(f"Overall (both correct): {both_correct_count}/{len(results)} ({overall_accuracy:.1%})")


if __name__ == "__main__":
    main()
