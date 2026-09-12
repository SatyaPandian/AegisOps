"""Evidence collector and baseline investigator for an AegisOps incident."""

import argparse
import json
import re
import statistics
from typing import Dict, List, Tuple
from urllib.request import urlopen


def fetch_json(url: str) -> Dict:
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str) -> str:
    with urlopen(url, timeout=10) as response:
        return response.read().decode("utf-8")


def counter_value(metrics: str, metric_name: str, label_value: str) -> float:
    pattern = rf'{re.escape(metric_name)}\{{result="{label_value}"\}}\s+([0-9.eE+-]+)'
    match = re.search(pattern, metrics)
    return float(match.group(1)) if match else 0.0


def gauge_value(metrics: str, metric_name: str) -> float:
    match = re.search(rf'^{re.escape(metric_name)}\s+([0-9.eE+-]+)', metrics, flags=re.MULTILINE)
    return float(match.group(1)) if match else 0.0


def analyze(events: List[Dict], metrics: str) -> Tuple[str, List[str], List[str]]:
    error_events = [event for event in events if event["event"] == "prediction_failed"]
    completed_events = [event for event in events if event["event"] == "prediction_complete"]
    successful_requests = counter_value(metrics, "aegisops_inference_requests_total", "success")
    failed_requests = counter_value(metrics, "aegisops_inference_requests_total", "error")
    total_requests = successful_requests + failed_requests
    error_rate = failed_requests / total_requests if total_requests else 0.0
    confidences = [event["confidence"] for event in completed_events]
    average_confidence = statistics.mean(confidences) if confidences else 0.0
    max_latency = max((event["latency_seconds"] for event in completed_events), default=0.0)
    drift_score = gauge_value(metrics, "aegisops_input_drift_score")

    evidence = [
        f"{int(failed_requests)} failed requests out of {int(total_requests)} total ({error_rate:.1%} error rate).",
        f"{len(error_events)} structured error events were captured.",
        f"Average successful-prediction confidence: {average_confidence:.3f}.",
        f"Maximum observed successful-prediction latency: {max_latency:.3f} seconds.",
        f"Input drift score: {drift_score:.2f}.",
    ]

    if error_rate >= 0.20:
        diagnosis = "High inference error rate caused by an upstream model-service failure."
        recommendations = [
            "Page the model-serving owner and inspect upstream service health.",
            "Pause rollout or route traffic to the last known healthy model version after human approval.",
            "Keep the incident open until the error rate returns to the defined baseline.",
        ]
    elif max_latency >= 1.0:
        diagnosis = "Inference latency breach detected; investigate resource saturation or a slow upstream dependency."
        recommendations = [
            "Inspect CPU, memory, and downstream dependency latency.",
            "Consider temporarily scaling serving replicas after human approval.",
        ]
    elif average_confidence and average_confidence < 0.65:
        diagnosis = "Prediction-confidence degradation detected; investigate model quality and input quality."
        recommendations = [
            "Sample low-confidence inputs for annotation and human review.",
            "Compare the current model against the previous approved version.",
        ]
    elif drift_score >= 0.50:
        diagnosis = "Significant input distribution drift detected."
        recommendations = [
            "Collect a representative recent input sample for validation.",
            "Trigger a data-quality review before scheduling retraining.",
        ]
    else:
        diagnosis = "No material incident detected from the currently available evidence."
        recommendations = ["Continue monitoring; no automated action is recommended."]

    return diagnosis, evidence, recommendations


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigate current AegisOps service evidence.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    event_response = fetch_json(f"{args.base_url}/admin/events?limit=500")
    metrics = fetch_text(f"{args.base_url}/metrics")
    diagnosis, evidence, recommendations = analyze(event_response["events"], metrics)

    print("# AegisOps Incident Report")
    print(f"\n## Diagnosis\n{diagnosis}")
    print("\n## Evidence")
    for item in evidence:
        print(f"- {item}")
    print("\n## Human-approved next actions")
    for item in recommendations:
        print(f"- {item}")


if __name__ == "__main__":
    main()
