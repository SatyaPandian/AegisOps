"""A free, local tool-using incident investigator powered by Ollama."""

import argparse
import json
import re
from typing import Any, Dict, List
from urllib.request import Request, urlopen


SYSTEM_PROMPT = """You are AegisOps, a careful ML incident-response investigator.
You must investigate an inference service using tools before reaching a conclusion.
Choose exactly one action in valid JSON with no markdown:

{"action":"get_recent_events","arguments":{"limit":50},"reason":"why this evidence is needed"}
{"action":"get_metrics_snapshot","arguments":{},"reason":"why this evidence is needed"}
{"action":"finish","diagnosis":"short diagnosis grounded only in tool evidence","severity":"low|medium|high","evidence":["specific observed fact"],"recommended_actions":["human-approved action"]}

Rules:
- Call at least two different evidence tools before action=finish.
- Never claim an action was executed. Recommendations always require human approval.
- Do not invent measurements, causes, or evidence.
- Keep your final diagnosis concise and operationally useful.
- In your final evidence list, quote exact values returned by tools; never use tool names as evidence.
"""


def fetch_json(url: str) -> Dict[str, Any]:
    with urlopen(url, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str) -> str:
    with urlopen(url, timeout=15) as response:
        return response.read().decode("utf-8")


def ollama_action(model: str, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    prompt = SYSTEM_PROMPT + "\n\nInvestigation transcript:\n" + "\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in conversation
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    request = Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        answer = json.loads(response.read().decode("utf-8"))["response"]
    return json.loads(answer)


def get_recent_events(base_url: str, limit: int) -> Dict[str, Any]:
    response = fetch_json(f"{base_url}/admin/events?limit={max(1, min(limit, 100))}")
    events = response["events"]
    failures = [event for event in events if event["event"] == "prediction_failed"]
    successes = [event for event in events if event["event"] == "prediction_complete"]
    fault_modes = sorted({event.get("fault") or event.get("mode") for event in events if event.get("fault") or event.get("mode")})
    return {
        "summary": {
            "recorded_events": len(events),
            "prediction_success_events": len(successes),
            "prediction_failure_events": len(failures),
            "fault_modes_observed": fault_modes,
            "failure_reasons": sorted({event.get("reason", "unknown") for event in failures}),
        },
        "recent_failure_samples": failures[-5:],
    }


def get_metrics_snapshot(base_url: str) -> Dict[str, Any]:
    raw_metrics = fetch_text(f"{base_url}/metrics")
    def metric(pattern: str) -> float:
        match = re.search(pattern, raw_metrics, flags=re.MULTILINE)
        return float(match.group(1)) if match else 0.0

    successes = metric(r'aegisops_inference_requests_total\{result="success"\}\s+([0-9.eE+-]+)')
    failures = metric(r'aegisops_inference_requests_total\{result="error"\}\s+([0-9.eE+-]+)')
    total = successes + failures
    return {
        "successful_requests": int(successes),
        "failed_requests": int(failures),
        "total_prediction_requests": int(total),
        "error_rate": round(failures / total, 4) if total else 0.0,
        "last_prediction_confidence": round(metric(r"^aegisops_last_prediction_confidence\s+([0-9.eE+-]+)"), 4),
        "input_drift_score": round(metric(r"^aegisops_input_drift_score\s+([0-9.eE+-]+)"), 4),
    }


def verified_evidence(tool_results: Dict[str, Dict[str, Any]]) -> List[str]:
    """Render facts directly from tools, independent of the model's prose."""
    lines = []
    events = tool_results.get("get_recent_events", {}).get("summary", {})
    metrics = tool_results.get("get_metrics_snapshot", {})
    if events:
        lines.append(
            f"Event stream: {events['prediction_failure_events']} failed and "
            f"{events['prediction_success_events']} successful prediction events; "
            f"fault modes observed: {', '.join(events['fault_modes_observed']) or 'none'}."
        )
        lines.append(f"Reported failure reason: {', '.join(events['failure_reasons']) or 'none'}.")
    if metrics:
        lines.append(
            f"Prometheus counters: {metrics['failed_requests']} failures / "
            f"{metrics['total_prediction_requests']} prediction requests "
            f"({metrics['error_rate']:.1%} error rate)."
        )
        lines.append(
            f"Last prediction confidence: {metrics['last_prediction_confidence']:.3f}; "
            f"input drift score: {metrics['input_drift_score']:.2f}."
        )
    return lines


def policy_severity(tool_results: Dict[str, Dict[str, Any]]) -> str:
    """Assign severity from measured thresholds, never from model prose."""
    metrics = tool_results.get("get_metrics_snapshot", {})
    if metrics.get("error_rate", 0.0) >= 0.20:
        return "high"
    if metrics.get("input_drift_score", 0.0) >= 0.50 or metrics.get("last_prediction_confidence", 1.0) < 0.65:
        return "medium"
    return "low"


def policy_actions(tool_results: Dict[str, Dict[str, Any]]) -> List[str]:
    """Recommend only actions relevant to evidence and require approval."""
    metrics = tool_results.get("get_metrics_snapshot", {})
    if metrics.get("error_rate", 0.0) >= 0.20:
        return [
            "Inspect upstream model-serving health and recent deployment changes.",
            "After human approval, consider routing traffic to the last known healthy model version.",
            "Keep the incident open until error rate returns to the defined baseline.",
        ]
    if metrics.get("input_drift_score", 0.0) >= 0.50:
        return [
            "Collect recent inputs for data-quality review before retraining.",
            "After human approval, schedule a model-quality evaluation on the reviewed sample.",
        ]
    return ["Continue monitoring; no remediation is recommended from current evidence."]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local AegisOps tool-using incident investigator.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="qwen2.5:3b")
    args = parser.parse_args()

    conversation: List[Dict[str, str]] = [
        {"role": "user", "content": "Investigate the current inference-service incident."}
    ]
    tools_used = set()
    tool_results: Dict[str, Dict[str, Any]] = {}

    for step in range(1, 6):
        action = ollama_action(args.model, conversation)
        action_name = action.get("action")

        if action_name == "get_recent_events":
            result = get_recent_events(args.base_url, int(action.get("arguments", {}).get("limit", 50)))
            tools_used.add(action_name)
        elif action_name == "get_metrics_snapshot":
            result = get_metrics_snapshot(args.base_url)
            tools_used.add(action_name)
        elif action_name == "finish" and len(tools_used) >= 2:
            print("# AegisOps Agentic Incident Report")
            print(f"\n**Severity:** {policy_severity(tool_results).upper()} (policy-derived)")
            print(f"\n## Agent diagnosis\n{action.get('diagnosis', 'No diagnosis returned.')}")
            print("\n## Verified tool evidence")
            for item in verified_evidence(tool_results):
                print(f"- {item}")
            print("\n## Human-approved next actions")
            for item in policy_actions(tool_results):
                print(f"- {item}")
            return
        else:
            result = {
                "error": "The requested action was invalid or premature.",
                "allowed_actions": ["get_recent_events", "get_metrics_snapshot", "finish after two tools"],
            }

        print(f"Agent step {step}: {action_name}")
        if action_name in {"get_recent_events", "get_metrics_snapshot"}:
            tool_results[action_name] = result
        conversation.append({"role": "assistant", "content": json.dumps(action)})
        conversation.append({"role": "tool", "content": json.dumps(result)})

    raise RuntimeError("Agent did not produce a final report after five steps.")


if __name__ == "__main__":
    main()
