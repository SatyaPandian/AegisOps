"""LangGraph-based incident investigation pipeline for AegisOps.

Replaces the hand-rolled loop in agentic_investigator.py with a real stateful
graph made of two specialized agents:

  investigator_node  -> chooses evidence tools via a local Ollama model,
                         calls them, and accumulates verified tool output.
  reporter_node       -> a SEPARATE Ollama call that only ever sees the
                         policy-computed facts (never raw model prose from
                         the investigator), and writes the final narrative.

Severity and recommended actions are still computed by deterministic policy
functions, never by either agent. This keeps the "don't trust the model as
ground truth" guardrail from the original design intact while making the
orchestration real instead of a single manual while-loop.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, TypedDict
from urllib.request import Request, urlopen

import mlflow
from langgraph.graph import StateGraph, END

MLFLOW_DB_PATH = Path(__file__).resolve().parent.parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
mlflow.set_experiment("aegisops_incident_investigations")


INVESTIGATOR_SYSTEM_PROMPT = """You are the AegisOps Investigator agent.
You must gather evidence about an inference service incident using tools
before handing off to the Reporter agent. Choose exactly one action in
valid JSON with no markdown:

{"action":"get_recent_events","arguments":{"limit":50},"reason":"why this evidence is needed"}
{"action":"get_metrics_snapshot","arguments":{},"reason":"why this evidence is needed"}
{"action":"handoff","reason":"why you have enough evidence now"}

Rules:
- Call at least two different evidence tools before action=handoff.
- Never claim an action was executed on production systems.
- Do not invent measurements, causes, or evidence.
"""

REPORTER_SYSTEM_PROMPT = """You are the AegisOps Reporter agent.
You did not gather evidence yourself. You may only use the verified facts
given to you below. Do not invent additional numbers, causes, or events.
Write a concise, operationally useful diagnosis (2-4 sentences) explaining
what is happening and why, grounded only in the facts provided. Respond
with plain text, not JSON.
"""


class InvestigationState(TypedDict, total=False):
    base_url: str
    model: str
    conversation: List[Dict[str, str]]
    tools_used: List[str]
    tool_results: Dict[str, Dict[str, Any]]
    verified_facts: List[str]
    severity: str
    recommended_actions: List[str]
    diagnosis: str
    step: int


def fetch_json(url: str) -> Dict[str, Any]:
    with urlopen(url, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str) -> str:
    with urlopen(url, timeout=15) as response:
        return response.read().decode("utf-8")


def ollama_json(model: str, system_prompt: str, user_content: str) -> Dict[str, Any]:
    prompt = f"{system_prompt}\n\n{user_content}"
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


def ollama_text(model: str, system_prompt: str, user_content: str) -> str:
    prompt = f"{system_prompt}\n\n{user_content}"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    request = Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))["response"].strip()


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
    """Render facts directly from tools, independent of any model's prose."""
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
    """Recommend only actions relevant to evidence; every action requires human approval."""
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


def investigator_node(state: InvestigationState) -> InvestigationState:
    conversation = state.get("conversation", [])
    tools_used = state.get("tools_used", [])
    tool_results = state.get("tool_results", {})
    step = state.get("step", 0) + 1

    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in conversation)
    user_content = f"Investigation transcript so far:\n{transcript}" if transcript else "No tools called yet."
    action = ollama_json(state["model"], INVESTIGATOR_SYSTEM_PROMPT, user_content)
    action_name = action.get("action")

    if action_name == "get_recent_events":
        result = get_recent_events(state["base_url"], int(action.get("arguments", {}).get("limit", 50)))
        tools_used = list(set(tools_used) | {action_name})
        tool_results = {**tool_results, action_name: result}
    elif action_name == "get_metrics_snapshot":
        result = get_metrics_snapshot(state["base_url"])
        tools_used = list(set(tools_used) | {action_name})
        tool_results = {**tool_results, action_name: result}
    elif action_name == "handoff" and len(tools_used) < 2:
        # Not enough evidence yet; treat as an invalid handoff attempt.
        result = {"error": "Cannot hand off yet: at least two distinct tools are required."}
    else:
        result = {"note": "Handoff accepted." if action_name == "handoff" else "Unrecognized action; try again."}

    new_conversation = conversation + [
        {"role": "assistant", "content": json.dumps(action)},
        {"role": "tool", "content": json.dumps(result)},
    ]

    return {
        **state,
        "conversation": new_conversation,
        "tools_used": tools_used,
        "tool_results": tool_results,
        "step": step,
    }


def route_after_investigator(state: InvestigationState) -> str:
    if state.get("step", 0) >= 6:
        return "reporter"
    last_message = state["conversation"][-2]["content"] if len(state.get("conversation", [])) >= 2 else "{}"
    try:
        last_action = json.loads(last_message).get("action")
    except (json.JSONDecodeError, AttributeError):
        last_action = None
    if last_action == "handoff" and len(state.get("tools_used", [])) >= 2:
        return "reporter"
    return "investigator"


def reporter_node(state: InvestigationState) -> InvestigationState:
    tool_results = state.get("tool_results", {})
    facts = verified_evidence(tool_results)
    severity = policy_severity(tool_results)
    actions = policy_actions(tool_results)

    facts_block = "\n".join(f"- {fact}" for fact in facts) or "- No verified facts were gathered."
    user_content = (
        f"Verified facts (policy-derived severity: {severity}):\n{facts_block}\n\n"
        "Write the diagnosis now."
    )
    diagnosis = ollama_text(state["model"], REPORTER_SYSTEM_PROMPT, user_content)

    return {
        **state,
        "verified_facts": facts,
        "severity": severity,
        "recommended_actions": actions,
        "diagnosis": diagnosis,
    }


def build_graph():
    graph = StateGraph(InvestigationState)
    graph.add_node("investigator", investigator_node)
    graph.add_node("reporter", reporter_node)
    graph.set_entry_point("investigator")
    graph.add_conditional_edges(
        "investigator",
        route_after_investigator,
        {"investigator": "investigator", "reporter": "reporter"},
    )
    graph.add_edge("reporter", END)
    return graph.compile()


def run_investigation(base_url: str, model: str) -> InvestigationState:
    app = build_graph()
    initial_state: InvestigationState = {
        "base_url": base_url,
        "model": model,
        "conversation": [],
        "tools_used": [],
        "tool_results": {},
        "step": 0,
    }

    with mlflow.start_run(run_name=f"investigation-{int(time.time())}", nested=True):
        mlflow.log_param("model", model)
        mlflow.log_param("base_url", base_url)

        started_at = time.perf_counter()
        final_state = app.invoke(initial_state)
        elapsed_seconds = time.perf_counter() - started_at

        metrics_snapshot = final_state.get("tool_results", {}).get("get_metrics_snapshot", {})
        mlflow.log_metric("tool_calls_made", len(final_state.get("tools_used", [])))
        mlflow.log_metric("investigator_steps", final_state.get("step", 0))
        mlflow.log_metric("run_duration_seconds", elapsed_seconds)
        mlflow.log_metric("observed_error_rate", metrics_snapshot.get("error_rate", 0.0))
        mlflow.log_metric("observed_drift_score", metrics_snapshot.get("input_drift_score", 0.0))
        mlflow.log_metric("observed_confidence", metrics_snapshot.get("last_prediction_confidence", 0.0))
        mlflow.set_tag("severity", final_state.get("severity", "unknown"))

        report_lines = [
            "# AegisOps Agentic Incident Report",
            f"\n**Severity:** {final_state.get('severity', 'unknown').upper()} (policy-derived)",
            f"\n## Reporter agent diagnosis\n{final_state.get('diagnosis', '')}",
            "\n## Verified tool evidence",
            *[f"- {item}" for item in final_state.get("verified_facts", [])],
            "\n## Human-approved next actions",
            *[f"- {item}" for item in final_state.get("recommended_actions", [])],
        ]
        mlflow.log_text("\n".join(report_lines), "incident_report.md")

    return final_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AegisOps LangGraph investigator/reporter pipeline.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="qwen2.5:3b")
    args = parser.parse_args()

    final_state = run_investigation(args.base_url, args.model)

    print("# AegisOps Agentic Incident Report")
    print(f"\n**Severity:** {final_state['severity'].upper()} (policy-derived)")
    print(f"\n## Reporter agent diagnosis\n{final_state['diagnosis']}")
    print("\n## Verified tool evidence")
    for item in final_state["verified_facts"]:
        print(f"- {item}")
    print("\n## Human-approved next actions")
    for item in final_state["recommended_actions"]:
        print(f"- {item}")


if __name__ == "__main__":
    main()
