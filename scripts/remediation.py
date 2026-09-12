"""Create and approve auditable, human-controlled remediation requests."""

import argparse
import json
from typing import Dict
from urllib.request import Request, urlopen


def post_json(url: str, payload: Dict) -> Dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage AegisOps remediation approvals.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--action", choices=["route_to_last_known_healthy_model", "scale_inference_service", "schedule_data_quality_review"], required=True)
    create.add_argument("--requested-by", required=True)
    create.add_argument("--rationale", required=True)

    approve = subparsers.add_parser("approve")
    approve.add_argument("--request-id", required=True)
    approve.add_argument("--approved-by", required=True)

    args = parser.parse_args()
    if args.command == "create":
        result = post_json(
            f"{args.base_url}/admin/remediations",
            {"action": args.action, "requested_by": args.requested_by, "rationale": args.rationale},
        )
    else:
        result = post_json(
            f"{args.base_url}/admin/remediations/{args.request_id}/approve",
            {"approved_by": args.approved_by},
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
