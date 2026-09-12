"""Generate repeatable traffic for the AegisOps inference-service simulator."""

import argparse
import json
import time
from typing import Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(
    url: str, method: str = "GET", payload: Optional[Dict] = None
) -> Tuple[int, Dict]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate inference traffic and an optional simulated incident.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--fault", choices=["none", "latency", "errors", "low_confidence", "drift"], default="none")
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args()

    status, result = request_json(f"{args.base_url}/admin/fault", "POST", {"mode": args.fault})
    if status != 200:
        raise RuntimeError(f"Could not set fault mode: {result}")
    print(f"Fault mode: {result['active_fault']}")

    successes = 0
    failures = 0
    for number in range(1, args.requests + 1):
        status, result = request_json(
            f"{args.base_url}/predict",
            "POST",
            {
                "image_id": f"factory-camera-17/frame-{number:05d}.jpg",
                "brightness": 0.62,
                "sharpness": 0.81,
            },
        )
        if status == 200:
            successes += 1
        else:
            failures += 1
        time.sleep(args.interval)

    print(f"Completed {args.requests} requests: {successes} succeeded, {failures} failed.")


if __name__ == "__main__":
    main()
