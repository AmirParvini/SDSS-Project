# -*- coding: utf-8 -*-
"""CLI entrypoint (alternative for Laravel via Symfony Process).

Reads the request JSON from stdin and writes the result JSON to stdout, so it
can be invoked as a subprocess. Note: each invocation reloads the graph from
the on-disk cache (no download), but rebuilds the KD-tree. For high throughput
prefer the HTTP service in api.py.

Usage:  echo '<input-json>' | python cli.py
"""
from __future__ import annotations

import json
import sys

from route_service import get_routing_service


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        json.dump({"error": f"invalid JSON input: {exc}"}, sys.stdout)
        return 1

    output = get_routing_service().compute_from_payload(payload)
    json.dump(output, sys.stdout, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
