#!/usr/bin/env python3
"""Score an output directory against the ground-truth table.

Lets a set of already-generated JSON files be graded without re-running extraction, which is
how the multi-model comparison in the README was produced (each model's run is scored from
disk rather than paying for the calls again).

    python tests/verify_output_dir.py data/output
    python tests/verify_output_dir.py /tmp/m_openai_gpt-4o-mini
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_accuracy import GROUND_TRUTH  # noqa: E402

_SKIP = {"qms_schema.json", "run_summary.json"}


def _slug(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")


def _resolve(payload, path: str):
    target = payload
    for part in path.split("."):
        if not isinstance(target, dict) or part not in target:
            return None
        target = target[part]
    return target


def _is_field(node) -> bool:
    return isinstance(node, dict) and "status" in node and "confidence" in node


def _check(payload, path: str, expectation) -> tuple:
    kind, expected = expectation
    node = _resolve(payload, path)
    if node is None and kind != "attr":
        return False, None

    if kind == "attr":
        actual = node.get("value") if _is_field(node) else node
        if hasattr(expected, "isoformat"):
            expected = expected.isoformat()
        if isinstance(expected, list) and isinstance(actual, list):
            return [float(x) for x in actual] == [float(x) for x in expected], actual
        return actual == expected, actual

    if not _is_field(node):
        return False, node
    if kind == "status":
        return node.get("status") == expected, node.get("status")

    actual = node.get("value")
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(actual) - float(expected)) < 0.01, actual
    if isinstance(expected, str) and isinstance(actual, str):
        return actual.strip().lower() == expected.strip().lower(), actual
    return actual == expected, actual


def main(output_dir: str) -> int:
    available = {}
    for name in os.listdir(output_dir):
        if name.endswith(".json") and name not in _SKIP:
            available[_slug(name)] = os.path.join(output_dir, name)

    total = correct = 0
    mode = None
    print(f"\nScoring {output_dir} against the ground-truth table")
    failures = []
    for file_name, expectations in GROUND_TRUTH.items():
        path = available.get(_slug(file_name))
        if not path:
            print(f"  {file_name[:52]:54}   MISSING")
            continue
        payload = json.load(open(path, encoding="utf-8"))
        mode = mode or payload.get("extraction", {}).get("mode")
        hits = 0
        for field_path, expectation in expectations.items():
            ok, actual = _check(payload, field_path, expectation)
            hits += int(ok)
            if not ok:
                failures.append(f"{file_name}: {field_path} expected "
                                f"{expectation[0]}={expectation[1]!r}, got {actual!r}")
        total += len(expectations)
        correct += hits
        print(f"  {file_name[:52]:54} {hits:3d}/{len(expectations):<3d} "
              f"{100.0 * hits / len(expectations):5.1f}%")

    model = payload.get("extraction", {}).get("llm_model")
    print(f"  {'TOTAL':54} {correct:3d}/{total:<3d} "
          f"{100.0 * correct / total if total else 0:5.1f}%")
    print(f"  mode={mode}  model={model}")
    for failure in failures:
        print(f"    MISMATCH {failure}")
    return 0 if correct == total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data/output"))
