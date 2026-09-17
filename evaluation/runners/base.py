"""Generic evaluation runner: iterates dataset-adapter samples, drives a
running SatQuery backend over its real HTTP API (POST /images/upload +
POST /query — the same endpoints the frontend uses, never an internal
shortcut), and records what actually happened.

Deviation from the plan's stated preference: the plan lists in-process
(via the planner/registry directly) as "preferred, faster", with the HTTP
API as the fallback option. This runner uses the HTTP API path instead —
deliberately: it needs no backend import path hacks or shared virtualenv
between evaluation/ and backend/, it exercises the exact same code path a
real user request takes (including the global exception handler and
upload-size checks added in Phase A8), and it can evaluate against a
backend running anywhere (this machine, a teammate's, CI), not just
in-process. The tradeoff is real HTTP overhead per sample, which is
immaterial next to VQA worker/CV inference time.

A "sample" is a plain dict:
    {
        "id": str,                     # stable, unique across runs
        "images": [                    # 1 or 2 entries
            {"path": str, "modality": "OPTICAL"|"SAR", "capture_date": str|None}
        ],
        "query_text": str,
        "meta": dict,                  # anything the adapter wants preserved
                                        # (ground truth, question type, etc.)
                                        # — copied into the result verbatim,
                                        # never used to fabricate a metric here.
    }

A dataset adapter is any Python generator/iterable yielding such dicts.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

import requests


@dataclass
class RunConfig:
    api_base_url: str = "http://127.0.0.1:8000"
    output_path: str = "evaluation/results/run.jsonl"
    limit: Optional[int] = None
    resume: bool = True
    request_timeout_s: float = 180.0


@dataclass
class SampleResult:
    sample_id: str
    query_text: str
    image_ids: list[str] = field(default_factory=list)
    task_classified: Optional[str] = None
    specialist_id: Optional[str] = None
    model_version: Optional[str] = None
    confidence_score: Optional[float] = None
    confidence_source: Optional[str] = None
    used_fallback: Optional[bool] = None
    warnings: list[str] = field(default_factory=list)
    answer_text: Optional[str] = None
    latency_ms: Optional[int] = None
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)
    timestamp: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, default=str)


def _already_done_ids(output_path: str) -> set[str]:
    if not os.path.isfile(output_path):
        return set()
    done = set()
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "sample_id" in row:
                done.add(row["sample_id"])
    return done


def _upload_image(api_base_url: str, path: str, modality: str, capture_date: Optional[str], timeout_s: float) -> str:
    data = {"modality": modality}
    if capture_date:
        data["capture_date"] = capture_date
    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f)}
        resp = requests.post(f"{api_base_url}/images/upload", data=data, files=files, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()["image_id"]


def run_evaluation(samples: Iterable[dict], config: RunConfig) -> list[SampleResult]:
    """Runs `samples` through a live SatQuery backend, appending one JSON
    line per sample to `config.output_path` as it goes (so a crash mid-run
    loses at most the in-flight sample, not the whole run). Resumable:
    samples whose id is already in the output file are skipped."""
    os.makedirs(os.path.dirname(config.output_path) or ".", exist_ok=True)
    done_ids = _already_done_ids(config.output_path) if config.resume else set()

    results: list[SampleResult] = []
    processed = 0
    mode = "a" if config.resume else "w"
    with open(config.output_path, mode) as out_f:
        for sample in samples:
            if config.limit is not None and processed >= config.limit:
                break
            sample_id = sample["id"]
            if sample_id in done_ids:
                continue

            result = SampleResult(
                sample_id=sample_id,
                query_text=sample["query_text"],
                meta=sample.get("meta", {}),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            try:
                image_ids = [
                    _upload_image(
                        config.api_base_url,
                        img["path"],
                        img["modality"],
                        img.get("capture_date"),
                        config.request_timeout_s,
                    )
                    for img in sample["images"]
                ]
                result.image_ids = image_ids

                t0 = time.perf_counter()
                resp = requests.post(
                    f"{config.api_base_url}/query",
                    json={"query_text": sample["query_text"], "image_ids": image_ids},
                    timeout=config.request_timeout_s,
                )
                result.latency_ms = int((time.perf_counter() - t0) * 1000)

                if resp.status_code >= 400:
                    body = resp.json()
                    detail = body.get("detail", body)
                    result.error = f"HTTP {resp.status_code}: {detail}"
                else:
                    body = resp.json()
                    result.task_classified = body.get("task_classified")
                    result.answer_text = body.get("answer_text")
                    result.confidence_score = body.get("confidence_score")
                    result.confidence_source = body.get("confidence_source")
                    result.used_fallback = body.get("used_fallback")
                    result.warnings = body.get("warnings") or []
                    meta_block = body.get("metadata") or {}
                    result.specialist_id = meta_block.get("specialist_id")
                    result.model_version = (body.get("execution_trace") or {}).get("model_version")
            except Exception as exc:  # noqa: BLE001 — record and continue, never abort the whole run
                result.error = f"{type(exc).__name__}: {exc}"

            out_f.write(result.to_json() + "\n")
            out_f.flush()
            results.append(result)
            processed += 1

    return results


def load_results(output_path: str) -> list[dict]:
    rows = []
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
