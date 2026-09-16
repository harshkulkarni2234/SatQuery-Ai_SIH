"""Agent planner (Phase A5).

Pipeline: parse query -> intent -> deterministic input validation ->
compatibility -> registry lookup -> specialist selection -> parameter
configuration. Wraps the existing deterministic pieces (services/router.py
classification, services/compatibility.py, services/registry.py selection)
into a single ExecutionPlan the API can expose as-is in `metadata.plan`, so
what the frontend shows always matches what the backend actually decided —
never a separately-animated guess.

Deliberately keyword/rule-based (see services/router.py's own docstring on
this) — `IntentClassifier` below is a Protocol so a learned classifier could
be swapped in later without touching the rest of the pipeline, but no such
classifier exists yet and none of this depends on an internet LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol

from app.contracts import ExecutionPlan
from app.services.compatibility import (
    check_optical_sar_pair,
    check_single_image,
    check_temporal_pair,
)
from app.services.registry import RegistryEntry
from app.services.registry import select as select_specialist
from app.services.router import GROUNDING_TARGETS, classify_query

_COUNT_RE = re.compile(r"\bhow many\b", re.IGNORECASE)
_PRESENCE_RE = re.compile(
    r"\b(is there|are there|does the image contain|contains?|is any|present in the image|exist)\b",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(r"\b(compare|comparison|versus|vs\.?)\b", re.IGNORECASE)


class IntentClassifier(Protocol):
    """Pluggable intent classifier interface. `classify_query` (the existing
    deterministic keyword router) is the only implementation today; a future
    learned classifier could implement the same signature."""

    def __call__(self, query_text: str, images: list[dict]) -> dict: ...


def _default_intent_classifier(query_text: str, images: list[dict]) -> dict:
    return classify_query(query_text, images)


def _infer_question_type(query_text: str, task: Optional[str]) -> Optional[str]:
    if not query_text:
        return None
    if task == "GROUNDING":
        return "location"
    if task == "CHANGE_DETECTION":
        return "comparison" if _COMPARISON_RE.search(query_text) else "change"
    if _COUNT_RE.search(query_text):
        return "count"
    if _PRESENCE_RE.search(query_text):
        return "presence"
    return "description"


@dataclass
class PlanResult:
    plan: ExecutionPlan
    selected_entry: Optional[RegistryEntry]


def build_plan(
    query_text: str,
    images: list[dict],
    intent_classifier: IntentClassifier = _default_intent_classifier,
) -> PlanResult:
    """Build a full ExecutionPlan for *query_text* against *images*.

    images: list of dicts with modality, capture_date, is_georeferenced,
    crs, bounds_wgs84, resolution, transform, width, height (whatever each
    check actually needs — see services/compatibility.py and
    services/registry.py for which keys they read).
    """
    query_text = (query_text or "").strip()

    if not query_text:
        return PlanResult(
            plan=ExecutionPlan(
                task=None,
                validation_passed=False,
                validation_reason="Query cannot be empty.",
                suggestion="Try asking about visible land cover (e.g. water, vegetation) "
                "or, with two images, what changed between them.",
            ),
            selected_entry=None,
        )

    classification = intent_classifier(query_text, images)
    task = classification.get("task_classified")
    target = classification.get("grounding_target")
    question_type = _infer_question_type(query_text, task)

    if not classification["validation_passed"]:
        suggestion = _suggestion_for(classification, len(images))
        return PlanResult(
            plan=ExecutionPlan(
                task=task,
                target=target,
                question_type=question_type,
                validation_passed=False,
                validation_reason=classification["reason"],
                suggestion=suggestion,
            ),
            selected_entry=None,
        )

    compatibility = None
    if task == "CHANGE_DETECTION":
        compatibility = check_temporal_pair(images[0], images[1])
    elif task == "CROSS_MODAL":
        optical = next(i for i in images if i.get("modality") == "OPTICAL")
        sar = next(i for i in images if i.get("modality") == "SAR")
        compatibility = check_optical_sar_pair(optical, sar)
    elif task in ("VQA", "GROUNDING") and images:
        compatibility = check_single_image(images[0])

    if compatibility is not None and not compatibility.ok:
        return PlanResult(
            plan=ExecutionPlan(
                task=task,
                target=target,
                question_type=question_type,
                validation_passed=False,
                validation_reason=f"These images are not compatible for {task}.",
                suggestion="Check the compatibility report for the specific check that failed.",
                compatibility=compatibility,
            ),
            selected_entry=None,
        )

    entry, selection_reason, rejected = select_specialist(task, images)
    parameters = _whitelisted_parameters(task, target)

    plan = ExecutionPlan(
        task=task,
        target=target,
        question_type=question_type,
        requested_outputs=_requested_outputs(task),
        validation_passed=True,
        validation_reason=classification["reason"],
        compatibility=compatibility,
        selected_specialist_id=entry.spec.id if entry else None,
        selection_reason=selection_reason,
        rejected_specialists=rejected,
        parameters=parameters,
    )
    return PlanResult(plan=plan, selected_entry=entry)


def _suggestion_for(classification: dict, n_images: int) -> str:
    task = classification.get("task_classified")
    if task == "GROUNDING" and n_images != 1:
        return "Grounding needs exactly 1 image — remove the extra image or ask a different question."
    if task == "GROUNDING":
        return f"Try naming one of the supported targets: {', '.join(sorted(GROUNDING_TARGETS))}."
    if task == "CHANGE_DETECTION" and n_images != 2:
        return "Upload a second image captured at a different date to compare against."
    if n_images == 2 and task is None:
        return "Ask a change/comparison question, or upload one OPTICAL and one SAR image for cross-modal analysis."
    return "Rephrase the question or check the number and modalities of the uploaded images."


def _whitelisted_parameters(task: Optional[str], target: Optional[str]) -> dict:
    if task == "GROUNDING":
        return {"target": target}
    if task == "VQA":
        return {"grounding_target": target}
    return {}


def _requested_outputs(task: Optional[str]) -> list[str]:
    return {
        "VQA": ["answer", "visual_evidence"],
        "GROUNDING": ["answer", "boxes", "regions"],
        "CHANGE_DETECTION": ["answer", "mask", "overlay", "stats"],
        "CROSS_MODAL": ["answer", "per_modality_evidence"],
    }.get(task, ["answer"])
