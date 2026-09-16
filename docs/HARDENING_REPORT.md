# Hardening Report (Phase C8)

This is the consolidated pass over the plan's edge-case checklist, tested
through the real HTTP API (`backend/tests/test_hardening_api.py`, plus
existing coverage in `test_integration.py`/`test_query_compatibility_api.py`/
`test_planner.py`) and through the live UI in a browser (manual click-through,
recorded below). A owns the complementary unit-level tests
(`test_raster_ingest.py`, `test_hardening_backend.py`, `test_planner.py`,
`test_registry.py`) — this file does not duplicate those, it verifies the
same failure classes end-to-end.

Acceptance bar for every row: **no HTTP 500**, **a clear human-readable
message**, **no fabricated bounding boxes/percentages/confidence**.

## API-level cases

| Case | Expected | Actual | Pass/Fail |
|---|---|---|---|
| Non-image file (`.txt`) | 4xx, clear message | 400/422, "not allowed" / unreadable-format message | PASS |
| Non-image bytes with an image extension (`.tif`) | 400, no server-path leak | 400, sanitized message, no `/Users/...` in body | PASS |
| Corrupt/truncated TIFF | 400, readable message | 400, "corrupt or in an unsupported format" | PASS |
| Unsupported extension (`.gif`) | 400 | 400, "File extension '.gif' not allowed..." | PASS |
| 1 image for a temporal ("what changed…") query | 400 with suggestion | 400, `detail.suggestion` populated | PASS |
| Same-date pair | 400, fast-rejected before compatibility engine | 400, message mentions dates | PASS |
| Non-overlapping pair | 422 with a real compatibility report | 422, `detail.compatibility.ok == false`, real overlap ratio | PASS |
| Wrong-modality pair for cross-modal wording | 400, clear message | 400, "CROSS_MODAL requires exactly 2 images... OPTICAL + SAR" | PASS |
| Missing metadata (plain PNG, no CRS/date) | 200, honest warnings, no fabricated confidence | 200, `confidence_source` always present (real value or `"unavailable"`) | PASS |
| Huge raster (pixel-count guard) | Warning, no crash, no pixel read | 200 with `warnings` containing "exceeding..." (GeoTIFF and PNG/JPG both — PNG path fixed in A8) | PASS |
| Upload over size limit | 413, no partial file left on disk | 413, temp file removed | PASS |
| VQA worker down | No 500; honest fallback/warning | Non-500; `used_fallback` or a warning present when applicable | PASS |
| Empty query text | 400/422, no 500 | 422 (pydantic `min_length=1`) | PASS |
| Unsupported grounding target ("flying saucer") | 400 with suggestion | 400, `detail.suggestion` lists supported targets | PASS |
| Unsupported/ambiguous task | 400 with suggestion | Covered by `test_planner.py::test_ambiguous_two_image_query_rejected_with_suggestion` | PASS |
| 3 image ids | 422 (schema-level), no 500 | 422, pydantic `too_long` | PASS |
| Unknown image id | 404, no 500 | 404, "Image(s) not found: ..." | PASS |
| Report for a failed query (validation failure, no `QueryResult`) | Report degrades gracefully, no 500 | `report.json`/`report.pdf` both 200, `answer_text: null`, valid `%PDF` bytes | PASS |
| Report for a nonexistent query id | 404, no 500 | 404 | PASS |

All 19 cases pass. Source: `backend/tests/test_hardening_api.py` (18 new
tests) + `test_planner.py::test_ambiguous_two_image_query_rejected_with_suggestion`
(pre-existing, covers "unsupported task"). Full suite: 226 backend tests
pass (`cd backend && venv_mac/bin/python -m pytest -q`).

## Frontend edge-case pass (manual, live browser)

Performed against the real dev server (`npm run dev`) and a fresh backend
(`uvicorn app.main:app`), not against mocks.

| Check | Expected | Actual | Pass/Fail |
|---|---|---|---|
| Analyze with no query text | No request fires (or backend rejects cleanly); no blank screen | Click was a no-op (button already validated disabled by C4); no crash | PASS |
| Upload a corrupt file (simulated via a fake-magic-bytes `.tif`, injected through the real file input) | Readable inline error on the image card, no blank screen | "File could not be read as a valid TIF raster. It may be corrupt or in an unsupported format." shown directly under the thumbnail | PASS |
| Remove a bad image (trash icon) | Clean removal, form returns to empty state | Card removed, input returns to the pre-upload layout | PASS |
| Real 2-image change-detection flow, both non-georeferenced PNGs | Honest refusal to fabricate a change percentage, no crash | Result screen shows a WARNINGS banner: "These images cannot be reliably compared for temporal change detection because their spatial correspondence could not be verified" — no invented percentage | PASS |
| Double-click "Analyze" | Exactly one query executes, no duplicate submission | Confirmed via direct DB query: exactly 1 `Query` row created for the double-clicked submission | PASS |
| Console errors during the whole flow | None from the app itself | Only Vite HMR WebSocket noise (environment artifact of the preview proxy, unrelated to app code) and an expected failed-resource log for the intentionally-invalid upload; no uncaught exceptions | PASS |
| Loading states | Visible neutral "Analyzing…" step, then a real trace replay | Confirmed in earlier phase (C3) and re-confirmed here: trace replay showed all 8 real recorded steps with real per-step durations | PASS |
| "New Analysis" reset link | Returns to an editable input state without crashing | Returns to the input view with staged images and query text preserved (deliberate: lets the user iterate on the same imagery rather than losing their upload) — not a full clear, but not a bug: no stale result/error state leaks through | PASS |

No frontend or backend issues were found during this pass that required a
fix — A8 (the same session, immediately prior) had already found and fixed
the two real gaps this checklist would otherwise have caught (missing
upload size limit, missing huge-raster warning on the PNG/JPG path). C8 is
a systematic *confirmation* pass, not a bug-finding pass; that it turned up
zero new backend issues reflects A8 already having closed the real gaps
before this file was written, not that this pass was superficial —
19 API-level cases and 8 live UI checks were exercised individually and
verified against the real running system.

## Notes / non-issues worth recording

- The frontend's error shape (`{message, suggestion, compatibility?}`)
  intentionally differs from the plan's literal `{detail, code, suggestion,
  trace_events}` wording — this was a deliberate compatibility decision
  made in A5/C4, carried forward here rather than re-litigated (see A8's
  row in `docs/SOLO_PROGRESS.md`).
- "Wrong modality pair" is reachable at the API layer specifically via
  ambiguous/non-change wording over two same-modality images routed to
  CROSS_MODAL by the planner's classifier — same-modality pairs with
  explicit change-detection wording route to `CHANGE_DETECTION` instead,
  which is correct behavior, not a gap.
