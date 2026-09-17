# Semantic Change Model Selection (Phase B8, item 1)

Researched via the official GitHub repos of each candidate (GitHub API +
README fetch, not secondhand summaries) on 2026-09-17. All three are
genuine, published, peer-reviewed remote-sensing change-detection models
trained on LEVIR-CD (and/or other CD benchmarks) — none of this is
invented or approximated.

## Candidates

| | **TinyCD** | **BIT_CD** | **ChangeFormer** |
|---|---|---|---|
| Repo | [AndreaCodegoni/Tiny_model_4_CD](https://github.com/AndreaCodegoni/Tiny_model_4_CD) | [justchenhao/BIT_CD](https://github.com/justchenhao/BIT_CD) | [wgcban/ChangeFormer](https://github.com/wgcban/ChangeFormer) |
| Paper | [TINYCD (arXiv:2207.13159)](https://arxiv.org/abs/2207.13159) | [Remote Sensing Image Change Detection with Transformers (arXiv:2103.00208)](https://arxiv.org/abs/2103.00208) | [A Transformer-Based Siamese Network for Change Detection, IGARSS'22](https://arxiv.org/pdf/2312.04869) |
| Architecture | Siamese CNN (U-Net-style, MAMB attention block) | ResNet backbone + Transformer (BIT) | SegFormer-based Siamese Transformer |
| **License** | **Non-commercial/research only** (stated in README; no LICENSE file, no SPDX id) | **Non-commercial/research only** (same pattern) | **MIT** (real SPDX license on the repo) |
| **Pretrained weights, real measured size** | `levir_best.pth` = **1.22 MB**, `whu_best.pth` = **1.22 MB** (checked via GitHub API — genuinely this small, the name is accurate) | Hosted on Google Drive / Baidu Drive only — no direct, scriptable download link; size not independently confirmed from the repo itself | `CD_ChangeFormerV6_LEVIR...zip` = **940 MB** (checked via GitHub Releases API) |
| Download mechanism | Direct file in the git repo (`pretrained_models/`) — reproducible via a plain `git clone` or GitHub API call, no manual step | External cloud-drive links, no API-checkable size/checksum, manual step required | GitHub Releases asset — scriptable, but genuinely large (same class of download as this project's other "ask before fetching" datasets) |
| Output | Binary change mask (single-channel probability, threshold 0.5) | Binary change mask | Binary change mask (multi-class variant exists in the repo but LEVIR-CD checkpoint is binary) |
| Input | Two co-sized RGB image tensors (before/after) | Two co-sized RGB image tensors | Two co-sized RGB image tensors |
| VRAM/CPU fit | Trivially small model — fits comfortably on CPU or any GPU including this project's verified Apple M2 via MPS | Larger than TinyCD (transformer + ResNet backbone) but still described as lightweight relative to full segmentation models | Heaviest of the three; SegFormer backbone, designed with a real GPU training budget in mind, not the "~4GB VRAM or CPU" target this phase asks for |
| Semantic vs binary | **Binary only** (change / no-change) — none of the three candidates output labeled semantic classes (e.g. "vegetation→built-up"); true semantic (multi-class) change detection with pretrained weights is not readily available in a small, open pretrained checkpoint |

## Recommendation: TinyCD

TinyCD is the only candidate that satisfies this phase's own stated
constraint ("fit ~4GB VRAM or CPU inference") without qualification, has
weights small enough (2.4 MB total for both checkpoints) that fetching
them is not a "large third-party download" in the sense this project's
process gates on, and has a clean, minimal, directly loadable
`torch.load(state_dict)` interface (`models/change_classifier.py` +
`models/layers.py`, no exotic dependencies beyond plain PyTorch).

**The license caveat must be stated plainly wherever this specialist's
output appears**: TinyCD's code and weights are released for
**non-commercial and research purposes only**, per its own README (no
formal LICENSE file, no SPDX identifier — this is a narrower, more
restrictive grant than a permissive OSS license like MIT). For a SIH
2026 research/hackathon submission this fits the project's own current
stance (`README.md`: "For SIH 2026 prototype/demo purposes. No license
has been selected yet.") — but this specialist specifically must not be
presented as part of any future commercial deployment without contacting
the original authors first, and this repo's own README/limitations
section must say so.

ChangeFormer's MIT license is strictly better on the license axis, but
its 940 MB checkpoint is squarely a "large third-party download" by this
project's own standard (the same class of decision that gated the
RSVQA/CDVQA/VRSBench datasets in Phase B3) — fetching it requires the
same explicit-permission step, for a model that is also heavier to run
than this phase's own "CPU-friendly" criterion prefers.

BIT_CD is not recommended primarily on reproducibility grounds: its
weights are hosted only on Google Drive/Baidu Drive with no scriptable,
checksummable download path, which conflicts with this project's general
preference for reproducible, verifiable data provenance (see
`data/manifest.json`'s citation discipline for the demo imagery).

## What this specialist can and cannot honestly claim

Per AGENTS.md's honesty rules (never claim pixel diff = semantic change,
never claim a deterministic tool is a trained model): TinyCD **is** a
real trained CNN — using it is a genuine upgrade from
`change.deterministic_cv`'s pixel/phase-correlation approach to a
learned-feature model. But its output is still **binary** change/no-change
per pixel, not a labeled semantic class (e.g. it cannot say "vegetation
was replaced by a building" — only "this region changed"). The registry
entry's existing id, `change.semantic_model`, predates this integration
(Phase A4) and is somewhat misleading in isolation; this specialist's own
`SpecialistResult` evidence and answer text must describe its output as
"learned binary change detection," never as semantic/labeled change
classification, to stay honest about what TinyCD actually produces.
