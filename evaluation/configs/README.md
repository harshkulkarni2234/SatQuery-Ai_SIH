# Evaluation configs

Reserved for future labeled-benchmark run configs (dataset paths, which
specialists/model versions to compare, sample sizes, seeds) once a real
labeled benchmark (RSVQA/CDVQA/VRSBench, per the original plan's B4–B6) is
actually fetched with explicit permission.

Nothing lives here yet: the only run this evaluation framework has
actually executed (`runners/run_testing_folder_eval.py`, see the top-level
`evaluation/README.md`) takes its parameters as CLI flags, since it has no
labeled splits/configs to select between — see
`runners/testing_folder_adapter.py`'s module docstring for why.
