import { useSpecialists } from "../../lib/useSpecialists.js";

// The live registry, read from GET /specialists. Kept deliberately simple:
// what each group can do, and how many ways it can do it. Raw ids stay in the
// tooltip for anyone who wants them.
const GROUPS = [
  { task: "VQA", label: "Answer questions" },
  { task: "GROUNDING", label: "Locate land cover" },
  { task: "CHANGE_DETECTION", label: "Compare two dates" },
  { task: "CROSS_MODAL", label: "Fuse optical and radar" },
];

const SHORT = {
  "vqa.smolvlm_base": "SmolVLM",
  "vqa.smolvlm_bigearthnet_lora_stage3": "BigEarthNet LoRA",
  "grounding.deterministic_cv": "HSV segmentation",
  "change.siamese_binary_cnn": "Siamese CNN",
  "change.deterministic_cv": "Pixel difference",
  "cross_modal.feature_fusion": "Feature fusion",
};

export default function RegistryPanel() {
  const specialists = useSpecialists();
  if (!specialists.length) return null;

  return (
    <aside className="panel registry-panel glass">
      <div className="panel-head">
        <h2 className="panel-title">What it can do</h2>
        <span className="panel-count">{specialists.length} specialists</span>
      </div>

      <ul className="registry-list">
        {GROUPS.map(({ task, label }) => {
          const items = specialists.filter((s) => s.task === task);
          if (!items.length) return null;
          return (
            <li key={task} className="registry-row">
              <span className="registry-task">{label}</span>
              <span className="registry-chips">
                {items.map((s) => (
                  <span
                    key={s.id}
                    className={`registry-chip chip-${s.kind}`}
                    title={`${s.id} (priority ${s.priority}${s.is_fallback ? ", fallback" : ""})`}
                  >
                    <span className="chip-dot" />
                    {SHORT[s.id] || s.name}
                  </span>
                ))}
              </span>
            </li>
          );
        })}
      </ul>

      <p className="panel-foot">
        One is chosen per question, by priority and a live health check. The
        result always names the one that actually ran.
      </p>
    </aside>
  );
}
