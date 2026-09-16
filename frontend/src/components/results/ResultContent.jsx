import VQAResult from "./VQAResult.jsx";
import GroundingResult from "./GroundingResult.jsx";
import ChangeResult from "./ChangeResult.jsx";
import CrossModalResult from "./CrossModalResult.jsx";

export default function ResultContent({ result, images }) {
  const task = result.task_classified;
  if (task === "VQA") return <VQAResult result={result} images={images} />;
  if (task === "GROUNDING")
    return <GroundingResult result={result} images={images} />;
  if (task === "CHANGE_DETECTION")
    return <ChangeResult result={result} images={images} />;
  if (task === "CROSS_MODAL")
    return <CrossModalResult result={result} images={images} />;
  return (
    <div className="result-block">
      <div className="answer-text">
        {result.answer_text || "Analysis returned no answer."}
      </div>
    </div>
  );
}
