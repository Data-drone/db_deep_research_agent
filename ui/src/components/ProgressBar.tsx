import type { JobStatus } from "../types";

const NODE_LABELS: Record<string, string> = {
  clarifier: "Clarifying query",
  planner: "Planning research",
  authorizer: "Checking permissions",
  researcher: "Gathering evidence",
  normalizer: "Normalizing data",
  evaluator: "Evaluating sufficiency",
  compressor: "Compressing findings",
  synthesizer: "Synthesizing answer",
  verifier: "Verifying citations",
};

const NODE_ORDER = [
  "clarifier",
  "planner",
  "authorizer",
  "researcher",
  "normalizer",
  "evaluator",
  "compressor",
  "synthesizer",
  "verifier",
];

interface Props {
  job: JobStatus;
  onCancel: () => void;
}

export function ProgressBar({ job, onCancel }: Props) {
  const currentNode = job.current_node;
  const currentIdx = currentNode ? NODE_ORDER.indexOf(currentNode) : -1;

  const label = currentNode
    ? NODE_LABELS[currentNode] || currentNode
    : job.status === "pending"
      ? "Starting..."
      : "Processing...";

  const progress =
    currentIdx >= 0 ? ((currentIdx + 1) / NODE_ORDER.length) * 100 : 10;

  return (
    <div className="progress-bar-container">
      <div className="progress-header">
        <span className="progress-label">{label}</span>
        <button className="cancel-btn" onClick={onCancel} title="Cancel">
          ✕
        </button>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <div className="progress-steps">
        {NODE_ORDER.map((node, idx) => (
          <span
            key={node}
            className={`step ${idx < currentIdx ? "done" : idx === currentIdx ? "active" : ""}`}
            title={NODE_LABELS[node]}
          >
            •
          </span>
        ))}
      </div>
    </div>
  );
}
